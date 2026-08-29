import os
import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

# Attach the actual observation to every prediction still waiting for one and
# report what changed. Deliberately not scoped to "yesterday": if a day's run
# was missed, that row would otherwise stay pending forever. Driven purely off
# forecast_date == observations.as_of, so there is no timezone decision here.
#
# predicted_pm2_5 IS NOT NULL because that column is nullable: scoring a NULL
# prediction makes np.mean below return NaN, and a NaN mae in model_performance
# 500s /leaderboard on JSON serialisation until the row is deleted by hand.
# vericast/pm25/predict.py no longer writes one, but a row from before that
# guard must not be scored either.
SCORE_SQL = """
UPDATE predictions p
SET actual_pm2_5 = o.pm2_5
FROM observations o
WHERE o.city = p.city
  AND o.as_of = p.forecast_date
  AND p.city = %s
  AND p.actual_pm2_5 IS NULL
  AND p.predicted_pm2_5 IS NOT NULL
  AND o.pm2_5 IS NOT NULL
RETURNING p.forecast_date, p.model, p.predicted_pm2_5, o.pm2_5;
"""

# The INSERT does not name `source`: vericast/schema.py declares
# DEFAULT 'daily', so the honest writer never has to. The DO UPDATE branch does
# have to - a DEFAULT applies to inserts only, so a daily score landing on a
# score_date a backtest already wrote would keep source='backtest' and stay
# filtered out of /leaderboard forever.
UPSERT_PERF_SQL = """
INSERT INTO model_performance (score_date, model, mae, rmse, sample_size)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (score_date, model) DO UPDATE SET
    mae = EXCLUDED.mae,
    rmse = EXCLUDED.rmse,
    sample_size = EXCLUDED.sample_size,
    source = 'daily',
    created_at = CURRENT_TIMESTAMP;
"""


def score_pending_predictions():
    """Fill in actuals for any pending predictions and record per-day MAE/RMSE."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(SCORE_SQL, (CITY,))
            scored = cur.fetchall()

            if not scored:
                print("No pending predictions had an actual observation available.")
                return 0

            # Group by (forecast_date, model) - one perf row per day per model,
            # matching the UNIQUE(score_date, model) constraint.
            groups = {}
            for forecast_date, model, predicted, actual in scored:
                groups.setdefault((forecast_date, model), []).append((predicted, actual))

            for (score_date, model), pairs in sorted(groups.items()):
                predicted = np.array([p for p, _ in pairs], dtype=float)
                actual = np.array([a for _, a in pairs], dtype=float)
                mae = float(np.mean(np.abs(predicted - actual)))
                rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

                cur.execute(UPSERT_PERF_SQL, (score_date, model, mae, rmse, len(pairs)))
                print(f"Scored {model} for {score_date}: MAE={mae:.4f}, RMSE={rmse:.4f} (n={len(pairs)})")

            conn.commit()

    print(f"Scored {len(scored)} prediction(s).")
    return len(scored)


if __name__ == "__main__":
    score_pending_predictions()
