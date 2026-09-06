import os
import math
import numpy as np
import psycopg
from dotenv import load_dotenv

from vericast import reopen_revised_actuals, require_city_of_record, require_database_url, revision_sql

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# model_performance has no city column, so CITY is constrained to Nagpur.
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))

# Attach arriving ground-truth observations to pending predictions (source='daily').
# `o.pm2_5 != 'NaN'::float` ensures PostgreSQL NaN values are not attached as actuals.
SCORE_SQL = """
UPDATE predictions p
SET actual_pm2_5 = o.pm2_5
FROM observations o
WHERE o.city = p.city
  AND o.as_of = p.forecast_date
  AND p.city = %s
  AND p.source = 'daily'
  AND p.actual_pm2_5 IS NULL
  AND p.predicted_pm2_5 IS NOT NULL
  AND o.pm2_5 IS NOT NULL
  AND o.pm2_5 != 'NaN'::float
RETURNING p.forecast_date, p.model, p.predicted_pm2_5, o.pm2_5;
"""

# Upsert daily model performance metrics, ensuring source='daily' on conflict.
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

# Re-score query pair: finds and clears actuals for rows whose upstream observation changed.
DIVERGED_SQL, REOPEN_SQL = revision_sql(
    "predictions", "observations", "city",
    actual="actual_pm2_5", predicted="predicted_pm2_5", observed="pm2_5")


def score_pending_predictions():
    """Fill in actuals for any pending predictions and record per-day MAE/RMSE.

    Re-opens revised days first, in the same transaction, so a day whose ground truth
    moved upstream is re-scored here rather than keeping an error computed against a
    value the observations table no longer holds.
    """
    require_database_url(DATABASE_URL)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            reopened = reopen_revised_actuals(
                cur, DIVERGED_SQL, REOPEN_SQL, CITY, "ug/m3")

            cur.execute(SCORE_SQL, (CITY,))
            scored = cur.fetchall()

            # Every row REOPEN_SQL cleared satisfies SCORE_SQL's predicates by
            # construction (same source, same NOT NULL prediction, same NOT NULL
            # observation) and this is one transaction, so all of them must come back
            # scored. Raised, not asserted: python -O erases `assert`, and the failure
            # this stands in front of is committing an erased actual - a published row
            # silently returning to unverified. Nothing is committed yet, so psycopg
            # rolls the clear back on the way out.
            if reopened > len(scored):
                raise RuntimeError(
                    f"{reopened} row(s) were re-opened for rescoring but only "
                    f"{len(scored)} row(s) scored; refusing to commit an erased actual")

            if not scored:
                print("No pending predictions had an actual observation available.")
                return 0

            # Group by (forecast_date, model) - one perf row per day per model,
            # matching the UNIQUE(score_date, model) constraint.
            groups = {}
            for forecast_date, model, predicted, actual in scored:
                if predicted is None or actual is None:
                    continue
                if isinstance(predicted, float) and math.isnan(predicted):
                    print(f"  [skip] {forecast_date} {model}: predicted is NaN; not scoring")
                    continue
                if isinstance(actual, float) and math.isnan(actual):
                    print(f"  [skip] {forecast_date} {model}: actual is NaN; not scoring")
                    continue
                groups.setdefault((forecast_date, model), []).append((predicted, actual))

            for (score_date, model), pairs in sorted(groups.items()):
                predicted = np.array([p for p, _ in pairs], dtype=float)
                actual = np.array([a for _, a in pairs], dtype=float)
                mae = float(np.mean(np.abs(predicted - actual)))
                rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

                if math.isnan(mae) or math.isnan(rmse):
                    raise RuntimeError(
                        f"Computed metric is NaN for {score_date} {model}; "
                        "aborting transaction to avoid committing unmeasured actuals"
                    )

                cur.execute(UPSERT_PERF_SQL, (score_date, model, mae, rmse, len(pairs)))
                print(f"Scored {model} for {score_date}: MAE={mae:.4f}, RMSE={rmse:.4f} (n={len(pairs)})")

            conn.commit()

    print(f"Scored {len(scored)} prediction(s)"
          + (f", {reopened} of them a rescore after an upstream revision."
             if reopened else "."))
    return len(scored)


if __name__ == "__main__":
    score_pending_predictions()
