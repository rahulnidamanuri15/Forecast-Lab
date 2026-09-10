import os
import math
import numpy as np
import psycopg
from dotenv import load_dotenv

from vericast import (
    acquire_pipeline_lock,
    reopen_revised_actuals,
    require_city_of_record,
    require_database_url,
    revision_sql,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# model_performance has no city column, so CITY is constrained to Nagpur.
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))

# Attach arriving ground-truth observations to pending predictions (source='daily').
# NaN/Infinity guards ensure non-finite PostgreSQL floats are never attached as
# actuals: scoring against them yields NaN metrics that abort the transaction.
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
  AND o.pm2_5 != 'Infinity'::float
  AND o.pm2_5 != '-Infinity'::float
RETURNING p.forecast_date, p.model, p.predicted_pm2_5, o.pm2_5;
"""

# Upsert daily model performance metrics, ensuring source='daily' on conflict.
# Writes city so the multi-city UNIQUE(city, score_date, model) index stays
# correct; ON CONFLICT stays on the legacy (score_date, model) key so this works
# both before and after the schema.py city migration has run.
UPSERT_PERF_SQL = """
INSERT INTO model_performance (city, score_date, model, mae, rmse, sample_size)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (score_date, model) DO UPDATE SET
    city = EXCLUDED.city,
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
            # Serialize overlapping scorers; released on COMMIT/ROLLBACK.
            acquire_pipeline_lock(cur, "pm25_score")
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
            def _non_finite(v):
                return isinstance(v, float) and (math.isnan(v) or math.isinf(v))

            groups = {}
            for forecast_date, model, predicted, actual in scored:
                if predicted is None or actual is None:
                    continue
                if _non_finite(predicted):
                    print(f"  [skip] {forecast_date} {model}: predicted is non-finite; not scoring")
                    continue
                if _non_finite(actual):
                    print(f"  [skip] {forecast_date} {model}: actual is non-finite; not scoring")
                    continue
                groups.setdefault((forecast_date, model), []).append((predicted, actual))

            # Per-group SAVEPOINTs: one corrupt day skips itself instead of rolling
            # back every valid day scored in this run.
            skipped = 0
            for (score_date, model), pairs in sorted(groups.items()):
                cur.execute("SAVEPOINT score_day")
                try:
                    predicted = np.array([p for p, _ in pairs], dtype=float)
                    actual = np.array([a for _, a in pairs], dtype=float)
                    mae = float(np.mean(np.abs(predicted - actual)))
                    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

                    if not (math.isfinite(mae) and math.isfinite(rmse)):
                        raise ValueError(f"non-finite metric (mae={mae}, rmse={rmse})")

                    cur.execute(UPSERT_PERF_SQL, (CITY, score_date, model, mae, rmse, len(pairs)))
                    cur.execute("RELEASE SAVEPOINT score_day")
                    print(f"Scored {model} for {score_date}: MAE={mae:.4f}, RMSE={rmse:.4f} (n={len(pairs)})")
                except Exception as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT score_day")
                    cur.execute("RELEASE SAVEPOINT score_day")
                    skipped += 1
                    print(f"  [skip] {score_date} {model}: {exc}; day skipped, rest committed")
            if skipped:
                print(f"  [warn] {skipped} day(s) skipped for non-finite metrics; "
                      f"valid days committed normally.")

            conn.commit()

    print(f"Scored {len(scored)} prediction(s)"
          + (f", {reopened} of them a rescore after an upstream revision."
             if reopened else "."))
    return len(scored)


if __name__ == "__main__":
    score_pending_predictions()
