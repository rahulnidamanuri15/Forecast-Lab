"""Attach actuals to pending electricity predictions and record per-day metrics.

Same shape as vericast/pm25/score.py - not scoped to "yesterday", so a missed run
self-heals - plus MAPE, the metric that actually travels for demand: 400 MW of
error means something different at 20,000 MW than a PM2.5 error of 400 would.
"""
import os
import math
import numpy as np
import psycopg
from dotenv import load_dotenv

from vericast import reopen_revised_actuals, require_database_url, revision_sql

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

# Attach arriving ground-truth observations to pending predictions.
# Scoped to source='daily' and non-null predictions to preserve backtest provenance.
SCORE_SQL = """
UPDATE electricity_predictions p
SET actual_demand_mw = o.peak_demand_mw
FROM electricity_observations o
WHERE o.state = p.state
  AND o.as_of = p.forecast_date
  AND p.state = %s
  AND p.source = 'daily'
  AND p.actual_demand_mw IS NULL
  AND p.predicted_demand_mw IS NOT NULL
  AND o.peak_demand_mw IS NOT NULL
  AND o.peak_demand_mw != 'NaN'::float
RETURNING p.forecast_date, p.model, p.predicted_demand_mw, o.peak_demand_mw;
"""

# Upsert daily model performance metrics, ensuring source='daily' on conflict.
UPSERT_PERF_SQL = """
INSERT INTO electricity_model_performance
    (state, score_date, model, mae, rmse, mape, sample_size)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (state, score_date, model) DO UPDATE SET
    mae = EXCLUDED.mae,
    rmse = EXCLUDED.rmse,
    mape = EXCLUDED.mape,
    sample_size = EXCLUDED.sample_size,
    source = 'daily',
    created_at = CURRENT_TIMESTAMP;
"""

# Re-score query pair: finds and clears actuals for rows whose upstream observation changed.
DIVERGED_SQL, REOPEN_SQL = revision_sql(
    "electricity_predictions", "electricity_observations", "state",
    actual="actual_demand_mw", predicted="predicted_demand_mw",
    observed="peak_demand_mw")


def score_pending_predictions():
    """Fill in actuals for any pending predictions and record per-day MAE/RMSE/MAPE.

    Re-opens revised days first, in the same transaction, so a day the mirror corrected
    is re-scored rather than keeping an error computed against the old value.
    """
    require_database_url(DATABASE_URL)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            reopened = reopen_revised_actuals(
                cur, DIVERGED_SQL, REOPEN_SQL, STATE, "MW")

            cur.execute(SCORE_SQL, (STATE,))
            scored = cur.fetchall()

            # See the PM2.5 twin: a re-opened row satisfies every SCORE_SQL predicate
            # by construction, so fewer scored rows than re-opened ones means an actual
            # would be committed erased. Nothing is committed yet.
            if reopened > len(scored):
                raise RuntimeError(
                    f"{reopened} row(s) were re-opened for rescoring but only "
                    f"{len(scored)} row(s) scored; refusing to commit an erased actual")

            if not scored:
                print("No pending electricity predictions had an actual available.")
                return 0

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
                # MAPE is undefined at actual == 0. Real peak demand is ~20-32 GW so
                # this never fires, but a bad upstream parse landing a 0 shouldn't
                # take the scoring step down.
                nonzero = actual != 0
                mape = (float(np.mean(np.abs((predicted[nonzero] - actual[nonzero])
                                             / actual[nonzero])) * 100)
                        if nonzero.any() else None)

                if math.isnan(mae) or math.isnan(rmse) or (mape is not None and math.isnan(mape)):
                    raise RuntimeError(
                        f"Computed metric is NaN for {score_date} {model}; "
                        "aborting transaction to avoid committing unmeasured actuals"
                    )

                cur.execute(UPSERT_PERF_SQL,
                            (STATE, score_date, model, mae, rmse, mape, len(pairs)))
                mape_str = f"{mape:.2f}%" if mape is not None else "n/a"
                print(f"Scored {model} for {score_date}: MAE={mae:.2f} MW, "
                      f"RMSE={rmse:.2f} MW, MAPE={mape_str} (n={len(pairs)})")

            conn.commit()

    print(f"Scored {len(scored)} electricity prediction(s)"
          + (f", {reopened} of them a rescore after an upstream revision."
             if reopened else "."))
    return len(scored)


if __name__ == "__main__":
    score_pending_predictions()
