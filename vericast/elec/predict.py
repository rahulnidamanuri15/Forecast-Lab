"""Publish tomorrow's Maharashtra peak demand forecast for three models.

The forecast is always labelled "the day after the data we actually have", not
blindly real-world tomorrow - same rule as vericast/pm25/predict.py, and it matters
more here: the upstream demand mirror runs 2-4 days behind, so a stale-data run
is the normal case rather than an incident. The staleness WARN threshold is
therefore looser than PM2.5's.

Three models compete: naive_baseline (persistence), seasonal_naive (same weekday
last week, which in a power grid beats persistence on Sundays), and lightgbm.

Every publish path is gated *before* its commit: staleness, then plausibility.
diagnose.py (step 6/6) still range-checks what was written, but it cannot be the
only gate - by the time it runs, a bad row is public.
"""
import os
import psycopg
import lightgbm as lgb
from datetime import timedelta
from dotenv import load_dotenv

from vericast import (
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    ELEC_STALE_LIMIT_DAYS,
    MODEL_ELEC as MODEL_PATH,
    local_time,
    refuse_implausible,
    refuse_stale,
)
from vericast.elec.train import FEATURE_COLUMNS

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

UNIT = "MW"

# `source` forced in the DO UPDATE for the reason vericast/pm25/predict.py's
# upsert_sql explains: schema.py's DEFAULT 'daily' applies to inserts only.
UPSERT_SQL = """
INSERT INTO electricity_predictions (state, forecast_date, predicted_demand_mw, model)
VALUES (%s, %s, %s, %s)
ON CONFLICT (state, forecast_date, model) DO UPDATE SET
    predicted_demand_mw = EXCLUDED.predicted_demand_mw,
    source = 'daily',
    created_at = CURRENT_TIMESTAMP;
"""


def load_lightgbm_model():
    """Load the production artifact. Returns None if it hasn't been trained yet,
    so the pipeline degrades to baselines-only instead of crashing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No LightGBM model artifact found at {MODEL_PATH}; "
              f"skipping LightGBM forecast. Run vericast/elec/train.py first.")
        return None
    return lgb.Booster(model_file=MODEL_PATH)


def get_latest_feature_row(cur):
    """Most recent electricity_features row. Its as_of should equal the latest
    observation's, since features are engineered same-day."""
    cur.execute(f"""
        SELECT as_of, {", ".join(FEATURE_COLUMNS)}
        FROM electricity_features
        WHERE state = %s
        ORDER BY as_of DESC
        LIMIT 1
    """, (STATE,))
    return cur.fetchone()


def make_daily_prediction():
    """Predict peak demand for the day after the latest observation and store it."""
    # Connection as context manager, as in score.py: a raise anywhere below
    # closes it instead of leaking it against Neon's connection limit.
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        lgbm_model = load_lightgbm_model()

        today = local_time.today()

        cur.execute("""
            SELECT as_of, peak_demand_mw
            FROM electricity_observations
            WHERE state = %s
            ORDER BY as_of DESC
            LIMIT 1
        """, (STATE,))

        row = cur.fetchone()
        if not row:
            raise Exception("No electricity observations found")

        as_of, peak_demand_mw = row
        forecast_date = as_of + timedelta(days=1)

        # Raises past the limit, before anything is published. 2-4 days is the
        # normal lag for this mirror, so only past ELEC_STALE_LIMIT_DAYS has it
        # actually stalled.
        stale_days = refuse_stale(as_of, today, ELEC_STALE_LIMIT_DAYS, "electricity")
        print(f"Latest observation {as_of} is {stale_days} day(s) old "
              f"(normal for this source).")

        print(f"Making prediction for {forecast_date} based on {as_of}'s observation")

        # 1/3 naive persistence: tomorrow == the latest actual.
        peak_demand_mw = refuse_implausible(peak_demand_mw, ELEC_MIN_MW, ELEC_MAX_MW,
                                            "naive_baseline", UNIT)
        cur.execute(UPSERT_SQL, (STATE, forecast_date, peak_demand_mw, "naive_baseline"))

        conn.commit()
        print(f"[OK] Stored naive_baseline prediction for {forecast_date}: "
              f"{peak_demand_mw:.0f} MW")

        # The features row backs both seasonal_naive and lightgbm, so fetch once
        # and apply the same publish-blocking guards to both.
        feature_row = get_latest_feature_row(cur)

        if feature_row is None:
            print("[WARN] No features row found; skipping seasonal_naive and LightGBM.")
        elif feature_row[0] != as_of:
            print(
                f"[WARN] Latest features row ({feature_row[0]}) doesn't match latest "
                f"observation ({as_of}); has vericast/elec/features.py been run for "
                f"today's data yet? Skipping seasonal_naive and LightGBM."
            )
        else:
            feature_values = feature_row[1:]
            named = dict(zip(FEATURE_COLUMNS, feature_values))

            # 2/3 seasonal naive: same weekday as the target, i.e. y(t-6).
            # Only needs the one column, so it can publish even when other
            # features are NULL.
            seasonal = named["demand_lag_6"]
            if seasonal is None:
                print("[WARN] demand_lag_6 is NULL (date gap within the last week); "
                      "skipping seasonal_naive.")
            else:
                seasonal = refuse_implausible(seasonal, ELEC_MIN_MW, ELEC_MAX_MW,
                                              "seasonal_naive", UNIT)
                cur.execute(UPSERT_SQL, (STATE, forecast_date, seasonal, "seasonal_naive"))

                conn.commit()
                print(f"[OK] Stored seasonal_naive prediction for {forecast_date}: "
                      f"{seasonal:.0f} MW")

            # 3/3 LightGBM: needs every feature present. Guarded rather than an
            # empty `if lgbm_model is None: pass` first arm - load_lightgbm_model()
            # has already printed the missing-artifact warning, so there was
            # nothing for that branch to do. Same shape as the PM2.5 twin.
            if lgbm_model is not None:
                if any(v is None for v in feature_values):
                    missing = [name for name, v in named.items() if v is None]
                    print(f"[WARN] Latest features row has NULL values {missing} (likely a "
                          f"date gap); skipping LightGBM forecast to avoid a garbage prediction.")
                else:
                    import numpy as np
                    X = np.array([feature_values], dtype=float)
                    lgbm_pred = float(lgbm_model.predict(X)[0])

                    lgbm_pred = refuse_implausible(lgbm_pred, ELEC_MIN_MW, ELEC_MAX_MW,
                                                   "lightgbm", UNIT)
                    cur.execute(UPSERT_SQL, (STATE, forecast_date, lgbm_pred, "lightgbm"))
                    conn.commit()
                    print(f"[OK] Stored lightgbm prediction for {forecast_date}: "
                          f"{lgbm_pred:.0f} MW")

        # Scoring is vericast/elec/score.py's job (step 4, before this one). It
        # scores *every* pending row, so a missed day self-heals.

        return True


if __name__ == "__main__":
    make_daily_prediction()
