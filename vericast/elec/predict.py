"""Publish tomorrow's Maharashtra peak electricity demand forecast (MW).

Anchors predictions to the day after the newest observation (latest_obs + 1 day).
Three models compete: naive_baseline (persistence), seasonal_naive (same weekday last week),
and LightGBM. All predictions pass staleness and physical plausibility gates before commit.
"""
import os
from datetime import timedelta
import numpy as np
import psycopg
import lightgbm as lgb
from dotenv import load_dotenv

from vericast import (
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    ELEC_STALE_LIMIT_DAYS,
    MODEL_ELEC as MODEL_PATH,
    acquire_pipeline_lock,
    local_time,
    refuse_implausible,
    refuse_stale,
    require_database_url,
    send_alert,
)
from vericast.elec.train import FEATURE_COLUMNS

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

UNIT = "MW"

# Upsert electricity predictions with source='daily'.
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
    """Predict peak demand for the day after the latest observation and store it.

    Returns True on a full or deliberately-degraded run; skips alert loudly and
    diagnose.py gates completeness, mirroring the PM2.5 twin.
    """
    require_database_url(DATABASE_URL)
    # Load the artifact BEFORE opening a pooled connection: see the PM2.5 twin.
    lgbm_model = load_lightgbm_model()
    # Connection as context manager, as in score.py: a raise anywhere below
    # closes it instead of leaking it against Neon's connection limit.
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        acquire_pipeline_lock(cur, "elec_predict")

        today = local_time.today()
        yesterday = today - timedelta(days=1)

        cur.execute("""
            SELECT as_of, peak_demand_mw
            FROM electricity_observations
            WHERE state = %s
            ORDER BY as_of DESC
            LIMIT 1
        """, (STATE,))

        row = cur.fetchone()
        if not row:
            raise RuntimeError("No electricity observations found")

        as_of, peak_demand_mw = row
        forecast_date = as_of + timedelta(days=1)

        # Raises past the limit, before anything is published. 2-4 days is the
        # normal lag for this mirror, so only past ELEC_STALE_LIMIT_DAYS has it
        # actually stalled.
        stale_days = refuse_stale(as_of, today, ELEC_STALE_LIMIT_DAYS, "electricity")
        if as_of != yesterday:
            print(f"[WARN] Most recent observation is from {as_of}, {stale_days} "
                  f"day(s) old (expected data through {yesterday}; normal lag for "
                  f"this mirror is 2-4 days). Forecasting for {forecast_date}, "
                  f"not {today + timedelta(days=1)}.")
        else:
            print(f"Latest observation {as_of} is {stale_days} day(s) old "
                  f"(normal for this source).")

        print(f"Making prediction for {forecast_date} based on {as_of}'s observation")

        skipped = []
        # 1/3 naive persistence: tomorrow == the latest actual. A NULL peak cannot
        # happen (NOT NULL schema) but a corrupt read must degrade like the PM2.5
        # twin instead of crashing the whole day via refuse_implausible(None).
        if peak_demand_mw is None:
            print(f"[WARN] Latest observation ({as_of}) has NULL peak_demand_mw; "
                  f"skipping naive_baseline forecast for {forecast_date}.")
            skipped.append("naive_baseline (NULL observation)")
        else:
            peak_demand_mw = refuse_implausible(peak_demand_mw, ELEC_MIN_MW, ELEC_MAX_MW,
                                                "naive_baseline", UNIT)
            cur.execute(UPSERT_SQL, (STATE, forecast_date, peak_demand_mw, "naive_baseline"))

            print(f"[OK] Stored naive_baseline prediction for {forecast_date}: "
                  f"{peak_demand_mw:.0f} MW")

        # The features row backs both seasonal_naive and lightgbm, so fetch once
        # and apply the same publish-blocking guards to both.
        feature_row = get_latest_feature_row(cur)

        if feature_row is None:
            print("[WARN] No features row found; skipping seasonal_naive and LightGBM.")
            skipped.append("seasonal_naive+lightgbm (no features row)")
        elif feature_row[0] != as_of:
            print(
                f"[WARN] Latest features row ({feature_row[0]}) doesn't match latest "
                f"observation ({as_of}); has vericast/elec/features.py been run for "
                f"today's data yet? Skipping seasonal_naive and LightGBM."
            )
            skipped.append(f"seasonal_naive+lightgbm (features {feature_row[0]} != obs {as_of})")
        else:
            feature_values = feature_row[1:]
            named = dict(zip(FEATURE_COLUMNS, feature_values))

            # 2/3 seasonal naive: same weekday as the target, i.e. y(t-6).
            # Only needs the one column, so it can publish even when other
            # features are NULL.
            seasonal = named.get("demand_lag_6")
            if seasonal is None:
                print("[WARN] demand_lag_6 is NULL (date gap within the last week); "
                      "skipping seasonal_naive.")
                skipped.append("seasonal_naive (NULL demand_lag_6)")
            else:
                seasonal = refuse_implausible(seasonal, ELEC_MIN_MW, ELEC_MAX_MW,
                                              "seasonal_naive", UNIT)
                cur.execute(UPSERT_SQL, (STATE, forecast_date, seasonal, "seasonal_naive"))

                print(f"[OK] Stored seasonal_naive prediction for {forecast_date}: "
                      f"{seasonal:.0f} MW")

            # 3/3 LightGBM: needs every feature present. Guarded rather than an
            # empty `if lgbm_model is None: pass` first arm - load_lightgbm_model()
            # has already printed the missing-artifact warning, so there was
            # nothing for that branch to do. Same shape as the PM2.5 twin.
            if lgbm_model is None:
                skipped.append("lightgbm (missing artifact)")
            else:
                if any(v is None for v in feature_values):
                    missing = [name for name, v in named.items() if v is None]
                    print(f"[WARN] Latest features row has NULL values {missing} (likely a "
                          f"date gap); skipping LightGBM forecast to avoid a garbage prediction.")
                    skipped.append(f"lightgbm (NULL features: {missing})")
                else:
                    X = np.array([feature_values], dtype=float)
                    lgbm_pred = float(lgbm_model.predict(X)[0])

                    lgbm_pred = refuse_implausible(lgbm_pred, ELEC_MIN_MW, ELEC_MAX_MW,
                                                   "lightgbm", UNIT)
                    cur.execute(UPSERT_SQL, (STATE, forecast_date, lgbm_pred, "lightgbm"))
                    print(f"[OK] Stored lightgbm prediction for {forecast_date}: "
                          f"{lgbm_pred:.0f} MW")

        # Atomic commit for the forecast date across all models.
        conn.commit()

        if skipped:
            # Loud partial run: see the PM2.5 twin for why exit 0 stays.
            print(f"[WARN] Partial electricity publish for {forecast_date}: "
                  f"skipped {', '.join(skipped)}.")
            send_alert(
                "VeriCast electricity partial publish",
                f"Forecast for {forecast_date} published with skips: "
                f"{', '.join(skipped)}. diagnose.py will gate completeness.",
            )

        return True


if __name__ == "__main__":
    make_daily_prediction()
