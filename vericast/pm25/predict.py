"""Publish next-day Nagpur PM2.5 forecasts for LightGBM and naive baseline.

Forecasts are anchored to the day after the latest observation (latest_obs + 1 day).
All predictions are validated for staleness and physical plausibility prior to DB commit.
"""
import os
import psycopg
import lightgbm as lgb
from datetime import timedelta
from dotenv import load_dotenv

from vericast import (
    MODEL_PM25 as MODEL_PATH,
    PM25_MAX,
    PM25_MIN,
    PM25_STALE_LIMIT_DAYS,
    local_time,
    refuse_implausible,
    refuse_stale,
    require_city_of_record,
)
from vericast.pm25.train import FEATURE_COLUMNS

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))

UNIT = "ug/m3"



def load_lightgbm_model():
    """Load the production model artifact. Returns None if it hasn't been
    trained yet, so the pipeline degrades to naive-only instead of crashing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No LightGBM model artifact found at {MODEL_PATH}; "
              f"skipping LightGBM forecast. Run vericast/pm25/train.py first.")
        return None
    return lgb.Booster(model_file=MODEL_PATH)


def get_latest_feature_row(cur):
    """Get the most recent row from `features` - this is what we condition
    tomorrow's forecast on. Its as_of should be the same as the latest
    observation (features are engineered same-day)."""
    cur.execute(f"""
        SELECT as_of, {", ".join(FEATURE_COLUMNS)}
        FROM features
        WHERE city = %s
        ORDER BY as_of DESC
        LIMIT 1
    """, (CITY,))
    return cur.fetchone()


def make_daily_prediction():
    """Make a daily prediction for the day after the latest observation and store it"""
    # The connection is the context manager, as in score.py: a raise anywhere
    # below closes it instead of leaking it against Neon's connection limit.
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        lgbm_model = load_lightgbm_model()

        today = local_time.today()
        yesterday = today - timedelta(days=1)

        # Get the most recent observation
        cur.execute("""
            SELECT as_of, pm2_5
            FROM observations
            WHERE city = %s
            ORDER BY as_of DESC
            LIMIT 1
        """, (CITY,))

        row = cur.fetchone()
        if not row:
            raise Exception("No observations found")

        as_of, pm2_5 = row
        forecast_date = as_of + timedelta(days=1)

        # Validate that the observation is fresh enough to anchor a forecast.
        stale_days = refuse_stale(as_of, today, PM25_STALE_LIMIT_DAYS, "PM2.5")
        if as_of != yesterday:
            print(f"[WARN] Most recent observation is from {as_of}, {stale_days} "
                  f"day(s) old (expected data through {yesterday}). Forecasting "
                  f"for {forecast_date}, not {today + timedelta(days=1)}.")

        print(f"Making prediction for {forecast_date} based on {as_of}'s observation")

        # Upsert prediction into the public record with source='daily'.
        upsert_sql = """
        INSERT INTO predictions (city, forecast_date, predicted_pm2_5, model)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (city, forecast_date, model) DO UPDATE SET
            predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
            source = 'daily',
            created_at = CURRENT_TIMESTAMP;
        """

        if pm2_5 is None:
            print(f"[WARN] Latest observation ({as_of}) has NULL pm2_5; "
                  f"skipping naive_baseline forecast for {forecast_date}.")
        else:
            pm2_5 = refuse_implausible(pm2_5, PM25_MIN, PM25_MAX,
                                       "naive_baseline", UNIT)
            cur.execute(upsert_sql, (CITY, forecast_date, pm2_5, "naive_baseline"))
            conn.commit()
            print(f"[OK] Stored naive_baseline prediction for {forecast_date}: {pm2_5:.2f} PM2.5")

        # LightGBM prediction, if a trained artifact is available. Conditions on the
        # latest features row, whose lag/rolling values are what predict forecast_date.
        if lgbm_model is not None:
            feature_row = get_latest_feature_row(cur)
            if feature_row is None:
                print("[WARN] No features row found; skipping LightGBM forecast.")
            else:
                feat_as_of = feature_row[0]
                feature_values = feature_row[1:]

                if feat_as_of != as_of:
                    print(
                        f"[WARN] Latest features row ({feat_as_of}) doesn't match latest "
                        f"observation ({as_of}); has vericast/pm25/features.py been run for "
                        f"today's data yet? Skipping LightGBM forecast."
                    )
                elif any(v is None for v in feature_values):
                    print("[WARN] Latest features row has NULL values (likely a date gap); "
                          "skipping LightGBM forecast to avoid a garbage prediction.")
                else:
                    import numpy as np
                    X = np.array([feature_values], dtype=float)
                    lgbm_pred = float(lgbm_model.predict(X)[0])

                    lgbm_pred = refuse_implausible(lgbm_pred, PM25_MIN, PM25_MAX,
                                                   "lightgbm", UNIT)
                    cur.execute(upsert_sql, (CITY, forecast_date, lgbm_pred, "lightgbm"))
                    conn.commit()
                    print(f"[OK] Stored lightgbm prediction for {forecast_date}: {lgbm_pred:.2f} PM2.5")

        # Scoring is score.py's job: it runs as step 4/6, before this script, and
        # scores *every* pending row rather than just yesterday's, so a missed day
        # self-heals. A second copy here only gave the rule two places to drift.

        return True

if __name__ == "__main__":
    make_daily_prediction()