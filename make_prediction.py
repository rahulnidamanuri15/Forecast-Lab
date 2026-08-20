import os
import psycopg
import lightgbm as lgb
from datetime import timedelta
from dotenv import load_dotenv

import local_time

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = "Nagpur"
MODEL_PATH = "lightgbm_model.txt"

# Must match the column order in train_production_model.py / compare_models.py
FEATURE_COLUMNS = [
    "pm2_5_lag_1", "pm10_lag_1", "temperature_lag_1", "wind_speed_lag_1", "precipitation_lag_1",
    "pm2_5_roll_7", "pm2_5_roll_30", "pm10_roll_7", "pm10_roll_30",
    "day_of_week", "month", "is_weekend",
    "temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum",
]


def load_lightgbm_model():
    """Load the production model artifact. Returns None if it hasn't been
    trained yet, so the pipeline degrades to naive-only instead of crashing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No LightGBM model artifact found at {MODEL_PATH}; "
              f"skipping LightGBM forecast. Run train_production_model.py first.")
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
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()

        lgbm_model = load_lightgbm_model()

        # Real-world "today" in the operating timezone (see local_time.py),
        # used only to detect staleness and to drive yesterday's scoring below.
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

        # The forecast is always "the day after the data we actually have",
        # NOT blindly "real-world tomorrow". If ingestion has stalled and
        # as_of is stale, we still label the forecast correctly relative to
        # the data it's based on, instead of mislabeling old data as a
        # forecast for real-world tomorrow.
        forecast_date = as_of + timedelta(days=1)

        if as_of != yesterday:
            staleness_days = (today - as_of).days
            print(
                f"[WARN] Warning: most recent observation is from {as_of}, which is "
                f"{staleness_days} day(s) old (expected data through {yesterday}). "
                f"Ingestion may not be running. Forecasting for {forecast_date} "
                f"based on stale data instead of for {today + timedelta(days=1)}."
            )

        print(f"Making prediction for {forecast_date} based on {as_of}'s observation")

        # Insert or update the naive baseline prediction for forecast_date.
        # Note: the unique constraint is (city, forecast_date, model), so
        # naive and lightgbm rows for the same date coexist rather than
        # overwriting each other.
        upsert_sql = """
        INSERT INTO predictions (city, forecast_date, predicted_pm2_5, model)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (city, forecast_date, model) DO UPDATE SET
            predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
            created_at = CURRENT_TIMESTAMP;
        """

        cur.execute(upsert_sql, (CITY, forecast_date, pm2_5, "naive_baseline"))
        conn.commit()

        print(f"[OK] Stored naive_baseline prediction for {forecast_date}: {pm2_5:.2f} PM2.5")

        # LightGBM prediction, if a trained artifact is available.
        # Uses the latest features row (as_of == as_of of latest observation),
        # since that row's lag/rolling features are what predict forecast_date.
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
                        f"observation ({as_of}); has engineer_features.py been run for "
                        f"today's data yet? Skipping LightGBM forecast."
                    )
                elif any(v is None for v in feature_values):
                    print("[WARN] Latest features row has NULL values (likely a date gap); "
                          "skipping LightGBM forecast to avoid a garbage prediction.")
                else:
                    import numpy as np
                    X = np.array([feature_values], dtype=float)
                    lgbm_pred = float(lgbm_model.predict(X)[0])

                    cur.execute(upsert_sql, (CITY, forecast_date, lgbm_pred, "lightgbm"))
                    conn.commit()
                    print(f"[OK] Stored lightgbm prediction for {forecast_date}: {lgbm_pred:.2f} PM2.5")

        # Now check if we have actuals for yesterday's forecast(s) and update those
        # records. There can be more than one row for `yesterday` now (one per
        # model - naive_baseline, lightgbm, ...), so this must handle all of them,
        # not just the first one found.
        cur.execute("""
            SELECT model, predicted_pm2_5
            FROM predictions
            WHERE city = %s
              AND forecast_date = %s
              AND actual_pm2_5 IS NULL
        """, (CITY, yesterday))

        pending_rows = cur.fetchall()
        if pending_rows:
            # Get the actual PM2.5 for yesterday from observations
            cur.execute("""
                SELECT pm2_5
                FROM observations
                WHERE city = %s
                  AND as_of = %s
            """, (CITY, yesterday))

            actual_row = cur.fetchone()
            if actual_row:
                actual_pm2_5 = actual_row[0]

                update_sql = """
                UPDATE predictions
                SET actual_pm2_5 = %s
                WHERE city = %s
                  AND forecast_date = %s
                  AND model = %s
                """

                for model_name, predicted_pm2_5 in pending_rows:
                    cur.execute(update_sql, (actual_pm2_5, CITY, yesterday, model_name))
                    conn.commit()
                    error = abs(actual_pm2_5 - predicted_pm2_5)
                    print(f"[OK] Updated {model_name} forecast for {yesterday}: "
                          f"predicted={predicted_pm2_5:.2f}, actual={actual_pm2_5:.2f}, error={error:.2f}")
            else:
                print(f"[WARN] No actual observation found for {yesterday} yet")
        else:
            print(f"[INFO] No pending forecast to update for {yesterday}")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"[FAIL] Error making prediction: {e}")
        raise

if __name__ == "__main__":
    make_daily_prediction()