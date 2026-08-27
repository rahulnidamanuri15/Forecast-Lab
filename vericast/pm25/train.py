import os
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

from vericast import MODEL_PM25 as MODEL_PATH

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CITY = "Nagpur"

# Single definition of the feature order for this target: predict.py and the
# experiments/ backtests import it from here, so they cannot drift.
FEATURE_COLUMNS = [
    "pm2_5_lag_1", "pm10_lag_1", "temperature_lag_1", "wind_speed_lag_1", "precipitation_lag_1",
    "pm2_5_roll_7", "pm2_5_roll_30", "pm10_roll_7", "pm10_roll_30",
    "day_of_week", "month", "is_weekend",
    "temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum",
]

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
}


def load_full_dataset():
    """Load the full t -> t+1 dataset (unlike backtest scripts, no walk-forward split;
    this is meant to train one final model on everything we have)."""
    query = """
        SELECT
            f.as_of AS feature_date,
            o.as_of AS target_date,

            f.pm2_5_lag_1, f.pm10_lag_1, f.temperature_lag_1,
            f.wind_speed_lag_1, f.precipitation_lag_1,

            f.pm2_5_roll_7, f.pm2_5_roll_30, f.pm10_roll_7, f.pm10_roll_30,

            f.day_of_week, f.month, f.is_weekend,

            f.temperature_2m_mean, f.wind_speed_10m_max, f.precipitation_sum,

            o.pm2_5 AS target_pm2_5

        FROM features f
        JOIN observations o
          ON o.city = f.city
         AND o.as_of = f.as_of + INTERVAL '1 day'

        WHERE f.city = %s
          AND f.pm2_5_lag_1 IS NOT NULL
          AND o.pm2_5 IS NOT NULL

        ORDER BY f.as_of;
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (CITY,))
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No training data available.")

    return rows


def train_and_save():
    rows = load_full_dataset()

    feature_dates = [row[0] for row in rows]
    target_dates = [row[1] for row in rows]

    for feature_date, target_date in zip(feature_dates, target_dates):
        if (target_date - feature_date).days != 1:
            raise ValueError(f"Invalid target alignment: {feature_date} -> {target_date}")

    X = np.array([list(row[2:-1]) for row in rows], dtype=float)
    y = np.array([row[-1] for row in rows], dtype=float)

    print(f"Training production LightGBM on {len(X)} samples "
          f"({feature_dates[0]} -> {feature_dates[-1]})")
    print(f"Feature matrix shape: {X.shape} ({len(FEATURE_COLUMNS)} columns expected)")
    assert X.shape[1] == len(FEATURE_COLUMNS), (
        f"Feature count mismatch: got {X.shape[1]}, expected {len(FEATURE_COLUMNS)}. "
        "The SQL column order and FEATURE_COLUMNS must match."
    )

    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    model = lgb.train(PARAMS, train_data, num_boost_round=100)

    model.save_model(MODEL_PATH)
    print(f"[OK] Saved production model to {MODEL_PATH}")

    # Sanity check: reload and confirm predictions are reproducible
    reloaded = lgb.Booster(model_file=MODEL_PATH)
    check_pred = reloaded.predict(X[-1:])
    print(f"Sanity check - reloaded model prediction on last row: {check_pred[0]:.4f} "
          f"(actual was {y[-1]:.4f})")

    return MODEL_PATH


if __name__ == "__main__":
    train_and_save()