import os
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

from vericast import MODEL_PM25 as MODEL_PATH
from vericast.gate import challenger_ships

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CITY = os.getenv("CITY", "Nagpur")

# 15 features, fixed order. Single definition of the feature order for this
# target: predict.py and the experiments/ backtests import it from here, so
# they cannot drift.
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

NUM_BOOST_ROUND = 100

UNIT = "ug/m3"


# The INTERVAL '1 day' join is what makes this a t -> t+1 dataset. Every feature
# is computed from data at or before f.as_of; the label is the next day's actual.
# Both the select list and the NOT-NULL filter are generated from
# FEATURE_COLUMNS, so a reorder or rename cannot silently mislabel the model's
# features - which the count-only assert below could not catch.
DATASET_SQL = f"""
    SELECT
        f.as_of AS feature_date,
        o.as_of AS target_date,
        {", ".join("f." + c for c in FEATURE_COLUMNS)},
        o.pm2_5 AS target_pm2_5
    FROM features f
    JOIN observations o
      ON o.city = f.city
     AND o.as_of = f.as_of + INTERVAL '1 day'
    WHERE f.city = %s
      AND {" AND ".join("f." + c + " IS NOT NULL" for c in FEATURE_COLUMNS)}
      AND o.pm2_5 IS NOT NULL
    ORDER BY f.as_of;
"""


def load_full_dataset():
    """Load the full t -> t+1 dataset (unlike backtest scripts, no walk-forward split;
    this is meant to train one final model on everything we have)."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(DATASET_SQL, (CITY,))
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

    # Retrain gate before anything is written. A refused retrain leaves the
    # incumbent artifact untouched and exits 0 - see vericast/gate.py.
    if not challenger_ships(X, y, PARAMS, NUM_BOOST_ROUND,
                            FEATURE_COLUMNS.index("pm2_5_lag_1"),
                            incumbent_path=MODEL_PATH,
                            feature_names=FEATURE_COLUMNS, unit=UNIT):
        print(f"[SKIP] Keeping the existing model at {MODEL_PATH}.")
        return None

    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    model = lgb.train(PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)

    model.save_model(MODEL_PATH)
    print(f"[OK] Saved production model to {MODEL_PATH}")

    # Sanity check: reload and confirm predictions are reproducible
    reloaded = lgb.Booster(model_file=MODEL_PATH)
    check_pred = reloaded.predict(X[-1:])
    print(f"Sanity check - reloaded model prediction on last row: {check_pred[0]:.4f} "
          f"(actual was {y[-1]:.4f})")

    importance = sorted(zip(FEATURE_COLUMNS, model.feature_importance("gain")),
                        key=lambda pair: pair[1], reverse=True)
    print("\nFeature importance (gain):")
    for name, gain in importance:
        print(f"  {name}: {gain:.0f}")

    return MODEL_PATH


if __name__ == "__main__":
    train_and_save()