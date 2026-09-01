"""Train the production LightGBM model for Maharashtra peak demand.

Same PARAMS and num_boost_round as vericast/pm25/train.py - no tuning until
there is a measured reason to. Separate artifact so the two targets' weekly
retrains cannot clobber each other.
"""
import os
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

from vericast import MODEL_ELEC as MODEL_PATH
from vericast.gate import challenger_ships

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = os.getenv("STATE", "Maharashtra")

# 14 features, fixed order. Single definition for this target: predict.py and
# experiments/save_elec_backtest_results.py import it from here.
FEATURE_COLUMNS = [
    "demand_lag_1", "demand_lag_2", "demand_lag_6",
    "demand_roll_7_mean", "demand_roll_7_max", "demand_roll_30_mean",
    "temp_lag_1", "temp_roll_7", "cooling_degree_days",
    "day_of_week", "month", "is_weekend",
    "temperature_2m_mean", "temperature_2m_max",
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

UNIT = "MW"

# The INTERVAL '1 day' join is what makes this a t -> t+1 dataset. Every feature
# is computed from data at or before f.as_of; the label is the next day's actual.
DATASET_SQL = f"""
    SELECT
        f.as_of AS feature_date,
        o.as_of AS target_date,
        {", ".join("f." + c for c in FEATURE_COLUMNS)},
        o.peak_demand_mw AS target_demand_mw
    FROM electricity_features f
    JOIN electricity_observations o
      ON o.state = f.state
     AND o.as_of = f.as_of + INTERVAL '1 day'
    WHERE f.state = %s
      AND {" AND ".join("f." + c + " IS NOT NULL" for c in FEATURE_COLUMNS)}
      AND o.peak_demand_mw IS NOT NULL
    ORDER BY f.as_of;
"""


def load_full_dataset():
    """Load the full t -> t+1 dataset. No walk-forward split - this trains one
    final model on everything available."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(DATASET_SQL, (STATE,))
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No electricity training data available.")

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
    # Raised, not asserted: python -O erases `assert`, and this is the only check
    # that the SQL column order still matches FEATURE_COLUMNS. A mismatch trains a
    # model on shuffled features that scores plausibly and is wrong every day.
    if X.shape[1] != len(FEATURE_COLUMNS):
        raise AssertionError(
            f"Feature count mismatch: got {X.shape[1]}, expected "
            f"{len(FEATURE_COLUMNS)}. The SQL column order and FEATURE_COLUMNS "
            f"must match."
        )

    # Retrain gate before anything is written. A refused retrain leaves the
    # incumbent artifact untouched and exits 0 - see vericast/gate.py.
    if not challenger_ships(X, y, PARAMS, NUM_BOOST_ROUND,
                            FEATURE_COLUMNS.index("demand_lag_1"),
                            incumbent_path=MODEL_PATH,
                            feature_names=FEATURE_COLUMNS, unit=UNIT):
        print(f"[SKIP] Keeping the existing model at {MODEL_PATH}.")
        return None

    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    model = lgb.train(PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)

    model.save_model(MODEL_PATH)
    print(f"[OK] Saved production model to {MODEL_PATH}")

    reloaded = lgb.Booster(model_file=MODEL_PATH)
    check_pred = reloaded.predict(X[-1:])
    print(f"Sanity check - reloaded model prediction on last row: {check_pred[0]:.2f} MW "
          f"(actual was {y[-1]:.2f} MW)")

    importance = sorted(zip(FEATURE_COLUMNS, model.feature_importance("gain")),
                        key=lambda pair: pair[1], reverse=True)
    print("\nFeature importance (gain):")
    for name, gain in importance:
        print(f"  {name}: {gain:.0f}")

    return MODEL_PATH


if __name__ == "__main__":
    train_and_save()
