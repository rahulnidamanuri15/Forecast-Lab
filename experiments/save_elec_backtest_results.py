"""Seed the electricity launch record with a walk-forward backtest.

Same discipline as save_backtest_results.py: retrain from scratch at every step,
predict strictly the next day, never let a model see its own target. Run once at
launch so the public leaderboard opens with a real measured record instead of an
empty table.

Imports FEATURE_COLUMNS / PARAMS / DATASET_SQL from vericast.elec.train rather than
re-declaring them, so the backtest cannot silently drift from the production
model it is meant to characterise.

Note on the 2025-05-21 -> 2025-05-24 gap: no special handling is needed. The
`o.as_of = f.as_of + INTERVAL '1 day'` join drops any pair that would straddle
the gap, and the NOT NULL filter drops the feature rows whose lags are NULL
because of it. Walk-forward training needs a set of aligned (features, target)
pairs, not calendar contiguity.
"""
import os
import sys
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vericast.elec.train import FEATURE_COLUMNS, PARAMS, DATASET_SQL, STATE  # noqa: E402

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

MIN_TRAIN_SIZE = 30

# Which feature column each baseline reads. Looked up by name so a change to
# FEATURE_COLUMNS' order can't silently repoint a baseline at the wrong series.
NAIVE_COL = FEATURE_COLUMNS.index("demand_lag_1")      # y(t)   -> persistence
SEASONAL_COL = FEATURE_COLUMNS.index("demand_lag_6")   # y(t-6) -> same weekday as target


def load_dataset():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(DATASET_SQL, (STATE,))
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No electricity forecasting data available.")

    return rows


def validate_alignment(feature_dates, target_dates):
    """Verify every feature date maps to exactly the next day."""
    for feature_date, target_date in zip(feature_dates, target_dates):
        if (target_date - feature_date).days != 1:
            raise ValueError(f"Invalid target alignment: {feature_date} -> {target_date}")

    print("[OK] Target alignment verified: features(t) -> target(t+1)")


def calculate_metrics(predictions, actuals):
    """MAE, RMSE, MAPE(%)."""
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    mae = float(np.mean(np.abs(actuals - predictions)))
    rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    nonzero = actuals != 0
    mape = (float(np.mean(np.abs((predictions[nonzero] - actuals[nonzero])
                                 / actuals[nonzero])) * 100)
            if nonzero.any() else None)

    return mae, rmse, mape


def run_backtest():
    rows = load_dataset()

    feature_dates = np.array([row[0] for row in rows])
    target_dates = np.array([row[1] for row in rows])
    X = np.array([list(row[2:-1]) for row in rows], dtype=float)
    y = np.array([row[-1] for row in rows], dtype=float)

    print(f"Retrieved {len(rows)} samples")
    print(f"Feature matrix shape: {X.shape}")
    assert X.shape[1] == len(FEATURE_COLUMNS), (
        f"Feature count mismatch: got {X.shape[1]}, expected {len(FEATURE_COLUMNS)}."
    )

    validate_alignment(feature_dates, target_dates)

    if len(X) <= MIN_TRAIN_SIZE:
        raise RuntimeError(f"Need more than {MIN_TRAIN_SIZE} samples.")

    predictions = {"naive_baseline": [], "seasonal_naive": [], "lightgbm": []}
    actuals = []
    evaluation_dates = []

    total = len(X) - MIN_TRAIN_SIZE

    for i in range(MIN_TRAIN_SIZE, len(X)):
        X_train, y_train = X[:i], y[:i]
        X_test = X[i:i + 1]

        predictions["naive_baseline"].append(X_test[0][NAIVE_COL])
        predictions["seasonal_naive"].append(X_test[0][SEASONAL_COL])

        model = lgb.train(PARAMS, lgb.Dataset(X_train, label=y_train),
                          num_boost_round=100)
        predictions["lightgbm"].append(model.predict(X_test)[0])

        actuals.append(y[i])
        evaluation_dates.append(target_dates[i])

        if (i - MIN_TRAIN_SIZE) % 100 == 0:
            print(f"   Processed {i - MIN_TRAIN_SIZE + 1}/{total} samples")

    return evaluation_dates, predictions, actuals


def save_results(evaluation_dates, predictions, actuals):
    """Store every individual prediction *with* its actual - these are verified
    historical results, not pending forecasts.

    source = 'backtest' so /electricity/evaluation can separate them from rows
    published before their actual existed. The DO UPDATE deliberately does not
    touch actual_demand_mw or source, and the WHERE keeps a re-run from
    overwriting a real daily forecast on an overlapping date.
    """
    insert_sql = """
        INSERT INTO electricity_predictions
            (state, forecast_date, predicted_demand_mw, actual_demand_mw, model, source)
        VALUES (%s, %s, %s, %s, %s, 'backtest')
        ON CONFLICT (state, forecast_date, model) DO UPDATE SET
            predicted_demand_mw = EXCLUDED.predicted_demand_mw,
            created_at = CURRENT_TIMESTAMP
        WHERE electricity_predictions.source = 'backtest';
    """

    records = [
        (STATE, date, float(preds[idx]), float(actuals[idx]), model)
        for model, preds in predictions.items()
        for idx, date in enumerate(evaluation_dates)
    ]

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, records)
        conn.commit()

    print(f"[OK] Saved {len(records)} prediction records")
    print(f"   {len(evaluation_dates)} target dates x {len(predictions)} models")


def save_model_performance(evaluation_dates, predictions, actuals):
    """One aggregate row per model at the last evaluated date - the launch record."""
    insert_sql = """
        INSERT INTO electricity_model_performance
            (state, score_date, model, mae, rmse, mape, sample_size, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'backtest')
        ON CONFLICT (state, score_date, model) DO UPDATE SET
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            mape = EXCLUDED.mape,
            sample_size = EXCLUDED.sample_size,
            source = EXCLUDED.source,
            created_at = CURRENT_TIMESTAMP;
    """

    score_date = evaluation_dates[-1]
    sample_size = len(actuals)
    metrics = {model: calculate_metrics(preds, actuals)
               for model, preds in predictions.items()}

    records = [(STATE, score_date, model, mae, rmse, mape, sample_size)
               for model, (mae, rmse, mape) in metrics.items()]

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, records)
        conn.commit()

    print(f"\n[OK] Model performance saved (n={sample_size}, score_date={score_date})")
    for model, (mae, rmse, mape) in sorted(metrics.items(), key=lambda kv: kv[1][0]):
        print(f"   {model:15s} MAE={mae:8.2f} MW  RMSE={rmse:8.2f} MW  MAPE={mape:5.2f}%")

    return metrics


def verify_saved_results():
    query = """
        SELECT model, COUNT(*), COUNT(actual_demand_mw),
               MIN(forecast_date), MAX(forecast_date)
        FROM electricity_predictions
        WHERE state = %s
        GROUP BY model
        ORDER BY model;
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (STATE,))
            rows = cur.fetchall()

    print("\nSaved prediction summary")
    for model, count, scored, min_date, max_date in rows:
        print(f"   {model:15s} {count} predictions ({scored} scored) | "
              f"{min_date} -> {max_date}")


def main():
    print("=" * 62)
    print("      VERICAST ELECTRICITY BACKTEST PERSISTENCE (Maharashtra)")
    print("=" * 62)

    evaluation_dates, predictions, actuals = run_backtest()
    save_results(evaluation_dates, predictions, actuals)
    save_model_performance(evaluation_dates, predictions, actuals)
    verify_saved_results()

    print("\n[OK] Electricity backtest results successfully persisted.")


if __name__ == "__main__":
    main()
