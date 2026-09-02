"""Seed the PM2.5 launch record with a walk-forward backtest.

Retrain from scratch at every step, predict strictly the next day, never let a
model see its own target. Run once at launch so the public leaderboard opens with
a real measured record instead of an empty table.

Imports FEATURE_COLUMNS / PARAMS / DATASET_SQL from vericast.pm25.train rather than
re-declaring them, so the backtest cannot drift from the model it characterises.
"""
import os
import sys
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vericast import require_city_of_record  # noqa: E402
from vericast.pm25.train import (  # noqa: E402
    CITY, DATASET_SQL, FEATURE_COLUMNS, NUM_BOOST_ROUND, PARAMS,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

MIN_TRAIN_SIZE = 30

# model_performance has no city column, so a run under another CITY would write its
# scores into Nagpur's row. Same refusal as app.py and vericast/pm25/score.py -
# train.py already applied it to CITY, this re-states the reason it matters here.
require_city_of_record(CITY)

# Looked up by name, not index 0, so a FEATURE_COLUMNS reorder cannot silently
# repoint the baseline.
NAIVE_COL = FEATURE_COLUMNS.index("pm2_5_lag_1")  # y(t) -> persistence


def load_dataset():
    """Load the exact t -> t+1 forecasting dataset."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(DATASET_SQL, (CITY,))
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No forecasting data available.")

    return rows


def validate_alignment(feature_dates, target_dates):
    """Verify every feature date maps to exactly the next day."""

    for feature_date, target_date in zip(feature_dates, target_dates):
        if (target_date - feature_date).days != 1:
            raise ValueError(
                f"Invalid target alignment: "
                f"{feature_date} -> {target_date}"
            )

    print("[OK] Target alignment verified: features(t) -> target(t+1)")


def calculate_metrics(predictions, actuals):
    """Calculate MAE and RMSE."""

    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    # float(), not the numpy scalar: psycopg has no adapter for np.float64, so an
    # unwrapped value reaches executemany as an unknown type.
    mae = float(np.mean(np.abs(actuals - predictions)))
    rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))

    return mae, rmse


def run_backtest():
    """Run the official walk-forward backtest."""

    rows = load_dataset()

    feature_dates = np.array([row[0] for row in rows])
    target_dates = np.array([row[1] for row in rows])

    X = np.array(
        [list(row[2:-1]) for row in rows],
        dtype=float,
    )

    y = np.array(
        [row[-1] for row in rows],
        dtype=float,
    )

    print(f"Retrieved {len(rows)} samples")
    print(f"Feature matrix shape: {X.shape}")

    # Raised, not asserted: this script writes actuals straight into the published
    # record, so it gets the same -O-proof gate as vericast/pm25/train.py.
    if X.shape[1] != len(FEATURE_COLUMNS):
        raise AssertionError(
            f"Feature count mismatch: got {X.shape[1]}, expected "
            f"{len(FEATURE_COLUMNS)}."
        )

    validate_alignment(feature_dates, target_dates)

    if len(X) <= MIN_TRAIN_SIZE:
        raise RuntimeError(
            f"Need more than {MIN_TRAIN_SIZE} samples."
        )

    naive_predictions = []
    lightgbm_predictions = []
    actuals = []
    evaluation_dates = []

    total_predictions = len(X) - MIN_TRAIN_SIZE

    for i in range(MIN_TRAIN_SIZE, len(X)):

        X_train = X[:i]
        y_train = y[:i]

        X_test = X[i:i + 1]
        y_true = y[i]

        # ---- Naive baseline ----
        naive_prediction = X_test[0][NAIVE_COL]

        # ---- LightGBM ----
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
        )

        model = lgb.train(
            PARAMS,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
        )

        lightgbm_prediction = model.predict(X_test)[0]

        naive_predictions.append(naive_prediction)
        lightgbm_predictions.append(lightgbm_prediction)
        actuals.append(y_true)
        evaluation_dates.append(target_dates[i])

        if (i - MIN_TRAIN_SIZE) % 50 == 0:
            completed = i - MIN_TRAIN_SIZE + 1

            print(
                f"   Processed "
                f"{completed}/{total_predictions} samples"
            )

    return (
        evaluation_dates,
        naive_predictions,
        lightgbm_predictions,
        actuals,
    )


def save_results(
    evaluation_dates,
    naive_predictions,
    lightgbm_predictions,
    actuals,
):
    """Save all individual predictions to PostgreSQL."""

    # source = 'backtest': these rows have their actual at write time, so they are
    # not published-then-verified and /evaluation reports them separately. The
    # DO UPDATE touches neither actual_pm2_5 nor source, and the WHERE keeps it off
    # daily rows: run-once or not, a re-run must not overwrite a verified actual
    # with a backtest-computed one, nor relabel a published row.
    insert_sql = """
        INSERT INTO predictions (
            city,
            forecast_date,
            predicted_pm2_5,
            actual_pm2_5,
            model,
            source
        )
        VALUES (%s, %s, %s, %s, %s, 'backtest')
        ON CONFLICT (city, forecast_date, model)
        DO UPDATE SET
            predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
            created_at = CURRENT_TIMESTAMP
        WHERE predictions.source = 'backtest';
    """

    records = []

    for date, naive, lightgbm_pred, actual in zip(
        evaluation_dates,
        naive_predictions,
        lightgbm_predictions,
        actuals,
    ):
        records.append(
            (
                CITY,
                date,
                float(naive),
                float(actual),
                "naive_baseline",
            )
        )

        records.append(
            (
                CITY,
                date,
                float(lightgbm_pred),
                float(actual),
                "lightgbm",
            )
        )

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, records)

        conn.commit()

    print(f"[OK] Saved {len(records)} prediction records")
    print(
        f"   {len(evaluation_dates)} target dates x 2 models"
    )


def save_model_performance(
    evaluation_dates,
    naive_predictions,
    lightgbm_predictions,
    actuals,
):
    """Save aggregate model performance."""

    naive_mae, naive_rmse = calculate_metrics(
        naive_predictions,
        actuals,
    )

    lightgbm_mae, lightgbm_rmse = calculate_metrics(
        lightgbm_predictions,
        actuals,
    )

    sample_size = len(actuals)

    score_date = evaluation_dates[-1]

    # score_date is the last evaluated day, which the daily scorer has usually
    # already scored. Same guard as save_results() one table up: without the WHERE
    # a re-run overwrites that day's real MAE with the backtest aggregate and
    # relabels the row 'backtest', and /leaderboard's source = 'daily' filter then
    # drops the newest verified score for good.
    insert_sql = """
        INSERT INTO model_performance (
            score_date,
            model,
            mae,
            rmse,
            sample_size,
            source
        )
        VALUES (%s, %s, %s, %s, %s, 'backtest')
        ON CONFLICT (score_date, model)
        DO UPDATE SET
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            sample_size = EXCLUDED.sample_size,
            created_at = CURRENT_TIMESTAMP
        WHERE model_performance.source = 'backtest';
    """

    records = [
        (
            score_date,
            "naive_baseline",
            float(naive_mae),
            float(naive_rmse),
            sample_size,
        ),
        (
            score_date,
            "lightgbm",
            float(lightgbm_mae),
            float(lightgbm_rmse),
            sample_size,
        ),
    ]

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, records)

        conn.commit()

    print("\n[OK] Model performance saved")

    print(
        f"   Naive:    MAE={naive_mae:.4f}, "
        f"RMSE={naive_rmse:.4f}"
    )

    print(
        f"   LightGBM: MAE={lightgbm_mae:.4f}, "
        f"RMSE={lightgbm_rmse:.4f}"
    )

    return naive_mae, naive_rmse, lightgbm_mae, lightgbm_rmse


def verify_saved_results():
    """Verify that predictions were actually stored."""

    query = """
        SELECT
            model,
            COUNT(*) AS prediction_count,
            MIN(forecast_date),
            MAX(forecast_date)
        FROM predictions
        WHERE city = %s
        GROUP BY model
        ORDER BY model;
    """

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (CITY,))
            rows = cur.fetchall()

    print("\nSaved prediction summary")

    for model, count, min_date, max_date in rows:
        print(
            f"   {model}: "
            f"{count} predictions | "
            f"{min_date} -> {max_date}"
        )


def main():
    print("=" * 60)
    print("           VERICAST BACKTEST PERSISTENCE")
    print("=" * 60)

    (
        evaluation_dates,
        naive_predictions,
        lightgbm_predictions,
        actuals,
    ) = run_backtest()

    save_results(
        evaluation_dates,
        naive_predictions,
        lightgbm_predictions,
        actuals,
    )

    save_model_performance(
        evaluation_dates,
        naive_predictions,
        lightgbm_predictions,
        actuals,
    )

    verify_saved_results()

    print("\n[OK] VeriCast backtest results successfully persisted.")


if __name__ == "__main__":
    main()