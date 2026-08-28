import os
import sys
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vericast.pm25.train import FEATURE_COLUMNS, PARAMS, DATASET_SQL, CITY  # noqa: E402

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

MIN_TRAIN_SIZE = 30

# Which feature column the naive baseline reads. Looked up by name so a
# reorder of FEATURE_COLUMNS cannot silently repoint it at another series.
NAIVE_COL = FEATURE_COLUMNS.index("pm2_5_lag_1")  # y(t) -> persistence


def load_dataset():
    """Load the exact t -> t+1 forecasting dataset used by both models."""

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(DATASET_SQL, (CITY,))
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No forecasting data available.")

    return rows


def validate_alignment(feature_dates, target_dates):
    """Verify every sample is exactly one day ahead."""
    for feature_date, target_date in zip(feature_dates, target_dates):
        if (target_date - feature_date).days != 1:
            raise ValueError(
                f"Invalid target alignment: "
                f"{feature_date} -> {target_date}"
            )

    print(" Target alignment verified: features(t) -> target(t+1)")


def calculate_metrics(predictions, actuals):
    """Calculate MAE and RMSE."""
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

    return mae, rmse


def run_comparison():
    rows = load_dataset()

    feature_dates = np.array([row[0] for row in rows])
    target_dates = np.array([row[1] for row in rows])

    X = np.array([list(row[2:-1]) for row in rows], dtype=float)
    y = np.array([row[-1] for row in rows], dtype=float)

    print(f"Retrieved {len(rows)} samples")
    print(f"Feature matrix shape: {X.shape}")

    validate_alignment(feature_dates, target_dates)

    if len(X) <= MIN_TRAIN_SIZE:
        raise RuntimeError(
            f"Need more than {MIN_TRAIN_SIZE} samples for walk-forward evaluation."
        )

    # Walk-forward evaluation

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

        # Naive baseline
        #
        # Predict tomorrow using today's PM2.5.
        # pm2_5_lag_1 at target t+1 = PM2.5(t)

        naive_prediction = X_test[0][NAIVE_COL]

        # LightGBM

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            free_raw_data=True,
        )

        model = lgb.train(
            PARAMS,
            train_data,
            num_boost_round=100,
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

    # Metrics

    naive_mae, naive_rmse = calculate_metrics(
        naive_predictions,
        actuals,
    )

    lightgbm_mae, lightgbm_rmse = calculate_metrics(
        lightgbm_predictions,
        actuals,
    )

    mae_improvement = (
        (naive_mae - lightgbm_mae) / naive_mae
    ) * 100

    rmse_improvement = (
        (naive_rmse - lightgbm_rmse) / naive_rmse
    ) * 100

    # Results

    print("\n" + "=" * 56)
    print("                 VERICAST")
    print("            OFFICIAL BACKTEST")
    print("=" * 56)

    print(f"City:                  {CITY}")
    print("Target:                Next-day PM2.5")
    print(f"Evaluation samples:    {len(actuals)}")
    print(
        f"Evaluation window:    "
        f"{evaluation_dates[0]} -> {evaluation_dates[-1]}"
    )

    print("\nModel Performance")
    print("-" * 56)
    print(
        f"{'Model':<20}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
    )
    print("-" * 56)

    print(
        f"{'Naive Baseline':<20}"
        f"{naive_mae:>12.4f}"
        f"{naive_rmse:>12.4f}"
    )

    print(
        f"{'LightGBM':<20}"
        f"{lightgbm_mae:>12.4f}"
        f"{lightgbm_rmse:>12.4f}"
    )

    print("-" * 56)

    print(f"\nMAE improvement:      {mae_improvement:.2f}%")
    print(f"RMSE improvement:     {rmse_improvement:.2f}%")

    winner = (
        "LightGBM"
        if lightgbm_mae < naive_mae
        else "Naive Baseline"
    )

    print(f"Winner:               {winner}")

    print("\nFirst 5 comparison predictions")
    print("-" * 80)

    for i in range(min(5, len(actuals))):
        print(
            f"{evaluation_dates[i]} | "
            f"Actual: {actuals[i]:>7.2f} | "
            f"Naive: {naive_predictions[i]:>7.2f} | "
            f"LightGBM: {lightgbm_predictions[i]:>7.2f}"
        )

    print("\nLast 5 comparison predictions")
    print("-" * 80)

    for i in range(max(0, len(actuals) - 5), len(actuals)):
        print(
            f"{evaluation_dates[i]} | "
            f"Actual: {actuals[i]:>7.2f} | "
            f"Naive: {naive_predictions[i]:>7.2f} | "
            f"LightGBM: {lightgbm_predictions[i]:>7.2f}"
        )

    print("\n Official VeriCast backtest completed.")

    return {
        "city": CITY,
        "samples": len(actuals),
        "start_date": str(evaluation_dates[0]),
        "end_date": str(evaluation_dates[-1]),
        "naive_mae": float(naive_mae),
        "naive_rmse": float(naive_rmse),
        "lightgbm_mae": float(lightgbm_mae),
        "lightgbm_rmse": float(lightgbm_rmse),
        "mae_improvement_percent": float(mae_improvement),
        "rmse_improvement_percent": float(rmse_improvement),
        "winner": winner,
    }


if __name__ == "__main__":
    run_comparison()