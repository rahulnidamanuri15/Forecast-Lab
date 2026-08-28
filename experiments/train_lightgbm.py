import os
import sys
import psycopg
import numpy as np
import lightgbm as lgb
from dotenv import load_dotenv
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vericast.pm25.train import PARAMS, DATASET_SQL, CITY  # noqa: E402

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def train_lightgbm_walkforward():
    """Train LightGBM model with walk-forward validation"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(DATASET_SQL, (CITY,))
                rows = cur.fetchall()

                if not rows:
                    print("ERROR: No data available for training")
                    return None

                print(f"Retrieved {len(rows)} samples for LightGBM training")

                # Prepare data
                X = []
                y = []
                feature_dates = []
                target_dates = []

                for row in rows:
                    feature_dates.append(row[0])
                    target_dates.append(row[1])  # as_of
                    # Features (all columns except as_of and target)
                    features = list(row[2:-1])  # Everything except first (as_of) and last (target)
                    X.append(features)
                    y.append(row[-1])  # target_pm2_5

                X = np.array(X, dtype=float)
                y = np.array(y, dtype=float)

                feature_dates = np.array(feature_dates)
                target_dates = np.array(target_dates)

                print(f"Feature matrix shape: {X.shape}")

                # Verify that every feature row predicts exactly the next day
                for feature_date, target_date in zip(feature_dates, target_dates):
                    if target_date != feature_date + timedelta(days=1):
                        raise ValueError(
                            f"Invalid target alignment: "
                            f"{feature_date} -> {target_date}"
                        )

                print(" Target alignment verified: features(t) -> target(t+1)")

                # Walk-forward validation
                predictions = []
                actuals = []
                errors = []

                # Start training from index 0, predict index 1, then expand window
                min_train_size = 30  # Minimum training size before we start predicting

                for i in range(min_train_size, len(X)):
                    # Train on data[0:i], predict data[i]
                    X_train = X[:i]
                    y_train = y[:i]
                    X_test = X[i:i+1]  # Just the current sample
                    y_true = y[i:i+1]

                    # Create LightGBM dataset
                    train_data = lgb.Dataset(X_train, label=y_train)

                    # Train model
                    model = lgb.train(PARAMS, train_data, num_boost_round=100)

                    # Predict
                    y_pred = model.predict(X_test)[0]

                    # Store results
                    predictions.append(y_pred)
                    actuals.append(y_true[0])
                    errors.append(abs(y_pred - y_true[0]))

                    # Progress indicator
                    if (i - min_train_size) % 50 == 0:
                        print(f"   Processed {i - min_train_size + 1}/{len(X) - min_train_size} samples")

                # Calculate final metrics
                mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
                rmse = np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)]))

                print(f"\nLightGBM Walk-Forward Backtest Results:")
                print(f"   Mean Absolute Error (MAE): {mae:.4f}")
                print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"   Number of predictions: {len(errors)}")
                print("   Baseline MAE: calculated separately")
                print("   Improvement: calculated after comparing identical prediction dates")

                # Show first few predictions
                print(f"\nFirst 5 predictions:")
                for i in range(min(5, len(predictions))):
                    print(f"   Date: {target_dates[min_train_size + i]}, Actual: {actuals[i]:.2f}, Predicted: {predictions[i]:.2f}, Error: {errors[i]:.2f}")

                # Show last few predictions
                print(f"\nLast 5 predictions:")
                for i in range(max(0, len(predictions)-5), len(predictions)):
                    print(f"   Date: {target_dates[min_train_size + i]}, Actual: {actuals[i]:.2f}, Predicted: {predictions[i]:.2f}, Error: {errors[i]:.2f}")

                return {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions,
                    'actuals': actuals,
                    'errors': errors,
                    'dates': target_dates[min_train_size:].tolist()
                }

    except Exception as e:
        print(f"Error in LightGBM training: {e}")
        raise

if __name__ == "__main__":
    results = train_lightgbm_walkforward()
    if results:
        print(f"\nLightGBM MAE: {results['mae']:.4f}")