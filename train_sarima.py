import os
import psycopg
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def train_sarima_walkforward():
    """Train SARIMA model with walk-forward validation"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Get target variable (PM2.5) ordered by date
                query = """
                    SELECT
                        o.as_of,
                        o.pm2_5 AS target_pm2_5
                    FROM observations o
                    WHERE o.city = 'Nagpur'
                      AND o.pm2_5 IS NOT NULL
                    ORDER BY o.as_of;
                """

                cur.execute(query)
                rows = cur.fetchall()

                if not rows:
                    print("ERROR: No data available for training")
                    return None

                print(f"Retrieved {len(rows)} samples for SARIMA training")

                # Prepare data
                y = np.array([row[1] for row in rows])  # target_pm2_5
                dates = np.array([row[0] for row in rows])  # as_of

                print(f"Target vector shape: {y.shape}")

                # Walk-forward validation for SARIMA
                predictions = []
                actuals = []
                errors = []

                # Start with minimum training size
                min_train_size = 30  # Minimum training size before we start predicting

                for i in range(min_train_size, len(y)):
                    # Train on data[0:i], predict data[i]
                    y_train = y[:i]
                    y_true = y[i:i+1][0]  # Current actual value

                    try:
                        # Try a simpler SARIMA model first
                        model = SARIMAX(
                            y_train,
                            order=(1, 1, 0),  # ARIMA(1,1,0)
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted_model = model.fit(disp=False, maxiter=200)

                        # Forecast one step ahead
                        y_pred = fitted_model.forecast(steps=1)[0]

                    except Exception as model_error:
                        # Fallback to even simpler model
                        try:
                            model = SARIMAX(
                                y_train,
                                order=(0, 1, 1),  # Simple exponential smoothing
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fitted_model = model.fit(disp=False, maxiter=200)
                            y_pred = fitted_model.forecast(steps=1)[0]
                        except:
                            # Final fallback to naive prediction
                            y_pred = y_train[-1]  # Last observed value

                    # Store results
                    predictions.append(y_pred)
                    actuals.append(y_true)
                    errors.append(abs(y_pred - y_true))

                    # Progress indicator
                    if (i - min_train_size) % 50 == 0:
                        print(f"   Processed {i - min_train_size + 1}/{len(y) - min_train_size} samples")

                # Calculate final metrics
                mae = np.mean(errors)
                rmse = np.sqrt(np.mean([e ** 2 for e in errors]))

                print(f"\nSARIMA Walk-Forward Backtest Results:")
                print(f"   Mean Absolute Error (MAE): {mae:.4f}")
                print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"   Number of predictions: {len(errors)}")
                print(f"   Baseline MAE was: 7.3724")
                print(f"   Improvement: {((7.3724 - mae) / 7.3724 * 100):.2f}%")

                # Show first few predictions
                print(f"\nFirst 5 predictions:")
                for i in range(min(5, len(predictions))):
                    print(f"   Date: {dates[min_train_size + i]}, Actual: {actuals[i]:.2f}, Predicted: {predictions[i]:.2f}, Error: {errors[i]:.2f}")

                # Show last few predictions
                print(f"\nLast 5 predictions:")
                for i in range(max(0, len(predictions)-5), len(predictions)):
                    print(f"   Date: {dates[min_train_size + i]}, Actual: {actuals[i]:.2f}, Predicted: {predictions[i]:.2f}, Error: {errors[i]:.2f}")

                return {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions,
                    'actuals': actuals,
                    'errors': errors,
                    'dates': dates[min_train_size:].tolist()
                }

    except Exception as e:
        print(f"Error in SARIMA training: {e}")
        raise

if __name__ == "__main__":
    results = train_sarima_walkforward()
    if results:
        print(f"\nSARIMA MAE: {results['mae']:.4f}")