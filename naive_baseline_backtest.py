import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def naive_baseline_backtest():
    """Run walk-forward backtest for naive baseline (predict yesterday's PM2.5)"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Join observations and features to get target and prediction
                query = """
                    SELECT
                        o.as_of,
                        o.pm2_5 AS actual_pm2_5,
                        f.pm2_5_lag_1 AS predicted_pm2_5
                    FROM observations o
                    JOIN features f ON o.city = f.city AND o.as_of = f.as_of
                    WHERE o.city = 'Nagpur'
                      AND f.pm2_5_lag_1 IS NOT NULL  -- Exclude first day (no lag)
                    ORDER BY o.as_of;
                """

                cur.execute(query)
                rows = cur.fetchall()

                if not rows:
                    print("ERROR: No data available for backtest")
                    return None

                print(f"Retrieved {len(rows)} days for backtest (excluding first day)")

                # Calculate errors
                errors = []
                squared_errors = []
                predictions = []
                actuals = []

                for as_of, actual, predicted in rows:
                    error = abs(actual - predicted)
                    squared_error = (actual - predicted) ** 2
                    errors.append(error)
                    squared_errors.append(squared_error)
                    predictions.append(predicted)
                    actuals.append(actual)

                # Calculate metrics
                mae = sum(errors) / len(errors)
                rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5

                print(f"\nNaive Baseline Backtest Results:")
                print(f"   Mean Absolute Error (MAE): {mae:.4f}")
                print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"   Number of predictions: {len(errors)}")

                # Show first few predictions for sanity check
                print(f"\nFirst 5 predictions:")
                for i in range(min(5, len(rows))):
                    as_of, actual, predicted = rows[i]
                    print(f"   Date: {as_of}, Actual: {actual:.2f}, Predicted: {predicted:.2f}, Error: {abs(actual-predicted):.2f}")

                # Show last few predictions
                print(f"\nLast 5 predictions:")
                for i in range(max(0, len(rows)-5), len(rows)):
                    as_of, actual, predicted = rows[i]
                    print(f"   Date: {as_of}, Actual: {actual:.2f}, Predicted: {predicted:.2f}, Error: {abs(actual-predicted):.2f}")

                return {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions,
                    'actuals': actuals,
                    'errors': errors,
                    'dates': [row[0] for row in rows]
                }

    except Exception as e:
        print(f"Error running backtest: {e}")
        raise

if __name__ == "__main__":
    results = naive_baseline_backtest()
    if results:
        print(f"\nBaseline MAE: {results['mae']:.4f} - This is the benchmark to beat.")