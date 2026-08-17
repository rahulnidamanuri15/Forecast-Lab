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
                            f.as_of AS feature_date,
                            o.as_of AS target_date,
                            f.pm2_5_lag_1 AS predicted_pm2_5,
                            o.pm2_5 AS actual_pm2_5
                        FROM features f
                        JOIN observations o
                        ON o.city = f.city
                        AND o.as_of = f.as_of + INTERVAL '1 day'
                        WHERE f.city = 'Nagpur'
                        AND f.pm2_5_lag_1 IS NOT NULL
                        AND o.pm2_5 IS NOT NULL
                        ORDER BY f.as_of;
                    """

                cur.execute(query)
                rows = cur.fetchall()

                MIN_TRAIN_SIZE = 30
                rows = rows[MIN_TRAIN_SIZE:]

                if not rows:
                    print("ERROR: No data available for backtest")
                    return None

                print(f"Retrieved {len(rows)} days for backtest")

                # Calculate errors
                errors = []
                squared_errors = []
                predictions = []
                actuals = []

                for feature_date, target_date, predicted, actual in rows:
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
                    feature_date, target_date, predicted, actual = rows[i]

                    print(
                        f"   Feature Date: {feature_date}, "
                        f"Target Date: {target_date}, "
                        f"Actual: {actual:.2f}, "
                        f"Predicted: {predicted:.2f}, "
                        f"Error: {abs(actual - predicted):.2f}"
                    )

                # Show last few predictions
                print(f"\nLast 5 predictions:")
                for i in range(max(0, len(rows) - 5), len(rows)):
                    feature_date, target_date, predicted, actual = rows[i]

                    print(
                        f"   Feature Date: {feature_date}, "
                        f"Target Date: {target_date}, "
                        f"Actual: {actual:.2f}, "
                        f"Predicted: {predicted:.2f}, "
                        f"Error: {abs(actual - predicted):.2f}"
                    )
                return {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions,
                    'actuals': actuals,
                    'errors': errors,
                    'feature_dates': [row[0] for row in rows],
                    'target_dates': [row[1] for row in rows]
                }

    except Exception as e:
        print(f"Error running backtest: {e}")
        raise

if __name__ == "__main__":
    results = naive_baseline_backtest()
    if results:
        print(f"\nBaseline MAE: {results['mae']:.4f} - This is the benchmark to beat.")