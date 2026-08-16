import os
import psycopg
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def score_yesterday_predictions():
    """Score yesterday's predictions against actuals and store performance metrics"""
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()

        # Yesterday's date (in UTC to match our data)
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        today = datetime.utcnow().date()

        print(f"Scoring predictions for {yesterday} (actuals from observations)")

        # Get predictions for yesterday that have actuals
        cur.execute("""
            SELECT p.predicted_pm2_5, o.pm2_5 AS actual_pm2_5, p.model
            FROM predictions p
            JOIN observations o ON p.city = o.city AND p.forecast_date = o.as_of
            WHERE p.city = 'Nagpur'
              AND p.forecast_date = %s
              AND p.actual_pm2_5 IS NOT NULL
              AND o.pm2_5 IS NOT NULL
        """, (yesterday,))

        rows = cur.fetchall()

        if not rows:
            print(f"⚠️ No predictions with actuals found for {yesterday}")
            cur.close()
            conn.close()
            return False

        print(f"Found {len(rows)} predictions to score for {yesterday}")

        # Group by model
        model_data = {}
        for predicted, actual, model in rows:
            if model not in model_data:
                model_data[model] = {'predicted': [], 'actual': []}
            model_data[model]['predicted'].append(predicted)
            model_data[model]['actual'].append(actual)

        # Calculate and store performance for each model
        for model, data in model_data.items():
            predicted_vals = np.array(data['predicted'])
            actual_vals = np.array(data['actual'])

            # Calculate MAE and RMSE
            mae = np.mean(np.abs(predicted_vals - actual_vals))
            rmse = np.sqrt(np.mean((predicted_vals - actual_vals) ** 2))
            sample_size = len(predicted_vals)

            # Insert or update performance record
            upsert_sql = """
            INSERT INTO model_performance (score_date, model, mae, rmse, sample_size)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (score_date, model) DO UPDATE SET
                mae = EXCLUDED.mae,
                rmse = EXCLUDED.rmse,
                sample_size = EXCLUDED.sample_size,
                created_at = CURRENT_TIMESTAMP;
            """

            cur.execute(upsert_sql, (yesterday, model, mae, rmse, sample_size))
            conn.commit()

            print(f"✅ Scored {model} for {yesterday}: MAE={mae:.4f}, RMSE={rmse:.4f} (n={sample_size})")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error scoring predictions: {e}")
        raise

if __name__ == "__main__":
    score_yesterday_predictions()