import os
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def make_daily_prediction():
    """Make a daily prediction for tomorrow using naive baseline and store it"""
    try:
        conn = psycopg.connect(DATABASE_URL)
        cur = conn.cursor()

        # Get today's date (in UTC to match our data)
        today = datetime.utcnow().date()
        tomorrow = today + timedelta(days=1)
        yesterday = today - timedelta(days=1)

        print(f"Making prediction for {tomorrow} based on {today}'s observation")

        # Get today's observation (most recent)
        cur.execute("""
            SELECT as_of, pm2_5
            FROM observations
            WHERE city = 'Nagpur'
            ORDER BY as_of DESC
            LIMIT 1
        """)

        row = cur.fetchone()
        if not row:
            raise Exception("No observations found")

        as_of, pm2_5 = row

        # Verify that as_of is today (we should have today's data)
        if as_of != today:
            print(f"Warning: Most recent observation is from {as_of}, not today ({today})")
            # Still proceed with what we have

        # Insert or update the prediction for tomorrow
        upsert_sql = """
        INSERT INTO predictions (city, forecast_date, predicted_pm2_5, model)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (city, forecast_date) DO UPDATE SET
            predicted_pm2_5 = EXCLUDED.predicted_pm2_5,
            model = EXCLUDED.model,
            created_at = CURRENT_TIMESTAMP;
        """

        cur.execute(upsert_sql, ("Nagpur", tomorrow, pm2_5, "naive_baseline"))
        conn.commit()

        print(f"✅ Stored prediction for {tomorrow}: {pm2_5:.2f} PM2.5")

        # Now check if we have actuals for yesterday's forecast and update those records
        # This handles the case where yesterday's forecast can now be scored
        cur.execute("""
            SELECT forecast_date, predicted_pm2_5
            FROM predictions
            WHERE city = 'Nagpur'
              AND forecast_date = %s
              AND actual_pm2_5 IS NULL
        """, (yesterday,))

        prediction_row = cur.fetchone()
        if prediction_row:
            forecast_date, predicted_pm2_5 = prediction_row

            # Get the actual PM2.5 for yesterday from observations
            cur.execute("""
                SELECT pm2_5
                FROM observations
                WHERE city = 'Nagpur'
                  AND as_of = %s
            """, (yesterday,))

            actual_row = cur.fetchone()
            if actual_row:
                actual_pm2_5 = actual_row[0]

                # Update the prediction with the actual value
                update_sql = """
                UPDATE predictions
                SET actual_pm2_5 = %s
                WHERE city = 'Nagpur'
                  AND forecast_date = %s
                """

                cur.execute(update_sql, (actual_pm2_5, yesterday))
                conn.commit()

                error = abs(actual_pm2_5 - predicted_pm2_5)
                print(f"✅ Updated forecast for {yesterday}: predicted={predicted_pm2_5:.2f}, actual={actual_pm2_5:.2f}, error={error:.2f}")
            else:
                print(f"⚠️ No actual observation found for {yesterday} yet")
        else:
            print(f"ℹ️ No pending forecast to update for {yesterday}")

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        raise

if __name__ == "__main__":
    make_daily_prediction()