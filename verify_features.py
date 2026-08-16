import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def verify_features():
    """Verify the features were engineered correctly"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Count total records
                cur.execute("SELECT COUNT(*) FROM features")
                total_count = cur.fetchone()[0]
                print(f"Total records in features table: {total_count}")

                # Check for Nagpur specifically
                cur.execute("SELECT COUNT(*) FROM features WHERE city = 'Nagpur'")
                nagpur_count = cur.fetchone()[0]
                print(f"Nagpur records: {nagpur_count}")

                # Show date range
                cur.execute("SELECT MIN(as_of), MAX(as_of) FROM features WHERE city = 'Nagpur'")
                min_date, max_date = cur.fetchone()
                print(f"Date range: {min_date} to {max_date}")

                # Show sample data (first few records where we have lagged values)
                cur.execute("""
                    SELECT as_of, pm2_5_lag_1, pm10_lag_1, temperature_lag_1,
                           pm2_5_roll_7, day_of_week, is_weekend
                    FROM features
                    WHERE city = 'Nagpur' AND pm2_5_lag_1 IS NOT NULL
                    ORDER BY as_of
                    LIMIT 5
                """)
                samples = cur.fetchall()
                print("\nSample data (first 5 records with lagged values):")
                for sample in samples:
                    pm2_5_lag = f"{sample[1]:.2f}" if sample[1] is not None else "None"
                    pm10_lag = f"{sample[2]:.2f}" if sample[2] is not None else "None"
                    temp_lag = f"{sample[3]:.2f}" if sample[3] is not None else "None"
                    roll_7 = f"{sample[4]:.2f}" if sample[4] is not None else "None"
                    print(f"  Date: {sample[0]}, PM2.5_lag_1: {pm2_5_lag}, PM10_lag_1: {pm10_lag}, "
                          f"Temp_lag_1: {temp_lag}, PM2.5_roll_7: {roll_7}, "
                          f"Day_of_week: {sample[5]}, Is_weekend: {sample[6]}")

                # Check for null values in key fields (first row should have nulls for lagged features)
                cur.execute("""
                    SELECT
                        SUM(CASE WHEN pm2_5_lag_1 IS NULL THEN 1 ELSE 0 END) as null_pm2_5_lag,
                        SUM(CASE WHEN pm10_lag_1 IS NULL THEN 1 ELSE 0 END) as null_pm10_lag,
                        SUM(CASE WHEN temperature_lag_1 IS NULL THEN 1 ELSE 0 END) as null_temp_lag,
                        SUM(CASE WHEN wind_speed_lag_1 IS NULL THEN 1 ELSE 0 END) as null_wind_lag,
                        SUM(CASE WHEN precipitation_lag_1 IS NULL THEN 1 ELSE 0 END) as null_precip_lag
                    FROM features WHERE city = 'Nagpur'
                """)
                null_counts = cur.fetchone()
                print(f"\nLagged feature null counts (first rows should have nulls):")
                print(f"  PM2.5_lag_1: {null_counts[0]}")
                print(f"  PM10_lag_1: {null_counts[1]}")
                print(f"  Temperature_lag_1: {null_counts[2]}")
                print(f"  Wind_lag_1: {null_counts[3]}")
                print(f"  Precipitation_lag_1: {null_counts[4]}")

                # Check rolling averages (should start having values after 7th and 30th day)
                cur.execute("""
                    SELECT
                        SUM(CASE WHEN pm2_5_roll_7 IS NULL THEN 1 ELSE 0 END) as null_roll_7,
                        SUM(CASE WHEN pm2_5_roll_30 IS NULL THEN 1 ELSE 0 END) as null_roll_30
                    FROM features WHERE city = 'Nagpur'
                """)
                roll_nulls = cur.fetchone()
                print(f"\nRolling average null counts:")
                print(f"  PM2.5_roll_7: {roll_nulls[0]} (should be 6 for first week)")
                print(f"  PM2.5_roll_30: {roll_nulls[1]} (should be 29 for first month)")

    except Exception as e:
        print(f"❌ Error verifying features: {e}")
        raise

if __name__ == "__main__":
    verify_features()