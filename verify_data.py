import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def verify_data():
    """Verify the data was inserted correctly"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Count total records
                cur.execute("SELECT COUNT(*) FROM observations")
                total_count = cur.fetchone()[0]
                print(f"Total records in observations table: {total_count}")

                # Check for Nagpur specifically
                cur.execute("SELECT COUNT(*) FROM observations WHERE city = 'Nagpur'")
                nagpur_count = cur.fetchone()[0]
                print(f"Nagpur records: {nagpur_count}")

                # Show date range
                cur.execute("SELECT MIN(as_of), MAX(as_of) FROM observations WHERE city = 'Nagpur'")
                min_date, max_date = cur.fetchone()
                print(f"Date range: {min_date} to {max_date}")

                # Show sample data
                cur.execute("""
                    SELECT as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum
                    FROM observations
                    WHERE city = 'Nagpur'
                    ORDER BY as_of
                    LIMIT 5
                """)
                samples = cur.fetchall()
                print("\nSample data (first 5 records):")
                for sample in samples:
                    print(f"  Date: {sample[0]}, PM2.5: {sample[1]}, PM10: {sample[2]}, "
                          f"Temp: {sample[3]}, Wind: {sample[4]}, Precip: {sample[5]}")

                # Check for any null values in key fields
                cur.execute("""
                    SELECT
                        SUM(CASE WHEN pm2_5 IS NULL THEN 1 ELSE 0 END) as null_pm2_5,
                        SUM(CASE WHEN pm10 IS NULL THEN 1 ELSE 0 END) as null_pm10,
                        SUM(CASE WHEN temperature_2m_mean IS NULL THEN 1 ELSE 0 END) as null_temp,
                        SUM(CASE WHEN wind_speed_10m_max IS NULL THEN 1 ELSE 0 END) as null_wind,
                        SUM(CASE WHEN precipitation_sum IS NULL THEN 1 ELSE 0 END) as null_precip
                    FROM observations WHERE city = 'Nagpur'
                """)
                null_counts = cur.fetchone()
                print(f"\nNull counts:")
                print(f"  PM2.5: {null_counts[0]}")
                print(f"  PM10: {null_counts[1]}")
                print(f"  Temperature: {null_counts[2]}")
                print(f"  Wind: {null_counts[3]}")
                print(f"  Precipitation: {null_counts[4]}")

    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        raise

if __name__ == "__main__":
    verify_data()