import os
import psycopg
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def engineer_features():
    """Engineer features from observations data and insert into features table"""

    # First, get all observations ordered by date
    select_sql = """
    SELECT
        as_of,
        pm2_5,
        pm10,
        temperature_2m_mean,
        wind_speed_10m_max,
        precipitation_sum
    FROM observations
    WHERE city = 'Nagpur'
    ORDER BY as_of;
    """

    # Insert features SQL
    insert_sql = """
    INSERT INTO features (
        city, as_of,
        pm2_5_lag_1, pm10_lag_1, temperature_lag_1, wind_speed_lag_1, precipitation_lag_1,
        pm2_5_roll_7, pm2_5_roll_30, pm10_roll_7, pm10_roll_30,
        day_of_week, month, is_weekend,
        temperature_2m_mean, wind_speed_10m_max, precipitation_sum
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (city, as_of) DO UPDATE SET
        pm2_5_lag_1 = EXCLUDED.pm2_5_lag_1,
        pm10_lag_1 = EXCLUDED.pm10_lag_1,
        temperature_lag_1 = EXCLUDED.temperature_lag_1,
        wind_speed_lag_1 = EXCLUDED.wind_speed_lag_1,
        precipitation_lag_1 = EXCLUDED.precipitation_lag_1,
        pm2_5_roll_7 = EXCLUDED.pm2_5_roll_7,
        pm2_5_roll_30 = EXCLUDED.pm2_5_roll_30,
        pm10_roll_7 = EXCLUDED.pm10_roll_7,
        pm10_roll_30 = EXCLUDED.pm10_roll_30,
        day_of_week = EXCLUDED.day_of_week,
        month = EXCLUDED.month,
        is_weekend = EXCLUDED.is_weekend,
        temperature_2m_mean = EXCLUDED.temperature_2m_mean,
        wind_speed_10m_max = EXCLUDED.wind_speed_10m_max,
        precipitation_sum = EXCLUDED.precipitation_sum,
        created_at = CURRENT_TIMESTAMP;
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Get all observations
                cur.execute(select_sql)
                rows = cur.fetchall()

                print(f"Retrieved {len(rows)} observation records")

                # Process each row to engineer features
                features_to_insert = []

                for i, row in enumerate(rows):
                    as_of, pm2_5, pm10, temp, wind, precip = row

                    # Calculate lagged features (previous day's values).
                    # If there's a gap in the observations (e.g. a missing day
                    # from an upstream API outage), the row immediately before
                    # this one in `rows` is NOT actually yesterday, so lag
                    # features would silently pull in stale/wrong data. Rather
                    # than crash the whole run, we skip the lag for this one
                    # row (leave it NULL) and keep going.
                    if i > 0:
                        prev_row = rows[i-1]
                        prev_date = prev_row[0]

                        if as_of - prev_date != timedelta(days=1):
                            print(
                                f"  Warning: date gap detected ({prev_date} -> {as_of}); "
                                f"lag features for {as_of} will be NULL"
                            )
                            pm2_5_lag_1 = None
                            pm10_lag_1 = None
                            temperature_lag_1 = None
                            wind_speed_lag_1 = None
                            precipitation_lag_1 = None
                        else:
                            pm2_5_lag_1 = prev_row[1]  # pm2_5 from previous day
                            pm10_lag_1 = prev_row[2]   # pm10 from previous day
                            temperature_lag_1 = prev_row[3]  # temperature from previous day
                            wind_speed_lag_1 = prev_row[4]   # wind from previous day
                            precipitation_lag_1 = prev_row[5]  # precip from previous day
                    else:
                        pm2_5_lag_1 = None
                        pm10_lag_1 = None
                        temperature_lag_1 = None
                        wind_speed_lag_1 = None
                        precipitation_lag_1 = None

                    # Calculate rolling averages (7-day and 30-day)
                    # For PM2.5
                    pm2_5_roll_7 = None
                    pm2_5_roll_30 = None
                    pm10_roll_7 = None
                    pm10_roll_30 = None

                    if i >= 6:  # At least 7 days for 7-day rolling
                        pm2_5_values_7 = [rows[j][1] for j in range(i-6, i+1) if rows[j][1] is not None]
                        if pm2_5_values_7:
                            pm2_5_roll_7 = sum(pm2_5_values_7) / len(pm2_5_values_7)

                    if i >= 29:  # At least 30 days for 30-day rolling
                        pm2_5_values_30 = [rows[j][1] for j in range(i-29, i+1) if rows[j][1] is not None]
                        if pm2_5_values_30:
                            pm2_5_roll_30 = sum(pm2_5_values_30) / len(pm2_5_values_30)

                    if i >= 6:  # At least 7 days for 7-day rolling
                        pm10_values_7 = [rows[j][2] for j in range(i-6, i+1) if rows[j][2] is not None]
                        if pm10_values_7:
                            pm10_roll_7 = sum(pm10_values_7) / len(pm10_values_7)

                    if i >= 29:  # At least 30 days for 30-day rolling
                        pm10_values_30 = [rows[j][2] for j in range(i-29, i+1) if rows[j][2] is not None]
                        if pm10_values_30:
                            pm10_roll_30 = sum(pm10_values_30) / len(pm10_values_30)

                    # Time-based features
                    dt = as_of if isinstance(as_of, datetime) else datetime.combine(as_of, datetime.min.time())
                    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
                    month = dt.month  # 1-12
                    is_weekend = day_of_week >= 5  # Saturday or Sunday

                    # Same-day weather features (already have these)
                    temperature_2m_mean = temp
                    wind_speed_10m_max = wind
                    precipitation_sum = precip

                    features_to_insert.append((
                        "Nagpur",  # city
                        as_of,     # as_of
                        pm2_5_lag_1,
                        pm10_lag_1,
                        temperature_lag_1,
                        wind_speed_lag_1,
                        precipitation_lag_1,
                        pm2_5_roll_7,
                        pm2_5_roll_30,
                        pm10_roll_7,
                        pm10_roll_30,
                        day_of_week,
                        month,
                        is_weekend,
                        temperature_2m_mean,
                        wind_speed_10m_max,
                        precipitation_sum
                    ))

                # Batch insert features
                if features_to_insert:
                    cur.executemany(insert_sql, features_to_insert)
                    conn.commit()
                    print(f"Successfully inserted {len(features_to_insert)} feature records")

                    # Verify insertion
                    cur.execute("SELECT COUNT(*) FROM features WHERE city = 'Nagpur'")
                    count = cur.fetchone()[0]
                    print(f"Total Nagpur feature records in database: {count}")
                else:
                    print("No features to insert")

    except Exception as e:
        print(f"Error engineering features: {e}")
        raise

if __name__ == "__main__":
    engineer_features()