import os
import httpx
import psycopg
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Configuration - same as check-data.py
LAT, LON = 21.1463, 79.0849          # Nagpur, India
START, END = "2023-08-01", "2025-08-01"

def fetch_and_aggregate_data():
    """Fetch hourly AQ and daily weather data, aggregate to daily level"""
    print("Fetching air quality data...")
    aq_response = httpx.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    aq_data = aq_response.json()

    print("Fetching weather data...")
    wx_response = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    wx_data = wx_response.json()

    # Extract hourly data
    times = aq_data["hourly"]["time"]
    pm2_5_values = aq_data["hourly"]["pm2_5"]
    pm10_values = aq_data["hourly"]["pm10"]

    # Extract daily weather data
    wx_times = wx_data["daily"]["time"]
    temp_values = wx_data["daily"]["temperature_2m_mean"]
    wind_values = wx_data["daily"]["wind_speed_10m_max"]
    precip_values = wx_data["daily"]["precipitation_sum"]

    print(f"AQI hours: {len(pm2_5_values)}, missing: {sum(v is None for v in pm2_5_values)}")
    print(f"Weather days: {len(wx_times)}")

    # Aggregate hourly to daily
    daily_data = {}

    for i, timestamp in enumerate(times):
        # Convert to date only (ignoring time)
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt.date().isoformat()

        if date_str not in daily_data:
            daily_data[date_str] = {
                'pm2_5_sum': 0,
                'pm2_5_count': 0,
                'pm10_sum': 0,
                'pm10_count': 0
            }

        if pm2_5_values[i] is not None:
            daily_data[date_str]['pm2_5_sum'] += pm2_5_values[i]
            daily_data[date_str]['pm2_5_count'] += 1

        if pm10_values[i] is not None:
            daily_data[date_str]['pm10_sum'] += pm10_values[i]
            daily_data[date_str]['pm10_count'] += 1

    # Calculate daily averages and prepare for insertion
    records_to_insert = []

    for date_str in sorted(daily_data.keys()):
        data = daily_data[date_str]

        # Find corresponding weather data
        try:
            wx_idx = wx_times.index(date_str)
            temp = temp_values[wx_idx]
            wind = wind_values[wx_idx]
            precip = precip_values[wx_idx]
        except ValueError:
            # Weather data might not exist for this date
            temp = wind = precip = None

        # Calculate averages (handle division by zero)
        pm2_5_avg = data['pm2_5_sum'] / data['pm2_5_count'] if data['pm2_5_count'] > 0 else None
        pm10_avg = data['pm10_sum'] / data['pm10_count'] if data['pm10_count'] > 0 else None

        records_to_insert.append((
            "Nagpur",  # city
            date_str,  # as_of
            pm2_5_avg,
            pm10_avg,
            temp,
            wind,
            precip
        ))

    print(f"Prepared {len(records_to_insert)} daily records for insertion")
    return records_to_insert

def insert_observations(records):
    """Insert aggregated observations into the database"""
    insert_sql = """
    INSERT INTO observations (city, as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (city, as_of) DO UPDATE SET
        pm2_5 = EXCLUDED.pm2_5,
        pm10 = EXCLUDED.pm10,
        temperature_2m_mean = EXCLUDED.temperature_2m_mean,
        wind_speed_10m_max = EXCLUDED.wind_speed_10m_max,
        precipitation_sum = EXCLUDED.precipitation_sum,
        created_at = CURRENT_TIMESTAMP;
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Execute batch insert
                cur.executemany(insert_sql, records)
                conn.commit()
                print(f"Successfully inserted {cur.rowcount} records")

                # Verify insertion
                cur.execute("SELECT COUNT(*) FROM observations WHERE city = 'Nagpur'")
                count = cur.fetchone()[0]
                print(f"Total Nagpur records in database: {count}")

    except Exception as e:
        print(f"Error inserting observations: {e}")
        raise

def main():
    """Main ingestion process"""
    print("Starting observations ingestion...")
    records = fetch_and_aggregate_data()
    insert_observations(records)
    print("Ingestion completed!")

if __name__ == "__main__":
    main()