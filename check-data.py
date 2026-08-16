import httpx

LAT, LON = 21.1463, 79.0849          # Nagpur, India
START, END = "2023-08-01", "2025-08-01"

# Increase timeout and add retry logic for robustness
def fetch_with_retry(url, params, max_retries=3):
    for i in range(max_retries):
        try:
            response = httpx.get(url, params=params, timeout=120.0)
            response.raise_for_status()
            return response.json()
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if i == max_retries - 1:
                raise
            print(f"Timeout on attempt {i+1}, retrying...")
        except httpx.HTTPStatusError as e:
            raise

print("Fetching air quality data...")
aq = fetch_with_retry("https://air-quality-api.open-meteo.com/v1/air-quality", {
    "latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
    "start_date": START, "end_date": END, "timezone": "UTC",
})

print("Fetching weather data...")
wx = fetch_with_retry("https://archive-api.open-meteo.com/v1/archive", {
    "latitude": LAT, "longitude": LON,
    "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
    "start_date": START, "end_date": END, "timezone": "UTC",
})

pm = aq["hourly"]["pm2_5"]
print(f"AQI hours: {len(pm)}, missing: {sum(v is None for v in pm)}")
print(f"Weather days: {len(wx['daily']['time'])}")
print("GO" if len(pm) > 15000 and len(wx["daily"]["time"]) > 700 else "PICK ANOTHER CITY")