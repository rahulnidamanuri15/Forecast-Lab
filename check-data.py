import httpx

LAT, LON = 21.1463, 79.0849          #Change to your city.
START, END = "2023-08-01", "2025-08-01"

aq = httpx.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
    "latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
    "start_date": START, "end_date": END, "timezone": "UTC",
}, timeout=60).json()

wx = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": LAT, "longitude": LON,
    "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
    "start_date": START, "end_date": END, "timezone": "UTC",
}, timeout=60).json()

pm = aq["hourly"]["pm2_5"]
print(f"AQI hours: {len(pm)}, missing: {sum(v is None for v in pm)}")
print(f"Weather days: {len(wx['daily']['time'])}")
print("GO" if len(pm) > 15000 and len(wx["daily"]["time"]) > 700 else "PICK ANOTHER CITY")