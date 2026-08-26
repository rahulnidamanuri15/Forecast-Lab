"""Ingest daily Maharashtra peak electricity demand + regional temperature.

Source: the Grid-Sentinel community mirror of Grid-India/NLDC daily PSP reports,
as an already-parsed CSV. Grid-India's own report.grid-india.in endpoints are
unreachable from CI (HTTP 000 on every probe) and would need pandas + openpyxl +
pdfplumber to read - three dependencies this repo deliberately dropped. The
mirror is stdlib `csv` over the already-installed httpx: zero new dependencies.

The tradeoff, stated plainly because the accuracy record depends on it: this is
a third-party mirror, not the operator. It runs 2-4 days behind real time and
skips the occasional day, so `stale_days` of 2-4 is NORMAL here (unlike PM2.5,
where Open-Meteo has yesterday by 05:00 UTC). Upgrade path: if Grid-India ever
exposes a reachable machine-readable endpoint, fetch_demand() below is the only
function that needs to change.
"""
import os
import csv
import io
import httpx
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

import local_time

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = "Maharashtra"

DEMAND_CSV_URL = (
    "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/"
    "Dataset/study3_states.csv"
)

# Maharashtra's temperature as the unweighted mean of its three largest cities.
# ponytail: unweighted 3-city mean. Population-weight it (or add a 4th city) only
# if temp features top LightGBM's importance and the margin over seasonal_naive stalls.
CITIES = [
    (19.0760, 72.8777),  # Mumbai
    (18.5204, 73.8567),  # Pune
    (21.1458, 79.0882),  # Nagpur
]

# The mirror covers 2018-12-31 onwards, but 2023-01-01 is where the Maharashtra
# series is gap-free and null-free (verified: 1328 days, one gap 2025-05-21->24).
INITIAL_START = "2023-01-01"


def get_last_observed_date():
    """Return the most recent as_of already stored for this state, or None."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(as_of) FROM electricity_observations WHERE state = %s",
                (STATE,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def resolve_date_range():
    """First run backfills from INITIAL_START; later runs resume the day after
    the latest stored observation. End is capped at yesterday - the demand mirror
    is historical and typically 2-4 days behind, so most runs find nothing new."""
    last_date = get_last_observed_date()

    if last_date is None:
        start = datetime.strptime(INITIAL_START, "%Y-%m-%d").date()
    else:
        start = last_date + timedelta(days=1)

    return start, local_time.yesterday()


def fetch_demand(start_date, end_date):
    """Fetch the mirror and return {date_str: (peak_demand_mw, energy_met_mu)}
    for Maharashtra rows inside the range. The only function that touches the
    demand source."""
    print(f"Fetching demand mirror ({DEMAND_CSV_URL.rsplit('/', 1)[-1]})...")
    # ponytail: refetches the whole ~6MB mirror each run to pull a handful of rows.
    # Byte-range or a local cache only if the daily job starts timing out.
    response = httpx.get(DEMAND_CSV_URL, timeout=180, follow_redirects=True)
    response.raise_for_status()

    start_str, end_str = start_date.isoformat(), end_date.isoformat()
    demand = {}
    skipped = 0

    for row in csv.DictReader(io.StringIO(response.text)):
        if row["state"].strip() != STATE:
            continue
        date_str = row["date"].strip()
        if not (start_str <= date_str <= end_str):
            continue

        # peak_demand_mw is NOT NULL in the schema, so a blank one is a skip, not
        # a NULL row - a demand row without demand has nothing to score against.
        raw_mw = row["max_demand_met_mw"].strip()
        if not raw_mw:
            skipped += 1
            continue

        raw_mu = row["energy_met_mu"].strip()
        demand[date_str] = (float(raw_mw), float(raw_mu) if raw_mu else None)

    print(f"{STATE} demand days in range: {len(demand)}"
          + (f" ({skipped} skipped for missing MW)" if skipped else ""))
    return demand


def fetch_temperature(start_date, end_date):
    """Fetch daily mean/max temperature for the three cities in one call and
    return {date_str: (mean, max)} of their averages.

    Comma-separated coordinates make Open-Meteo return a JSON *array* of
    per-location objects, each with its own `daily` block."""
    print("Fetching regional temperature...")
    response = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": ",".join(str(lat) for lat, _ in CITIES),
        "longitude": ",".join(str(lon) for _, lon in CITIES),
        "daily": "temperature_2m_mean,temperature_2m_max",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }, timeout=60)
    response.raise_for_status()
    payload = response.json()

    # A single location returns a bare object; multiple return a list.
    locations = payload if isinstance(payload, list) else [payload]

    sums = {}
    for location in locations:
        daily = location["daily"]
        for i, date_str in enumerate(daily["time"]):
            mean_v, max_v = daily["temperature_2m_mean"][i], daily["temperature_2m_max"][i]
            acc = sums.setdefault(date_str, [0.0, 0, 0.0, 0])
            if mean_v is not None:
                acc[0] += mean_v
                acc[1] += 1
            if max_v is not None:
                acc[2] += max_v
                acc[3] += 1

    temps = {
        date_str: (
            acc[0] / acc[1] if acc[1] else None,
            acc[2] / acc[3] if acc[3] else None,
        )
        for date_str, acc in sums.items()
    }
    print(f"Temperature days: {len(temps)} (mean of {len(locations)} cities)")
    return temps


def insert_observations(records):
    insert_sql = """
    INSERT INTO electricity_observations
        (state, as_of, peak_demand_mw, energy_met_mu, temperature_2m_mean, temperature_2m_max)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (state, as_of) DO UPDATE SET
        peak_demand_mw = EXCLUDED.peak_demand_mw,
        energy_met_mu = EXCLUDED.energy_met_mu,
        temperature_2m_mean = EXCLUDED.temperature_2m_mean,
        temperature_2m_max = EXCLUDED.temperature_2m_max,
        created_at = CURRENT_TIMESTAMP;
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, records)
                conn.commit()
                print(f"Successfully inserted {cur.rowcount} records")

                cur.execute(
                    "SELECT COUNT(*), MIN(as_of), MAX(as_of) FROM electricity_observations "
                    "WHERE state = %s", (STATE,))
                count, first, last = cur.fetchone()
                print(f"Total {STATE} records in database: {count} ({first} -> {last})")
    except Exception as e:
        print(f"Error inserting electricity observations: {e}")
        raise


def main():
    print("Starting electricity ingestion...")

    start_date, end_date = resolve_date_range()

    if start_date > end_date:
        print(
            f"Nothing new to fetch: latest data already covers through "
            f"{start_date - timedelta(days=1)}, and end date is capped at "
            f"{end_date} (yesterday). Skipping."
        )
        return

    print(f"Fetching data for {start_date} -> {end_date}")

    demand = fetch_demand(start_date, end_date)
    if not demand:
        # Expected on most days: the mirror runs 2-4 days behind, so there is
        # frequently no new demand row even though the date range is non-empty.
        print(f"No new {STATE} demand rows in range (mirror lags real time by "
              f"a few days); nothing to insert.")
        return

    temps = fetch_temperature(start_date, end_date)

    records = [
        (STATE, date_str, peak_mw, energy_mu, *temps.get(date_str, (None, None)))
        for date_str, (peak_mw, energy_mu) in sorted(demand.items())
    ]

    print(f"Prepared {len(records)} daily records for insertion")
    insert_observations(records)
    print("Ingestion completed!")


if __name__ == "__main__":
    main()
