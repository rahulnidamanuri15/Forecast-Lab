import os
import httpx
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

from vericast import PM25_MAX, PM25_MIN, RESCAN_DAYS, local_time, resume_start

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Configuration
LAT, LON = 21.1463, 79.0849          # Nagpur, India
CITY = os.getenv("CITY", "Nagpur")

# Hours of hourly data a day needs before its mean is called a daily mean. One
# hour averaged alone is indistinguishable downstream from 24, and it feeds the
# lag and rolling features. ponytail: a flat threshold, not a coverage-weighted
# average - go weighted only if partial days turn out to be common.
MIN_HOURS_PER_DAY = 18

# Fallback start date used only when the observations table is empty
# (i.e. the very first run / initial backfill).
INITIAL_START = "2023-08-01"


def plausible_pm25(value):
    """True when `value` could be a daily mean PM2.5 for this city, in ug/m3.

    The symmetric guard to elec/ingest.py's plausible_mw(). Open-Meteo's CAMS
    field is a unit assumption, not a guarantee: a switch to mg/m3 (/1000), a
    sentinel like -999 served in place of a null, or the field coming to mean
    something else would all parse as a float and enter the record as an
    ordinary-looking number. An observation is worse than a bad forecast here,
    because it becomes the *actual* every model is scored against - a permanent
    error in the published record that afterwards looks like a forecasting miss.

    Bounds live in vericast/__init__.py, shared with the publish gate in
    diagnose.py. Out-of-range means NULL, not a dropped row: that is exactly what
    a thin-hours day already does below, and the rest of the pipeline handles a
    NULL observation (features.py NULLs the lags, train.py filters, and
    get_earliest_hole() re-fetches the date on the next run). Contrast with elec,
    where peak_demand_mw is NOT NULL so the row has to be skipped instead.
    """
    return PM25_MIN <= value <= PM25_MAX


def get_last_observed_date():
    """Return the most recent as_of date already stored for this city, or None."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(as_of) FROM observations WHERE city = %s",
                (CITY,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def get_earliest_hole(cur, since):
    """Earliest date in [since, yesterday] this city has no usable pm2_5 for, or None.

    Two kinds of hole, one query: a date with no row at all (the upstream skipped
    the day, or an ingest run never covered it) and a date whose row carries a NULL
    pm2_5 (a thin-hours day under MIN_HOURS_PER_DAY). Both are re-fetchable, because
    the API may have filled in since - and an upsert makes a re-fetch of a good day
    free.

    generate_series is the reason this is one query: LEFT JOIN against the dates
    that *should* exist finds an absent row, which no scan of the stored rows can.
    """
    cur.execute(
        """
        SELECT MIN(d.day)::date FROM generate_series(%s::date, %s::date, '1 day') d(day)
        LEFT JOIN observations o ON o.as_of = d.day AND o.city = %s
        WHERE o.as_of IS NULL OR o.pm2_5 IS NULL
        """,
        (since, local_time.yesterday(), CITY),
    )
    return cur.fetchone()[0]


def resolve_date_range():
    """
    Decide what date range to fetch.

    - First run (no data yet): backfill from INITIAL_START.
    - Subsequent runs: the earlier of "the day after the latest stored
      observation" and "the earliest hole in the last RESCAN_DAYS days".
    - End date is always "yesterday", since the weather/air-quality
      archive APIs are historical and may not have a complete record
      for the current day yet.

    The re-scan is what makes a hole temporary. A monotonic resume off MAX(as_of)
    alone left every skipped or NULL day permanently behind the resume point;
    RESCAN_DAYS bounds the re-read so it never re-fetches 700 days. Filling one is
    still best-effort: if the upstream never serves that date, the next run simply
    starts from it again, and every write is an upsert so a re-fetched good day is
    a no-op.
    """
    last_date = get_last_observed_date()

    if last_date is None:
        return datetime.strptime(INITIAL_START, "%Y-%m-%d").date(), local_time.yesterday()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            hole = get_earliest_hole(
                cur, local_time.yesterday() - timedelta(days=RESCAN_DAYS))

    start = resume_start(last_date, hole)
    if hole and start == hole:
        print(f"Re-scanning from {hole}: earliest missing or NULL day in the last "
              f"{RESCAN_DAYS} days (a hole the monotonic resume would skip forever).")

    return start, local_time.yesterday()


def fetch_and_aggregate_data(start_date, end_date):
    """Fetch hourly AQ and daily weather data, aggregate to daily level.

    pm2_5/pm10 here are CAMS *reanalysis* (a model), not ground-station
    readings — the scoring target is a model estimate, not measured air. Chosen for
    gap-free coverage over 2023->present, which is what makes the publish-then-verify
    record clean. Upgrade path: swap the air-quality call below for OpenAQ (CPCB
    stations, free key) and handle its gaps; this is the only function that touches
    the AQ source.

    A "daily mean" here is a **UTC** day: `"timezone": "UTC"` below, and the
    bucketing loop keys on the UTC date of each hourly timestamp. The date range
    it is asked for comes from local_time (Asia/Kolkata), so an as_of of
    2026-08-27 labels 2026-08-27 00:00-23:00 UTC, which in IST is 05:30 that day
    to 04:30 the next. Both models see the same definition on both sides of the
    train/score boundary, so the record is internally consistent - that is the
    property that matters here, not which 24 hours the label names.

    Deliberately NOT switched to Asia/Kolkata: the whole series was ingested this
    way, and re-ingesting under a different timezone would silently redefine every
    historical actual - moving the numbers this repo has already published and
    scored against. That is retro-fitting, the one thing a publish-then-verify
    record cannot do. Anyone who wants IST-day means starts a new city key rather
    than rewriting this one.
    """
    START, END = start_date.isoformat(), end_date.isoformat()

    print("Fetching air quality data...")
    aq_response = httpx.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    aq_response.raise_for_status()
    aq_data = aq_response.json()

    print("Fetching weather data...")
    wx_response = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    wx_response.raise_for_status()
    wx_data = wx_response.json()

    # Extract hourly data
    times = aq_data["hourly"]["time"]
    pm2_5_values = aq_data["hourly"]["pm2_5"]
    pm10_values = aq_data["hourly"]["pm10"]

    # Extract daily weather data, keyed on the date string the loop below looks up.
    # A dict rather than wx_times.index(date_str): that re-scanned the whole array
    # once per day, which is invisible for a 1-day run and O(n^2) on a 700-day
    # backfill - the one run where it matters.
    weather = dict(zip(wx_data["daily"]["time"], zip(
        wx_data["daily"]["temperature_2m_mean"],
        wx_data["daily"]["wind_speed_10m_max"],
        wx_data["daily"]["precipitation_sum"],
    )))

    print(f"AQI hours: {len(pm2_5_values)}, missing: {sum(v is None for v in pm2_5_values)}")
    print(f"Weather days: {len(weather)}")

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

        # Weather data might not exist for this date, which stays a NULL triple
        # rather than a skip: the pm2_5 mean is the target and is usable without it.
        temp, wind, precip = weather.get(date_str, (None, None, None))

        # A day with too few hours becomes NULL rather than a mean over a
        # window that isn't a day. The rest of the pipeline already handles
        # NULL observations (features.py NULLs the lags, train.py filters).
        pm2_5_avg = (data['pm2_5_sum'] / data['pm2_5_count']
                     if data['pm2_5_count'] >= MIN_HOURS_PER_DAY else None)
        pm10_avg = (data['pm10_sum'] / data['pm10_count']
                    if data['pm10_count'] >= MIN_HOURS_PER_DAY else None)
        if pm2_5_avg is None or pm10_avg is None:
            print(f"  [skip] {date_str}: only {data['pm2_5_count']}h pm2_5 / "
                  f"{data['pm10_count']}h pm10 (need {MIN_HOURS_PER_DAY})")

        # Implausible is the third form of the same case, and the only one that
        # would otherwise be accepted silently: enough hours, arithmetic fine, wrong
        # number. Checked on pm2_5 only - it is the scored target, and any upstream
        # unit change or sentinel value hits both fields together, so guarding the
        # one that enters the record catches the class. pm10 is a feature and would
        # need its own bounds, which is more numbers to justify than it earns.
        if pm2_5_avg is not None and not plausible_pm25(pm2_5_avg):
            print(f"  [null] {date_str}: pm2_5 {pm2_5_avg:.1f} is outside "
                  f"{PM25_MIN:g}-{PM25_MAX:g} ug/m3 (unit change or sentinel "
                  f"upstream?)")
            pm2_5_avg = None

        records_to_insert.append((
            CITY,      # city
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
                cur.execute("SELECT COUNT(*) FROM observations WHERE city = %s", (CITY,))
                count = cur.fetchone()[0]
                print(f"Total {CITY} records in database: {count}")

    except Exception as e:
        print(f"Error inserting observations: {e}")
        raise

def main():
    """Main ingestion process"""
    print("Starting observations ingestion...")

    start_date, end_date = resolve_date_range()

    if start_date > end_date:
        print(
            f"Nothing new to fetch: latest data already covers through "
            f"{start_date - timedelta(days=1)}, and end date is capped at "
            f"{end_date} (yesterday). Skipping."
        )
        return

    print(f"Fetching data for {start_date} -> {end_date}")
    records = fetch_and_aggregate_data(start_date, end_date)

    if not records:
        print("No records returned from APIs; nothing to insert.")
        return

    insert_observations(records)
    print("Ingestion completed!")

if __name__ == "__main__":
    main()