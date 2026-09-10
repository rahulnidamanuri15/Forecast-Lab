import math
import os
import time
import httpx
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

from vericast import (
    ACTUAL_TOLERANCE,
    PM25_MAX,
    PM25_MIN,
    RESCAN_DAYS,
    local_time,
    require_city_of_record,
    require_database_url,
    resume_start,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

LAT, LON = 21.1463, 79.0849          # Nagpur, India
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))

# Minimum hourly samples required to compute a valid daily mean (>= 75% day coverage).
MIN_HOURS_PER_DAY = 18

# Fallback start date used only when the observations table is empty
# (i.e. the very first run / initial backfill).
INITIAL_START = "2023-08-01"


def plausible_pm25(value):
    """True when `value` could be a daily mean PM2.5 for this city, in ug/m3.

    A unit change (mg/m3), or a -999 sentinel served in place of a null, parses
    as an ordinary float and becomes the *actual* every model is scored against -
    a permanent error in the published record that later looks like a forecasting
    miss. Bounds are shared with the publish gate in diagnose.py.
    """
    return PM25_MIN <= value <= PM25_MAX


def get_last_observed_date(cur):
    """Return the most recent as_of date already stored for this city, or None."""
    cur.execute(
        "SELECT MAX(as_of) FROM observations WHERE city = %s",
        (CITY,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_earliest_hole(cur, since, end=None):
    """Earliest date in [since, end] this city has no usable pm2_5 for, or None.

    Covers both a missing row and a row with a NULL pm2_5 (a thin-hours day).
    generate_series is why this is one query: a LEFT JOIN against the dates that
    *should* exist finds an absent row, which no scan of stored rows can.

    `end` defaults to yesterday for backwards compatibility, but callers should
    pass the single yesterday value they already computed: calling
    local_time.yesterday() twice across midnight gives an inconsistent range.
    """
    end = end if end is not None else local_time.yesterday()
    cur.execute(
        """
        SELECT MIN(d.day)::date FROM generate_series(%s::date, %s::date, '1 day') d(day)
        LEFT JOIN observations o ON o.as_of = d.day AND o.city = %s
        WHERE o.as_of IS NULL OR o.pm2_5 IS NULL
        """,
        (since, end, CITY),
    )
    return cur.fetchone()[0]


def resolve_date_range():
    """
    Decide what date range to fetch: the earlier of "the day after the latest
    stored observation" and "the earliest hole in the last RESCAN_DAYS days".
    First run (no data yet) backfills from INITIAL_START. The end is always
    yesterday, since the archive APIs have no complete record for today.

    The re-scan is what makes a hole temporary: a monotonic resume off MAX(as_of)
    alone left every skipped or NULL day permanently behind the resume point.
    Still best-effort - if the upstream never serves that date, the next run
    starts from it again, and every write is an upsert.

    One connection for both queries: they are sequential and read the same table.
    """
    require_database_url(DATABASE_URL)
    # Single clock read: calling yesterday() twice across an IST midnight rollover
    # previously gave an inconsistent [since, end] window.
    yesterday = local_time.yesterday()
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            last_date = get_last_observed_date(cur)

            if last_date is None:
                return (datetime.strptime(INITIAL_START, "%Y-%m-%d").date(),
                        yesterday)

            hole = get_earliest_hole(
                cur, yesterday - timedelta(days=RESCAN_DAYS), yesterday)

    start = resume_start(last_date, hole)
    if hole and start == hole:
        print(f"Re-scanning from {hole}: earliest missing or NULL day in the last "
              f"{RESCAN_DAYS} days (a hole the monotonic resume would skip forever).")
        if (yesterday - hole).days >= RESCAN_DAYS:
            print(f"  [warn] hole {hole} sits at the edge of the {RESCAN_DAYS}-day "
                  f"re-scan window; if the upstream never serves it, every run "
                  f"re-queues from here. Investigate after repeated occurrences.")

    return start, yesterday


def fetch_and_aggregate_data(start_date, end_date):
    """Fetch hourly AQ and daily weather data, aggregate to daily level.

    pm2_5/pm10 here are CAMS *reanalysis* (a model), not ground-station
    readings - the scoring target is a model estimate, not measured air. Chosen for
    gap-free coverage over 2023->present. Upgrade path: swap the air-quality call
    below for OpenAQ (CPCB stations, free key) and handle its gaps; this is the
    only function that touches the AQ source.

    A "daily mean" here is a **UTC** day: `"timezone": "UTC"` below, and the
    bucketing loop keys on the UTC date of each hourly timestamp. Both models see
    the same definition on both sides of the train/score boundary, so the record is
    internally consistent - that is the property that matters, not which 24 hours
    the label names.

    Deliberately NOT switched to Asia/Kolkata: re-ingesting under a different
    timezone would silently redefine every historical actual, moving numbers this
    repo has already published and scored against. That is retro-fitting, the one
    thing a publish-then-verify record cannot do. Anyone who wants IST-day means
    starts a new city key rather than rewriting this one.
    """
    START, END = start_date.isoformat(), end_date.isoformat()

    def _get_with_retry(url, params, timeout, attempts=3):
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                resp = httpx.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 - retry then raise
                last_exc = exc
                if attempt < attempts:
                    backoff = 2 ** (attempt - 1)
                    print(f"  [retry] GET {url} attempt {attempt}/{attempts} failed: {exc}; sleeping {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"  [retry] GET {url} attempt {attempt}/{attempts} failed: {exc}")
        raise RuntimeError(f"GET {url} failed after {attempts} attempts: {last_exc}")

    print("Fetching air quality data...")
    aq_response = _get_with_retry("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    try:
        aq_data = aq_response.json()
        times = aq_data["hourly"]["time"]
        pm2_5_values = aq_data["hourly"]["pm2_5"]
        pm10_values = aq_data["hourly"]["pm10"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Open-Meteo AQ payload missing hourly fields: {exc}")
    if not (len(times) == len(pm2_5_values) == len(pm10_values)):
        raise RuntimeError(
            f"Open-Meteo AQ length mismatch: {len(times)} times vs "
            f"{len(pm2_5_values)} pm2_5 vs {len(pm10_values)} pm10")

    print("Fetching weather data...")
    wx_response = _get_with_retry("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
        "start_date": START, "end_date": END, "timezone": "UTC",
    }, timeout=60)
    try:
        wx_data = wx_response.json()
    except ValueError as exc:
        raise RuntimeError(f"Open-Meteo archive payload is not JSON: {exc}")
    try:
        wx_daily = wx_data["daily"]
        wx_times = wx_daily["time"]
        wx_temp = wx_daily["temperature_2m_mean"]
        wx_wind = wx_daily["wind_speed_10m_max"]
        wx_precip = wx_daily["precipitation_sum"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Open-Meteo archive payload missing daily fields: {exc}")
    if not (len(wx_times) == len(wx_temp) == len(wx_wind) == len(wx_precip)):
        raise RuntimeError(
            f"Open-Meteo archive length mismatch: {len(wx_times)} times vs "
            f"{len(wx_temp)} temp vs {len(wx_wind)} wind vs {len(wx_precip)} precip")

    # Keyed on the date string the loop below looks up. A dict rather than
    # wx_times.index(date_str): that is O(n^2) on a 700-day backfill.
    weather = dict(zip(wx_times, zip(wx_temp, wx_wind, wx_precip)))

    print(f"AQI hours: {len(pm2_5_values)}, missing: {sum(v is None for v in pm2_5_values)}")
    print(f"Weather days: {len(weather)}")

    # Aggregate hourly to daily
    daily_data = {}

    for i, timestamp in enumerate(times):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            print(f"  [skip] unparseable hourly timestamp {timestamp!r}; skipping hour")
            continue
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
        # [null], not [skip]: the row is still appended below, with NULLs. The
        # date has to exist for get_earliest_hole()'s LEFT JOIN to distinguish
        # "thin day, already fetched" from "never fetched".
        if pm2_5_avg is None or pm10_avg is None:
            print(f"  [null] {date_str}: only {data['pm2_5_count']}h pm2_5 / "
                  f"{data['pm10_count']}h pm10 (need {MIN_HOURS_PER_DAY})")

        # Implausible is the third form of the same case, and the only one that
        # would otherwise be accepted silently: enough hours, arithmetic fine, wrong
        # number. Checked on pm2_5 only - it is the scored target, and any upstream
        # unit change or sentinel hits both fields together.
        if pm2_5_avg is not None and not plausible_pm25(pm2_5_avg):
            print(f"  [null] {date_str}: pm2_5 {pm2_5_avg:.1f} is outside "
                  f"{PM25_MIN:g}-{PM25_MAX:g} ug/m3 (unit change or sentinel "
                  f"upstream?)")
            pm2_5_avg = None

        # Aux plausibility: a -999 sentinel or unit change in a non-scored
        # column still poisons lags/rolls and training. Null it, keep the row.
        if pm10_avg is not None and not (1.0 <= pm10_avg <= 1000.0):
            print(f"  [null] {date_str}: pm10 {pm10_avg:.1f} outside 1-1000 ug/m3; nulling")
            pm10_avg = None
        if temp is not None and not (-10.0 <= temp <= 55.0):
            print(f"  [null] {date_str}: temp {temp:.1f}C implausible; nulling")
            temp = None
        if wind is not None and not (0.0 <= wind <= 60.0):
            print(f"  [null] {date_str}: wind {wind:.1f} implausible; nulling")
            wind = None
        if precip is not None and not (0.0 <= precip <= 500.0):
            print(f"  [null] {date_str}: precip {precip:.1f} implausible; nulling")
            precip = None

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
    """Insert aggregated observations into the database."""
    if not records:
        return
    require_database_url(DATABASE_URL)

    insert_sql = """
    INSERT INTO observations (city, as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (city, as_of) DO UPDATE SET
        pm2_5 = EXCLUDED.pm2_5,
        pm10 = EXCLUDED.pm10,
        temperature_2m_mean = EXCLUDED.temperature_2m_mean,
        wind_speed_10m_max = EXCLUDED.wind_speed_10m_max,
        precipitation_sum = EXCLUDED.precipitation_sum,
        created_at = CURRENT_TIMESTAMP
    WHERE observations.pm2_5 IS DISTINCT FROM EXCLUDED.pm2_5
       OR observations.pm10 IS DISTINCT FROM EXCLUDED.pm10
       OR observations.temperature_2m_mean IS DISTINCT FROM EXCLUDED.temperature_2m_mean
       OR observations.wind_speed_10m_max IS DISTINCT FROM EXCLUDED.wind_speed_10m_max
       OR observations.precipitation_sum IS DISTINCT FROM EXCLUDED.precipitation_sum;
    """

    # Query existing observation values in range to log upstream revisions.
    before_sql = """
    SELECT as_of, pm2_5 FROM observations
    WHERE city = %s AND as_of BETWEEN %s AND %s
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                dates = [as_of for _, as_of, *_ in records]
                cur.execute(before_sql, (CITY, min(dates), max(dates)))
                before = {as_of.isoformat(): pm for as_of, pm in cur.fetchall()}

                cur.executemany(insert_sql, records)
                conn.commit()
                # rowcount is new-or-changed rows only, thanks to the WHERE above: an
                # unchanged re-scan day is skipped rather than rewritten with a fresh
                # created_at.
                print(f"Wrote {cur.rowcount} new or changed record(s) of "
                      f"{len(records)} submitted")

                report_revisions(before, records)

                # Verify insertion
                cur.execute("SELECT COUNT(*) FROM observations WHERE city = %s", (CITY,))
                count = cur.fetchone()[0]
                print(f"Total {CITY} records in database: {count}")

    except Exception as e:
        print(f"Error inserting observations: {e}")
        raise


def report_revisions(before, records):
    """Log days where CAMS ground-truth values were revised from prior stored observations."""
    def _moved(old, new):
        if new is None:
            return True
        try:
            diff = abs(float(new) - float(old))
        except (TypeError, ValueError):
            return True
        # NaN never compares: abs(nan - x) > eps is False, so a NaN revision
        # would previously pass silently. Treat non-finite as a revision.
        if not math.isfinite(diff):
            return True
        return diff > ACTUAL_TOLERANCE

    revised = [(as_of, before[as_of], new)
               for _, as_of, new, *_ in records
               if before.get(as_of) is not None
               and _moved(before[as_of], new)]
    if not revised:
        return

    print(f"  {len(revised)} day(s) had their pm2_5 REVISED upstream:")
    for as_of, old, new in sorted(revised):
        shown = f"{new:.2f}" if new is not None else "NULL"
        print(f"    [revised] {as_of}: {old:.2f} -> {shown} ug/m3")
    print("  Any of these already scored against are re-opened and re-scored by "
          "vericast.pm25.score, so the published error follows the observation.")



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