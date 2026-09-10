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
import time
import csv
import io
import httpx
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

from vericast import (
    ACTUAL_TOLERANCE,
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    RESCAN_DAYS,
    local_time,
    require_database_url,
    resume_start,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = os.getenv("STATE", "Maharashtra")

DEMAND_CSV_URL = (
    "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/"
    "Dataset/study3_states.csv"
)

REQUIRED_CSV_COLUMNS = {"state", "date", "max_demand_met_mw", "energy_met_mu"}

# Tracking `main`, not a pinned commit SHA. A pin would freeze the file, and this
# mirror runs 2-4 days behind and backfills late, so a pinned SHA can only ever
# serve dates that already existed when it was taken - which is exactly the
# hole-filling the RESCAN_DAYS re-scan depends on.
#
# What replaces it: REQUIRED_CSV_COLUMNS catches a rename or a dropped column before
# any row is accepted, and plausible_mw() catches a changed *value* - a unit switch
# to kW or GW - which no column-name check can see. Neither catches a
# plausible-but-wrong number, and nothing can.


def plausible_mw(value):
    """True when `value` could be a whole state's daily peak demand, in MW.

    The mirror is a third-party CSV at a mutable branch, so the unit is an
    assumption. A switch to kW (x1000) or GW (/1000) would otherwise flow into the
    scored record and be indistinguishable from a real forecasting error afterwards.
    Out-of-range days are skipped, exactly like a blank one, rather than raising and
    abandoning the rows already accepted.
    """
    return ELEC_MIN_MW <= value <= ELEC_MAX_MW

# Regional temperature representative coordinates across Maharashtra
CITIES = [
    (19.0760, 72.8777),  # Mumbai
    (18.5204, 73.8567),  # Pune
    (21.1458, 79.0882),  # Nagpur
]

# The mirror covers 2018-12-31 onwards, but 2023-01-01 is where the Maharashtra
# series is gap-free and null-free (verified: 1328 days, one gap 2025-05-21->24).
INITIAL_START = "2023-01-01"


def get_last_observed_date(cur):
    """Return the most recent as_of already stored for this state, or None."""
    cur.execute(
        "SELECT MAX(as_of) FROM electricity_observations WHERE state = %s",
        (STATE,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_earliest_hole(cur, since, end=None):
    """Earliest date in [since, end] with no observation for this state, or None.

    Only absent rows, unlike the PM2.5 twin: peak_demand_mw is NOT NULL here, so a
    day the mirror served blank was skipped at insert time rather than stored as
    NULL. Those are the dates worth re-reading, since the mirror backfills late.
    A temp-only NULL row is intentionally NOT a hole: it still carries a scoreable
    peak, and re-reading it would fail identically until the upstream cell changes.

    `end` defaults to yesterday for backwards compatibility; callers pass their
    single yesterday value to avoid a midnight-rollover skew.
    """
    end = end if end is not None else local_time.yesterday()
    cur.execute(
        """
        SELECT MIN(d.day)::date FROM generate_series(%s::date, %s::date, '1 day') d(day)
        LEFT JOIN electricity_observations o
               ON o.as_of = d.day AND o.state = %s
        WHERE o.as_of IS NULL
        """,
        (since, end, STATE),
    )
    return cur.fetchone()[0]


def resolve_date_range():
    """First run backfills from INITIAL_START; later runs take the earlier of
    "the day after the latest stored observation" and "the earliest missing day in
    the last RESCAN_DAYS days". End is capped at yesterday.

    The re-scan is what makes a skipped day temporary: a monotonic resume off
    MAX(as_of) alone left 2025-05-21 -> 05-24 permanently behind the resume point.
    RESCAN_DAYS bounds the re-read so it never re-fetches 1,300 days, and every
    write is an upsert, so re-fetching a stored day is a no-op.

    One connection for both queries, as in the PM2.5 twin.
    """
    require_database_url(DATABASE_URL)
    # Single clock read: see the PM2.5 twin for why yesterday() is read once.
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
        print(f"Re-scanning from {hole}: earliest missing day in the last "
              f"{RESCAN_DAYS} days (a hole the monotonic resume would skip forever).")
        if (yesterday - hole).days >= RESCAN_DAYS:
            print(f"  [warn] hole {hole} sits at the edge of the {RESCAN_DAYS}-day "
                  f"re-scan window; if the mirror never backfills it, every run "
                  f"re-queues from here. Investigate after repeated occurrences.")

    return start, yesterday


def fetch_demand(start_date, end_date):
    """Fetch the mirror and return {date_str: (peak_demand_mw, energy_met_mu)}
    for Maharashtra rows inside the range. The only function that touches the
    demand source."""
    print(f"Fetching demand mirror ({DEMAND_CSV_URL.rsplit('/', 1)[-1]})...")
    # Fetch full CSV mirror (~6MB) with exponential backoff retries.
    last_exc = None
    response = None
    for attempt in range(1, 4):
        try:
            response = httpx.get(DEMAND_CSV_URL, timeout=180, follow_redirects=True)
            response.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001 - retry then raise
            last_exc = exc
            if attempt < 3:
                backoff = 2 ** (attempt - 1)
                print(f"  [retry] demand mirror attempt {attempt}/3 failed: {exc}; sleeping {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"  [retry] demand mirror attempt {attempt}/3 failed: {exc}")
    if response is None:
        raise RuntimeError(f"Demand mirror fetch failed after 3 attempts: {last_exc}")

    start_str, end_str = start_date.isoformat(), end_date.isoformat()
    demand = {}
    skipped = 0

    reader = csv.DictReader(io.StringIO(response.text))
    # Fail naming the URL, not with a KeyError mid-loop after some rows have
    # already been accepted.
    missing = REQUIRED_CSV_COLUMNS - set(reader.fieldnames or ())
    if missing:
        raise RuntimeError(
            f"Demand mirror is missing column(s) {sorted(missing)}; "
            f"got {reader.fieldnames}. Source: {DEMAND_CSV_URL}"
        )

    for row in reader:
        if row["state"].strip() != STATE:
            continue
        date_str = row["date"].strip()
        if not (start_str <= date_str <= end_str):
            continue

        # peak_demand_mw is NOT NULL in the schema, so a blank one is a skip, not a
        # NULL row - a demand row without demand has nothing to score against.
        # Unparseable is the same case: a stray 'N/A' from a third-party mirror must
        # skip the day, not raise halfway through and abandon the accepted rows.
        raw_mw = row["max_demand_met_mw"].strip()
        if not raw_mw:
            skipped += 1
            continue

        try:
            peak_mw = float(raw_mw)
        except ValueError:
            skipped += 1
            continue

        # Implausible is the third form of the same case, and the only one that
        # would otherwise be accepted silently: a unit change upstream parses fine.
        if not plausible_mw(peak_mw):
            print(f"  [skip] {date_str}: {peak_mw} MW is outside "
                  f"{ELEC_MIN_MW:,.0f}-{ELEC_MAX_MW:,.0f} MW (unit change upstream?)")
            skipped += 1
            continue

        # energy_met_mu is nullable and nothing scores against it - /electricity/history
        # only echoes it - so an unparseable one degrades to NULL rather than throwing
        # away the NOT NULL peak_demand_mw that the whole target is scored on. Skipping
        # here would also be permanent in practice: the RESCAN_DAYS re-read fails the
        # same way every run until the upstream cell changes.
        raw_mu = row["energy_met_mu"].strip()
        try:
            energy_mu = float(raw_mu) if raw_mu else None
        except ValueError:
            print(f"  [null] {date_str}: energy_met_mu {raw_mu!r} is unparseable; "
                  f"storing NULL and keeping the {peak_mw:,.0f} MW peak.")
            energy_mu = None

        demand[date_str] = (peak_mw, energy_mu)

    print(f"{STATE} demand days in range: {len(demand)}"
          + (f" ({skipped} skipped for missing, unparseable or implausible MW)"
             if skipped else ""))
    return demand


def fetch_temperature(start_date, end_date):
    """Fetch daily mean/max temperature for the three cities in one call and
    return {date_str: (mean, max)} of their averages.

    Comma-separated coordinates make Open-Meteo return a JSON *array* of
    per-location objects, each with its own `daily` block."""
    print("Fetching regional temperature...")
    last_exc = None
    response = None
    for attempt in range(1, 4):
        try:
            response = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": ",".join(str(lat) for lat, _ in CITIES),
                "longitude": ",".join(str(lon) for _, lon in CITIES),
                "daily": "temperature_2m_mean,temperature_2m_max",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timezone": "UTC",
            }, timeout=60)
            response.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 3:
                backoff = 2 ** (attempt - 1)
                print(f"  [retry] temperature attempt {attempt}/3 failed: {exc}; sleeping {backoff}s...")
                time.sleep(backoff)
            else:
                print(f"  [retry] temperature attempt {attempt}/3 failed: {exc}")
    if response is None:
        raise RuntimeError(f"Temperature fetch failed after 3 attempts: {last_exc}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Temperature payload is not JSON: {exc}")

    # A single location returns a bare object; multiple return a list.
    locations = payload if isinstance(payload, list) else [payload]

    sums = {}
    for location in locations:
        try:
            daily = location["daily"]
            loc_times = daily["time"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Temperature payload missing daily block: {exc}")
        for i, date_str in enumerate(loc_times):
            try:
                mean_v, max_v = daily["temperature_2m_mean"][i], daily["temperature_2m_max"][i]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Temperature payload malformed for {date_str}: {exc}")
            acc = sums.setdefault(date_str, [0.0, 0, 0.0, 0])
            if mean_v is not None:
                acc[0] += mean_v
                acc[1] += 1
            if max_v is not None:
                acc[2] += max_v
                acc[3] += 1

    def _plausible_temp(v, lo=-10.0, hi=55.0):
        return v is not None and lo <= v <= hi

    temps = {}
    for date_str, acc in sums.items():
        mean = acc[0] / acc[1] if acc[1] else None
        mx = acc[2] / acc[3] if acc[3] else None
        if acc[1] and acc[1] < len(locations):
            print(f"  [warn] {date_str}: temp mean from {acc[1]}/{len(locations)} cities")
        if mean is not None and not _plausible_temp(mean):
            print(f"  [null] {date_str}: temp mean {mean:.1f}C implausible; nulling")
            mean = None
        if mx is not None and not _plausible_temp(mx, hi=60.0):
            print(f"  [null] {date_str}: temp max {mx:.1f}C implausible; nulling")
            mx = None
        temps[date_str] = (mean, mx)
    print(f"Temperature days: {len(temps)} (mean of {len(locations)} cities)")
    return temps


def insert_observations(records):
    """Write the day's rows, skipping unchanged rows and reporting revisions."""
    if not records:
        return
    require_database_url(DATABASE_URL)

    insert_sql = """
    INSERT INTO electricity_observations
        (state, as_of, peak_demand_mw, energy_met_mu, temperature_2m_mean, temperature_2m_max)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (state, as_of) DO UPDATE SET
        peak_demand_mw = EXCLUDED.peak_demand_mw,
        energy_met_mu = EXCLUDED.energy_met_mu,
        temperature_2m_mean = EXCLUDED.temperature_2m_mean,
        temperature_2m_max = EXCLUDED.temperature_2m_max,
        created_at = CURRENT_TIMESTAMP
    WHERE electricity_observations.peak_demand_mw IS DISTINCT FROM EXCLUDED.peak_demand_mw
       OR electricity_observations.energy_met_mu IS DISTINCT FROM EXCLUDED.energy_met_mu
       OR electricity_observations.temperature_2m_mean IS DISTINCT FROM EXCLUDED.temperature_2m_mean
       OR electricity_observations.temperature_2m_max IS DISTINCT FROM EXCLUDED.temperature_2m_max;
    """

    # Query existing observation values in range to log upstream revisions.
    before_sql = """
    SELECT as_of, peak_demand_mw FROM electricity_observations
    WHERE state = %s AND as_of BETWEEN %s AND %s
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                dates = [as_of for _, as_of, *_ in records]
                cur.execute(before_sql, (STATE, min(dates), max(dates)))
                before = {as_of.isoformat(): mw for as_of, mw in cur.fetchall()}

                cur.executemany(insert_sql, records)
                conn.commit()
                print(f"Wrote {cur.rowcount} new or changed record(s) of "
                      f"{len(records)} submitted")

                report_revisions(before, records)

                cur.execute(
                    "SELECT COUNT(*), MIN(as_of), MAX(as_of) FROM electricity_observations "
                    "WHERE state = %s", (STATE,))
                count, first, last = cur.fetchone()
                print(f"Total {STATE} records in database: {count} ({first} -> {last})")
    except Exception as e:
        print(f"Error inserting electricity observations: {e}")
        raise


def report_revisions(before, records):
    """Log days where peak demand ground truth was revised from prior stored observations."""
    revised = [(as_of, before[as_of], new)
               for _, as_of, new, *_ in records
               if before.get(as_of) is not None and abs(new - before[as_of]) > ACTUAL_TOLERANCE]
    if not revised:
        return

    print(f"  {len(revised)} day(s) had their peak_demand_mw REVISED upstream:")
    for as_of, old, new in sorted(revised):
        print(f"    [revised] {as_of}: {old:,.0f} -> {new:,.0f} MW")
    print("  Any of these already scored against are re-opened and re-scored by "
          "vericast.elec.score, so the published error follows the observation.")


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
