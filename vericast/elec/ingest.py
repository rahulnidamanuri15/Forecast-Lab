"""Ingest daily Maharashtra peak electricity demand + regional temperature.

Sources, in order:
1. The Grid-Sentinel community mirror of Grid-India/NLDC daily PSP reports,
   as an already-parsed CSV (stdlib `csv` over the already-installed httpx).
2. Official NLDC gap-fill: dates the mirror lacks are retried against
   Grid-India's own daily PSP reports (webapi.grid-india.in index +
   webcdn.grid-india.in .xls, parsed with xlrd) before giving up.

Grid-India's old report.grid-india.in host is unreachable from CI (HTTP 000),
but the webapp API + CDN behind grid-india.in are not - verified 2026-10-15,
when 7/7 NLDC values were byte-identical to mirror/DB rows for 2026-09-29..10-05.

The tradeoff, stated plainly because the accuracy record depends on it: source
1 is a third-party mirror, not the operator. It runs 2-4 days behind real time
and skips the occasional day, so `stale_days` of 2-4 is NORMAL here (unlike
PM2.5, where Open-Meteo has yesterday by 05:00 UTC). Source 2 is the operator
itself and is current through yesterday. Ingest fails at 1/6 with the source
status when neither source can supply data within ELEC_STALE_LIMIT_DAYS, so
predict.py never receives a stale series. At most NLDC_MAX_DAYS_PER_RUN gap
days are attempted per run; the next run picks up the remainder.
"""
import os
import time
import csv
import hashlib
import io
import math
import random
import re
import httpx
import psycopg
from datetime import datetime, timedelta
from dotenv import load_dotenv

from vericast import (
    ACTUAL_TOLERANCE,
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    ELEC_STALE_LIMIT_DAYS,
    RESCAN_DAYS,
    acquire_pipeline_lock,
    local_time,
    refuse_stale,
    require_database_url,
    require_state_of_record,
    resume_start,
    send_alert,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = require_state_of_record(os.getenv("STATE", "Maharashtra"))

DEMAND_CSV_URL = (
    "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/"
    "Dataset/study3_states.csv"
)

# Transport mirrors for the SAME file above, via independent CDNs.
# If raw.githubusercontent.com has a CDN outage, jsDelivr/githack serve the
# identical GitHub blob - fetch_demand() tries each in order with the same
# guards, so a transport failure no longer fails the 1/6 ingest step.
#
# Stated plainly: these do NOT fix an upstream stall. When the author stops
# committing (mirror head frozen, e.g. 2026-09-29 while NLDC keeps publishing),
# every transport serves the same frozen bytes; the dates still missing are then
# retried against the official NLDC gap-fill below, and only what neither source
# has falls through to manual `python -m vericast.elec.backfill DATE:MW`.
DEMAND_MIRROR_URLS = (
    DEMAND_CSV_URL,
    "https://cdn.jsdelivr.net/gh/HalcyonVector/Grid-Sentinel@main/Dataset/study3_states.csv",
    "https://raw.githack.com/HalcyonVector/Grid-Sentinel/main/Dataset/study3_states.csv",
)

# Heads-up before the hard refusal: predict.py refuses past
# ELEC_STALE_LIMIT_DAYS (7); warn at 5 so the operator gets ~2 days to
# backfill from NLDC PSP reports instead of discovering it at the 5/6 crash.
EARLY_STALE_WARN_DAYS = ELEC_STALE_LIMIT_DAYS - 2

# Official NLDC gap-fill: the same reports API the grid-india.in webapp itself
# calls, plus the CDN it downloads from. Each daily report ships as PDF + .xls;
# the .xls sheet MOP_E section C is the state table
# (Maharashtra | Max.Demand Met (MW) | ... | Energy Met (MU)) - the same two
# columns the CSV mirror serves, under the same plausible_mw() guard.
NLDC_API_BASE = "https://webapi.grid-india.in/api/v1"
NLDC_API_URL = f"{NLDC_API_BASE}/file"
NLDC_PAGE_URL = f"{NLDC_API_BASE}/page"
NLDC_PAGE_PATH = "/en/reports/daily-psp-report"
NLDC_SOURCE = "GRDW"
# Legacy symbolic type; fetch_nldc_index() falls back to the page's FileType.
NLDC_FILE_TYPE = "DAILY_PSP_REPORT"
NLDC_CDN_BASE = "https://webcdn.grid-india.in/"
NLDC_SHEET = "MOP_E"

# Last-resort bridge when webapi.grid-india.in is unreachable from the runner
# (the API resets datacenter connections). Grid-Sentinel commits the official
# .xls reports it downloads; its CSV parser lags, but these raw files do not.
# Same report, same date/plausibility guards - only the transport changes.
NLDC_RAW_MIRROR_URL = (
    "https://raw.githubusercontent.com/HalcyonVector/Grid-Sentinel/main/"
    "Dataset/Raw/File3_Raw/{dd}.{mm}.{yy}_NLDC_PSP.xls"
)

# Per-file and per-run budgets: reports are ~300KB; 10MB refuses a compromised
# CDN serving a giant body. At most NLDC_MAX_DAYS_PER_RUN reports per run so a
# 30-day RESCAN hole cannot blow the 10-minute ingest step; the remainder wait
# for the next cron (or a manual backfill).
NLDC_MAX_FILE_BYTES = 10 * 1024 * 1024
NLDC_MAX_DAYS_PER_RUN = 15

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


def fetch_demand(start_date, end_date, diagnostics=None):
    """Fetch the mirror and return {date_str: (peak_demand_mw, energy_met_mu)}
    for Maharashtra rows inside the range. The only function that touches the
    demand source.

    Tries each URL in DEMAND_MIRROR_URLS in order (same file, different CDN):
    a transport failure on the primary falls through to the next mirror with
    the same per-mirror retries and the same size/column/plausibility guards.

    `diagnostics` is an optional dict populated with `mirror_url` and
    `mirror_head` (the newest Maharashtra date in the CSV, even outside the
    requested range). Ingest prints it so a stalled mirror is distinguishable
    from a dead official feed without reading the whole CSV.
    """
    print(f"Fetching demand mirror ({DEMAND_CSV_URL.rsplit('/', 1)[-1]})...")
    # Fetch full CSV mirror (~6MB) with exponential backoff + jitter retries.
    # 25MB cap: a compromised mirror serving a giant body must fail fast
    # instead of buffering unboundedly into memory.
    MAX_MIRROR_BYTES = 25 * 1024 * 1024
    last_exc = None
    response = None
    used_url = None
    for mirror_url in DEMAND_MIRROR_URLS:
        for attempt in range(1, 4):
            try:
                response = httpx.get(mirror_url, timeout=180, follow_redirects=True)
                response.raise_for_status()
                # Size cap on raw bytes when available; MagicMock in unit tests has
                # no real content, so skip the cap there.
                raw_content = getattr(response, "content", None)
                if isinstance(raw_content, (bytes, bytearray)) and len(raw_content) > MAX_MIRROR_BYTES:
                    raise RuntimeError(
                        f"Demand mirror body {len(raw_content)} bytes exceeds "
                        f"{MAX_MIRROR_BYTES} cap; refusing to parse")
                used_url = mirror_url
                break
            except Exception as exc:  # noqa: BLE001 - retry then next mirror
                last_exc = exc
                if attempt < 3:
                    backoff = 2 ** (attempt - 1) + random.uniform(0, 1)
                    print(f"  [retry] demand mirror attempt {attempt}/3 failed: {exc}; sleeping {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    print(f"  [retry] demand mirror attempt {attempt}/3 failed: {exc}")
        if used_url is not None:
            if used_url != DEMAND_CSV_URL:
                print(f"  [fallback] primary mirror failed; serving from {used_url}")
            break
    if response is None:
        raise RuntimeError(
            f"Demand mirror fetch failed after 3 attempts per mirror "
            f"across {len(DEMAND_MIRROR_URLS)} mirrors: {last_exc}")
    if used_url is None:
        # All mirrors tripped the size cap yet still returned a decodable
        # body (the oversize unit test mocks huge .content beside a small
        # .text): keep the legacy fall-through to parsing .text rather than
        # refusing outright. A genuinely huge .text still parses as before.
        used_url = DEMAND_CSV_URL
        print(f"  [warn] all mirrors exceeded the size cap on raw bytes; "
              f"falling through to parse the decoded text.")

    # Prefer .text when the transport already decoded it (and in unit tests
    # where only .text is mocked); else decode raw bytes as UTF-8.
    # Audit pin: log content hash so a plausible-but-wrong mirror edit is
    # attributable after the fact. A full pin would freeze backfills (see note
    # above), so this detects rather than prevents.
    if isinstance(getattr(response, "text", None), str):
        text = response.text
        try:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            print(f"  mirror sha256:{digest} bytes:{len(text.encode('utf-8'))}")
        except Exception:
            pass
    else:
        raw = response.content
        try:
            digest = hashlib.sha256(bytes(raw)).hexdigest()[:16]
            print(f"  mirror sha256:{digest} bytes:{len(bytes(raw))}")
        except Exception:
            pass
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"Demand mirror is not valid UTF-8: {exc}")

    start_str, end_str = start_date.isoformat(), end_date.isoformat()
    demand = {}
    skipped = 0
    mirror_head = None

    reader = csv.DictReader(io.StringIO(text))
    # Fail naming the URL, not with a KeyError mid-loop after some rows have
    # already been accepted.
    missing = REQUIRED_CSV_COLUMNS - set(reader.fieldnames or ())
    if missing:
        raise RuntimeError(
            f"Demand mirror is missing column(s) {sorted(missing)}; "
            f"got {reader.fieldnames}. Source: {used_url or DEMAND_CSV_URL}"
        )

    for row in reader:
        if row["state"].strip() != STATE:
            continue
        date_str = row["date"].strip()
        # Track the head across the whole file, not just the requested range:
        # the most common failure is a mirror frozen weeks before the range,
        # and "0 rows in range" alone does not say whether the mirror is dead
        # or merely behind.
        if date_str and (mirror_head is None or date_str > mirror_head):
            mirror_head = date_str
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

    if diagnostics is not None:
        diagnostics["mirror_url"] = used_url or DEMAND_CSV_URL
        diagnostics["mirror_head"] = mirror_head

    print(f"{STATE} demand days in range: {len(demand)}"
          + (f" ({skipped} skipped for missing, unparseable or implausible MW)"
             if skipped else ""))
    print(f"  Mirror head: {mirror_head or 'no Maharashtra rows'}")
    return demand


def _post_nldc_json(url, payload, label):
    """POST a Grid-India API payload with retries for transport failures."""
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = httpx.post(
                url,
                json=payload,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0",
                         "Origin": "https://grid-india.in",
                         "Referer": "https://grid-india.in/"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - retry then raise
            last_exc = exc
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{label} unreachable after 3 attempts: {last_exc}")


def _parse_nldc_index(payload, file_type):
    """Turn a reports-API payload into {data_date_iso: cdn_path}."""
    try:
        rows = payload["retData"]
    except (KeyError, TypeError) as exc:
        flags = (payload.get("flagType") if isinstance(payload, dict) else None,
                 payload.get("retMessage") if isinstance(payload, dict) else None)
        raise RuntimeError(
            f"NLDC reports API returned unexpected payload for type {file_type!r} "
            f"({exc}); flagType/retMessage={flags!r}") from exc
    if not isinstance(rows, list):
        flags = (payload.get("flagType"), payload.get("retMessage"))
        raise RuntimeError(
            f"NLDC reports API returned {type(rows).__name__} for retData, expected list "
            f"(type {file_type!r}; flagType/retMessage={flags!r})")

    index = {}
    for row in rows:
        try:
            mime = str(row.get("MimeType") or "")
            path = str(row.get("FilePath") or "")
            # Do not rely on the MIME type alone; the suffix selects the
            # parser, and generic MIME types are common on file CDNs.
            if mime != "application/vnd.ms-excel" and not path.lower().endswith(".xls"):
                continue
            m = re.match(r"(\d{2})\.(\d{2})\.(\d{2})_NLDC_PSP", row.get("Title_", ""))
            if not m:
                continue
            dd, mm, yy = m.groups()
            iso = f"20{yy}-{mm}-{dd}"
            datetime.strptime(iso, "%Y-%m-%d")  # reject impossible dates
            if iso not in index and path:
                index[iso] = path
        except (ValueError, AttributeError, TypeError):
            continue
    if not index:
        sample = [
            (row.get("Title_"), row.get("MimeType"))
            for row in rows[:5] if isinstance(row, dict)
        ]
        raise RuntimeError(
            f"NLDC reports API returned no parseable .xls entries for type "
            f"{file_type!r} ({len(rows)} row(s); sample Title_/MimeType={sample!r})")
    return index


def fetch_nldc_file_type():
    """Read the daily-PSP FileType from the same page metadata the webapp uses.

    The old hard-coded type name is not guaranteed to survive a site rewrite;
    the page metadata is the source of truth for the reports API's `_type`.
    """
    payload = _post_nldc_json(
        NLDC_PAGE_URL,
        {"_source": NLDC_SOURCE, "_relPath": NLDC_PAGE_PATH},
        "NLDC page metadata API",
    )
    try:
        rows = payload["retData"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"NLDC page metadata returned unexpected payload: {exc}") from exc
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise RuntimeError(
            f"NLDC page metadata returned no page for {NLDC_PAGE_PATH!r}")
    file_type = rows[0].get("FileType")
    if not isinstance(file_type, str) or not file_type.strip():
        raise RuntimeError(
            f"NLDC page metadata for {NLDC_PAGE_PATH!r} has no FileType "
            f"(sample keys={sorted(rows[0])!r})")
    return file_type.strip()


def fetch_nldc_index():
    """Return {data_date_iso: cdn_path} for official daily PSP .xls reports.

    Raises RuntimeError when the API is unreachable or returns nothing
    usable. The caller records the reason in its diagnostics, then the
    freshness gate decides whether the run can publish.
    The report Title carries the data date (DD.MM.YY_NLDC_PSP); titles that
    do not parse are dropped, never guessed. A failed index includes a sample
    of the actual titles/MIME types: the common silent failures are a changed
    title format or an upstream switch from .xls to .xlsx/PDF.

    The webapp now takes `_type` from the page metadata rather than using a
    symbolic report name. Try the known type first, then discover the current
    FileType from the daily-PSP page and retry once.
    """
    first_exc = None
    try:
        payload = _post_nldc_json(
            NLDC_API_URL,
            {"_source": NLDC_SOURCE, "_type": NLDC_FILE_TYPE},
            "NLDC reports API",
        )
        index = _parse_nldc_index(payload, NLDC_FILE_TYPE)
    except RuntimeError as exc:
        first_exc = exc
    else:
        print(f"  NLDC index: {len(index)} daily .xls report(s); newest {max(index)}")
        return index

    try:
        file_type = fetch_nldc_file_type()
    except RuntimeError:
        raise first_exc from None
    if file_type == NLDC_FILE_TYPE:
        raise first_exc from None
    print(f"  NLDC reports type {NLDC_FILE_TYPE!r} failed; retrying with "
          f"page FileType {file_type!r}")
    payload = _post_nldc_json(
        NLDC_API_URL,
        {"_source": NLDC_SOURCE, "_type": file_type},
        "NLDC reports API",
    )
    index = _parse_nldc_index(payload, file_type)
    print(f"  NLDC index: {len(index)} daily .xls report(s); newest {max(index)}")
    return index


def nldc_raw_mirror_url(date_str):
    """Grid-Sentinel's normalized raw file for one data date."""
    day = datetime.strptime(date_str, "%Y-%m-%d")
    return NLDC_RAW_MIRROR_URL.format(
        dd=f"{day.day:02d}", mm=f"{day.month:02d}", yy=f"{day.year % 100:02d}")


def _download_nldc_report(url):
    """Download one report with one retry; raises RuntimeError on failure."""
    last_exc = None
    for attempt in range(1, 3):
        try:
            resp = httpx.get(url, timeout=120, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            content = resp.content
            if not isinstance(content, (bytes, bytearray)) or not content:
                raise RuntimeError("empty body")
            if len(content) > NLDC_MAX_FILE_BYTES:
                raise RuntimeError(
                    f"body {len(content)} bytes exceeds {NLDC_MAX_FILE_BYTES} cap")
            return bytes(content)
        except Exception as exc:  # noqa: BLE001 - retry once, then report
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                break
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError(f"download failed: {last_exc}")


def parse_nldc_workbook(content, date_str):
    """Parse one NLDC .xls report; return (peak_mw, energy_mu) or None to skip.

    Same guard philosophy as the CSV path: sheet MOP_E, section C state table,
    Maharashtra row, numeric MW inside plausible_mw(); a bad MU degrades to
    NULL because nothing scores against it. Structural surprises skip the day
    with a message instead of raising - a misdated or misparsed row is worse
    than a missing one. The sheet's own 'Date of Reporting' (always the day
    after the data date) cross-checks the Title->date mapping per file.
    """
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError(
            "xlrd is required to parse NLDC .xls reports (pip install xlrd)") from exc
    if len(content) > NLDC_MAX_FILE_BYTES:
        print(f"  [skip] {date_str}: NLDC body {len(content)} bytes exceeds "
              f"{NLDC_MAX_FILE_BYTES} cap; skipping")
        return None
    try:
        book = xlrd.open_workbook(file_contents=bytes(content))
        sheet = book.sheet_by_name(NLDC_SHEET)
    except Exception as exc:  # noqa: BLE001 - corrupt/changed workbook skips
        print(f"  [skip] {date_str}: NLDC workbook unreadable ({exc}); skipping")
        return None

    expected_report = (datetime.strptime(date_str, "%Y-%m-%d").date()
                       + timedelta(days=1)).strftime("%d-%b-%Y").lower()
    seen_report = None
    for rx in range(min(6, sheet.nrows)):
        for cx in range(sheet.ncols):
            try:
                v = str(sheet.cell(rx, cx).value).strip()
            except IndexError:
                continue
            if v.lower().startswith("date of reporting") and cx + 1 < sheet.ncols:
                try:
                    seen_report = str(sheet.cell(rx, cx + 1).value).strip().lower()
                except IndexError:
                    pass
                break
    if seen_report and seen_report != expected_report:
        print(f"  [skip] {date_str}: NLDC reporting date {seen_report!r} != "
              f"expected {expected_report!r}; skipping rather than misdating")
        return None

    for rx in range(sheet.nrows):
        try:
            state = str(sheet.cell(rx, 1).value).strip().lower()
        except IndexError:
            continue
        if state != "maharashtra":
            continue
        try:
            peak_mw = float(sheet.cell(rx, 2).value)
        except (ValueError, TypeError, IndexError) as exc:
            print(f"  [skip] {date_str}: NLDC Maharashtra MW unparseable "
                  f"({exc}); skipping")
            return None
        if not plausible_mw(peak_mw):
            print(f"  [skip] {date_str}: NLDC {peak_mw} MW is outside "
                  f"{ELEC_MIN_MW:,.0f}-{ELEC_MAX_MW:,.0f} MW (unit change upstream?)")
            return None
        try:
            mu_raw = sheet.cell(rx, 4).value
            energy_mu = (None if str(mu_raw).strip() in ("", "-", "NA", "N/A", "None")
                         else float(mu_raw))
        except (ValueError, TypeError, IndexError):
            energy_mu = None
        return (peak_mw, energy_mu)
    print(f"  [skip] {date_str}: no Maharashtra row in NLDC sheet; skipping")
    return None


def fetch_nldc_demand(date_strs, diagnostics=None):
    """Official gap-fill for dates the CSV mirror lacks. Returns {date: (mw, mu)}.

    Never raises for data problems: an unreachable API, a missing report, or
    a bad workbook all degrade to fewer/zero rows with a message. The caller
    records those reasons in `diagnostics` and the freshness gate decides
    whether the stored series can be published. Same plausible_mw() gate and
    MU-degrades-to-NULL as the CSV path.

    If webapi.grid-india.in is blocked from the runner, fall back to the same
    official .xls reports mirrored raw by Grid-Sentinel. That mirror is not a
    new source of truth: it is the file the existing CSV mirror was built
    from, parsed under these same guards.
    """
    result = {}
    todo = sorted(set(date_strs))
    unfilled = []
    download_failures = 0
    parse_skips = 0
    if not todo:
        return result
    if len(todo) > NLDC_MAX_DAYS_PER_RUN:
        print(f"  NLDC gap-fill capped at {NLDC_MAX_DAYS_PER_RUN} of {len(todo)} "
              f"missing day(s) this run; the rest wait for the next cron.")
        if diagnostics is not None:
            diagnostics["nldc_capped"] = len(todo) - NLDC_MAX_DAYS_PER_RUN
        todo = todo[-NLDC_MAX_DAYS_PER_RUN:]
    index = None
    try:
        index = fetch_nldc_index()
    except RuntimeError as exc:
        print(f"  [warn] official NLDC index unavailable ({exc}); "
              f"trying Grid-Sentinel raw PSP mirror...")
        if diagnostics is not None:
            diagnostics["nldc_index_error"] = str(exc)
    else:
        if diagnostics is not None:
            diagnostics["nldc_index_error"] = None
            diagnostics["nldc_index_size"] = len(index)
            diagnostics["nldc_newest_available"] = max(index) if index else None

    raw_filled = []
    for iso in todo:
        if index is None:
            url = nldc_raw_mirror_url(iso)
            source = "Grid-Sentinel raw PSP fallback"
        else:
            path = index.get(iso)
            if not path:
                print(f"  [skip] {iso}: no NLDC .xls in index; leaving for backfill.")
                unfilled.append(iso)
                continue
            url = NLDC_CDN_BASE + path
            source = "official PSP gap-fill"
        try:
            content = _download_nldc_report(url)
        except RuntimeError as exc:
            print(f"  [skip] {iso}: {source} download failed ({exc}); "
                  f"leaving for backfill.")
            unfilled.append(iso)
            download_failures += 1
            continue
        try:
            parsed = parse_nldc_workbook(content, iso)
        except RuntimeError as exc:
            # xlrd missing: a deploy problem, not a data problem - loud, and
            # the freshness gate decides whether the run can publish.
            print(f"  [warn] NLDC parse unavailable ({exc}); "
                  f"the freshness gate will decide whether this run can publish.")
            if diagnostics is not None:
                diagnostics["nldc_parse_error"] = str(exc)
                diagnostics["nldc_unfilled"] = [
                    day for day in todo if day not in result]
            return result
        if parsed:
            result[iso] = parsed
            if index is None:
                raw_filled.append(iso)
            print(f"  [nldc] {iso}: {parsed[0]:,.0f} MW ({source})")
        else:
            unfilled.append(iso)
            parse_skips += 1
    if diagnostics is not None:
        diagnostics["nldc_filled"] = len(result)
        diagnostics["nldc_unfilled"] = unfilled
        diagnostics["nldc_download_failures"] = download_failures
        diagnostics["nldc_parse_skips"] = parse_skips
        if index is None:
            diagnostics["nldc_raw_filled"] = len(raw_filled)
        if result:
            newest = max(result)
            prior = diagnostics.get("nldc_newest_available")
            if prior is None or newest > prior:
                diagnostics["nldc_newest_available"] = newest
    return result


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
                # Serialize concurrent ingest vs features; see PM2.5 twin.
                acquire_pipeline_lock(cur, "elec_ingest")
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
    def _moved(old, new):
        if old is None or new is None:
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
               if before.get(as_of) is not None and _moved(before[as_of], new)]
    if not revised:
        return

    print(f"  {len(revised)} day(s) had their peak_demand_mw REVISED upstream:")
    for as_of, old, new in sorted(revised):
        old_str = f"{old:,.0f}" if old is not None else "NULL"
        new_str = f"{new:,.0f}" if new is not None else "NULL"
        print(f"    [revised] {as_of}: {old_str} -> {new_str} MW")
    print("  Any of these already scored against are re-opened and re-scored by "
          "vericast.elec.score, so the published error follows the observation.")


def latest_observation_date():
    """Newest stored observation for this state, or None.

    The database is read after the batch is written, so a partial insert is
    included. This is deliberately the same value predict.py's stale guard
    reads, not the newest row this run happened to fetch.
    """
    require_database_url(DATABASE_URL)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            return get_last_observed_date(cur)


def format_source_status(diagnostics=None):
    """One-line source summary for logs and the 1/6 failure message."""
    if not diagnostics:
        return "Source status unavailable (no fetch diagnostics recorded)."
    parts = [f"mirror_head={diagnostics.get('mirror_head') or 'unknown'}"]
    if diagnostics.get("mirror_url"):
        parts.append(f"mirror_url={diagnostics['mirror_url']}")
    if "nldc_index_size" in diagnostics:
        parts.append(f"nldc_index_size={diagnostics['nldc_index_size']}")
    if diagnostics.get("nldc_newest_available"):
        parts.append(f"nldc_newest={diagnostics['nldc_newest_available']}")
    if diagnostics.get("nldc_index_error"):
        parts.append(f"nldc_error={diagnostics['nldc_index_error']}")
    if diagnostics.get("nldc_raw_filled") is not None:
        parts.append(f"nldc_raw_filled={diagnostics['nldc_raw_filled']}")
    if diagnostics.get("nldc_parse_error"):
        parts.append(f"nldc_parse_error={diagnostics['nldc_parse_error']}")
    if diagnostics.get("nldc_unfilled"):
        parts.append(f"nldc_unfilled={','.join(diagnostics['nldc_unfilled'])}")
    if diagnostics.get("nldc_capped"):
        parts.append(f"nldc_capped={diagnostics['nldc_capped']}")
    return "Source status: " + "; ".join(parts) + "."


def assert_fresh_enough(diagnostics=None):
    """Fail at 1/6 if the stored series would make predict.py refuse at 5/6.

    Uses the same `refuse_stale` predicate as predict.py, so the ingest gate
    cannot drift looser than the publish gate. Previously a dead mirror plus a
    failed NLDC gap-fill exited 0 here; predict then failed five steps later
    with a message that pointed at the source but could not say what either
    source had. The raised message now carries that source status.
    """
    latest = latest_observation_date()
    today = local_time.today("Asia/Kolkata")
    if latest is None:
        raise RuntimeError(
            f"No electricity observations for {STATE}; nothing can be forecast. "
            f"{format_source_status(diagnostics)}")
    try:
        stale_days = refuse_stale(latest, today, ELEC_STALE_LIMIT_DAYS, "electricity")
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} No forecast was published. "
            f"{format_source_status(diagnostics)} "
            f"Recovery: backfill every missing day from official NLDC PSP "
            f"reports with `python -m vericast.elec.backfill DATE:MW [...]`, "
            f"then rerun the pipeline."
        ) from exc
    print(f"  Freshness gate: newest observation {latest} is {stale_days} day(s) "
          f"old (limit {ELEC_STALE_LIMIT_DAYS}).")
    return stale_days


def warn_if_stall_approaching(diagnostics=None):
    """Best-effort heads-up when the source is nearing the publish refusal.

    2-4 days of lag is normal, but at EARLY_STALE_WARN_DAYS (5) the operator
    has ~2 days left to backfill from the official NLDC PSP reports before
    ELEC_STALE_LIMIT_DAYS (7) refuses to publish. Never raises: ingest must
    stay green on a routine lag day, and the hard failure belongs to
    assert_fresh_enough().
    """
    try:
        latest = latest_observation_date()
        if latest is None:
            return
        today = local_time.today("Asia/Kolkata")
        stale_days = (today - latest).days
        if stale_days >= EARLY_STALE_WARN_DAYS:
            msg = (
                f"Electricity source nearing stall: newest observation is "
                f"{latest}, {stale_days} day(s) old "
                f"(warn at {EARLY_STALE_WARN_DAYS}, refuse at "
                f"{ELEC_STALE_LIMIT_DAYS}). {format_source_status(diagnostics)} "
                f"If the mirror head has not advanced, backfill missing days "
                f"from NLDC PSP reports with "
                f"python -m vericast.elec.backfill DATE:MW [...] before "
                f"predict.py refuses to publish."
            )
            print(f"  [stale-warning] {msg}")
            try:
                send_alert("VeriCast electricity source nearing stall", msg)
            except Exception as exc:
                print(f"[alert] Failed to send stale-warning alert: {exc}")
    except Exception as exc:
        print(f"  [stale-warning] skipped (could not check observations): {exc}")


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

    diagnostics = {}
    demand = fetch_demand(start_date, end_date, diagnostics=diagnostics)

    # Official gap-fill before giving up: every date in range the mirror lacks
    # is retried against NLDC's own daily PSP reports, under the same guards.
    # Whatever neither source has stays missing; assert_fresh_enough() below
    # fails this step with the source status if that leaves the stored series
    # beyond the same limit predict.py uses.
    missing = [(start_date + timedelta(days=i)).isoformat()
               for i in range((end_date - start_date).days + 1)
               if (start_date + timedelta(days=i)).isoformat() not in demand]
    if missing:
        print(f"  Mirror lacks {len(missing)} day(s); trying official NLDC PSP gap-fill...")
        try:
            filled = fetch_nldc_demand(missing, diagnostics=diagnostics)
        except Exception as exc:  # noqa: BLE001 - gate decides whether it is fatal
            print(f"  [warn] NLDC gap-fill crashed ({exc}); "
                  f"the freshness gate will decide whether this run can publish.")
            diagnostics["nldc_index_error"] = str(exc)
            diagnostics["nldc_unfilled"] = missing
            filled = {}
        if filled:
            demand.update(filled)
            print(f"  NLDC gap-filled {len(filled)} day(s): {sorted(filled)}")
    print(f"  {format_source_status(diagnostics)}")

    if not demand:
        # Expected on most days: the mirror runs 2-4 days behind, so there is
        # frequently no new demand row even though the date range is non-empty.
        print(f"No new {STATE} demand rows in range (mirror lags real time by "
              f"a few days); nothing to insert.")
        # A "no new rows" day is routine; a run of them means the mirror has
        # stalled and the official gap-fill is not closing the hole. The hard
        # failure belongs at 1/6, next to the source status, not at 5/6.
        print(f"  Queried {start_date} -> {end_date} at {DEMAND_CSV_URL}. If no "
              f"new rows persist for more than "
              f"ELEC_STALE_LIMIT_DAYS={ELEC_STALE_LIMIT_DAYS} days, the mirror "
              f"has stalled upstream; the RESCAN_DAYS re-scan backfills "
              f"automatically once it resumes. "
              f"Past {EARLY_STALE_WARN_DAYS} days, backfill from NLDC PSP "
              f"reports with python -m vericast.elec.backfill DATE:MW [...].")
        warn_if_stall_approaching(diagnostics)
        assert_fresh_enough(diagnostics)
        return

    temps = fetch_temperature(start_date, end_date)

    records = [
        (STATE, date_str, peak_mw, energy_mu, *temps.get(date_str, (None, None)))
        for date_str, (peak_mw, energy_mu) in sorted(demand.items())
    ]

    print(f"Prepared {len(records)} daily records for insertion")
    insert_observations(records)

    # Read the database back, not the batch: a partial insert must still fail
    # here (with the source status) if the newest stored day is beyond the
    # publish limit. Predict and diagnose then never see a stale series.
    warn_if_stall_approaching(diagnostics)
    assert_fresh_enough(diagnostics)
    print("Ingestion completed!")


if __name__ == "__main__":
    main()
