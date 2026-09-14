"""Single source of truth for "what day is it" across the whole pipeline.

Every application-level date decision (what to ingest, what to score, what
date a forecast is for) must come from here. Postgres runs in GMT and the
GitHub Actions runner is UTC, so calling date.today() in different scripts
silently gave two different answers around midnight IST.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kolkata")


def today(tz=None):
    """Today's date in `tz` (a ZoneInfo/str), defaulting to Asia/Kolkata.

    Pass each target's own zone: UTC for PM2.5 (UTC target days), Asia/Kolkata
    for electricity. Comparing an IST `today` against a UTC-day `as_of`
    inflates staleness by one day between 00:00–05:30 IST.
    """
    zone = ZoneInfo(tz) if isinstance(tz, str) else (tz or TZ)
    return datetime.now(zone).date()


def today_in(tz_name):
    """Today's date in the named timezone (e.g. 'UTC', 'Asia/Kolkata')."""
    return today(tz_name)


def yesterday(tz=None):
    return today(tz) - timedelta(days=1)


if __name__ == "__main__":
    # Verify timezone database is available and IST offset (+05:30) is correct.
    assert (today() - yesterday()).days == 1
    now = datetime.now(TZ)
    assert now.utcoffset() == timedelta(hours=5, minutes=30), now.utcoffset()
    print(f"OK - {now.isoformat()} (today={today()}, yesterday={yesterday()})")
