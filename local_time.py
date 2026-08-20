"""Single source of truth for "what day is it" across the whole pipeline.

Every application-level date decision (what to ingest, what to score, what
date a forecast is for) must come from here. Postgres runs in GMT and the
GitHub Actions runner is UTC, so calling date.today() in different scripts
silently gave two different answers around midnight IST.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kolkata")


def today():
    """Today's date in the operating timezone (Asia/Kolkata)."""
    return datetime.now(TZ).date()


def yesterday():
    return today() - timedelta(days=1)


if __name__ == "__main__":
    # ponytail: smallest check that fails if the tz database is missing from
    # the image (python:*-slim ships without it) or the offset is wrong.
    assert (today() - yesterday()).days == 1
    now = datetime.now(TZ)
    assert now.utcoffset() == timedelta(hours=5, minutes=30), now.utcoffset()
    print(f"OK - {now.isoformat()} (today={today()}, yesterday={yesterday()})")
