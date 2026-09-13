"""Advance-forecast issuance policy (UTC PM2.5 days, IST demand days)."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def publication_timing(forecast_date, target_timezone, now=None):
    """Classify issuance, never move the model's t+1 target to meet a cutoff."""
    issued_at = now if now is not None else datetime.now(timezone.utc)
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("Issuance time must be timezone-aware")
    cutoff = datetime.combine(forecast_date, time.min, ZoneInfo(target_timezone))
    source = "daily" if issued_at < cutoff else "nowcast"
    return {
        "source": source,
        "issued_at": issued_at,
        "cutoff": cutoff,
        "timing_status": "advance_forecast" if source == "daily" else "delayed_estimate",
        "target_timezone": target_timezone,
        "feature_as_of": forecast_date - timedelta(days=1),
        "horizon_days": 1,
    }


def publish_predictions(cur, sql, records, forecast_date, target_timezone):
    """Classify after inference; all models share one explicit issuance time.

    If a daily batch crosses midnight while writing, roll back rather than
    backdating it. A retry will publish as nowcast. Existing rows are immutable.
    """
    timing = publication_timing(forecast_date, target_timezone)
    if not records:
        print(f"[WARN] No predictions to publish for {forecast_date} "
              f"(all models skipped); nothing written as {timing['source']}.")
        return timing
    for record in records:
        cur.execute(sql, (*record, timing["source"], timing["issued_at"]))
    if timing["source"] == "daily":
        # Second clock read (fresh now()): a batch that started before midnight
        # but finished after must not commit as daily. The caller rolls back on
        # the RuntimeError below.
        require_advance_forecast(forecast_date, target_timezone)
    print(f"[OK] Publication type: {timing['timing_status']} ({timing['source']})")
    return timing


def require_advance_forecast(forecast_date, target_timezone, now=None):
    """Reject issuance at or after the beginning of the target period.

    Do not shift forecast_date to satisfy this check: the model was trained for
    features(t) -> target(t+1). Delayed inputs require a different model horizon.
    """
    issued_at = now if now is not None else datetime.now(timezone.utc)
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("Issuance time must be timezone-aware")
    cutoff = datetime.combine(forecast_date, time.min, ZoneInfo(target_timezone))
    if issued_at >= cutoff:
        raise RuntimeError(
            f"Advance-forecast cutoff passed for {forecast_date} "
            f"({target_timezone}); refusing to publish a late estimate as daily. "
            "Use timely inputs or train a model for the actual forecast horizon."
        )
    return issued_at
