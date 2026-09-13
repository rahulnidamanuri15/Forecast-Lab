"""Advance-forecast issuance policy (UTC PM2.5 days, IST demand days)."""
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


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
