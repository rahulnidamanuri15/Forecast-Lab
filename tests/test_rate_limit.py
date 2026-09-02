"""The rate limiter must cap a runaway client without capping everyone else.

The exposure it covers is the 8-slot connection pool: a client looping /evaluation
with a cache-busting query string walks straight past the Cache-Control middleware
and can hold every slot until requests start timing out at 10s.

Two failures matter more than the counting. Sharing one bucket across callers
would throttle the dashboard because a crawler is busy - and since Render's proxy
is `request.client.host` for every request, keying on that alone is exactly that
bug. Growing a dict key per caller forever would turn the limiter into the
memory leak, so the overflow bucket is checked too.

The window rolls over on distance from the *stored* start, not from whatever `now`
a test invents, and app.py is module state that the TestClient tests elsewhere in
this suite have already stamped with the real monotonic clock. So each test takes
its `now` from fresh_window(), which is always a full window ahead of both.
"""
from time import monotonic

from app import (
    RATE_LIMIT_MAX_CLIENTS,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    over_rate_limit,
)

_offset = 0


def fresh_window():
    """A `now` a full window ahead of the real clock and of every earlier call.

    Ahead of the real clock because other tests in this suite drive the middleware
    through TestClient and leave `monotonic()` in app.py's window start; ahead of
    earlier calls because two tests reading the clock microseconds apart would
    otherwise land inside one window and share a counter.
    """
    global _offset
    _offset += 1000
    return monotonic() + _offset


def test_traffic_under_the_cap_is_never_limited():
    now = fresh_window()
    assert not any(over_rate_limit("1.1.1.1", now)
                   for _ in range(RATE_LIMIT_MAX_REQUESTS))


def test_the_request_past_the_cap_is_limited():
    now = fresh_window()
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        over_rate_limit("2.2.2.2", now)
    assert over_rate_limit("2.2.2.2", now)


def test_one_loud_client_does_not_limit_a_quiet_one():
    """The bug this guards: a shared counter takes the dashboard down because a
    crawler is busy."""
    now = fresh_window()
    for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
        over_rate_limit("3.3.3.3", now)
    assert not over_rate_limit("4.4.4.4", now)


def test_a_new_window_forgives():
    now = fresh_window()
    for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
        over_rate_limit("5.5.5.5", now)
    assert over_rate_limit("5.5.5.5", now)
    assert not over_rate_limit("5.5.5.5", now + RATE_LIMIT_WINDOW_SECONDS)


def test_key_count_is_bounded_so_the_limiter_is_not_the_leak():
    """A caller rotating X-Forwarded-For must not grow the dict without limit."""
    now = fresh_window()
    for i in range(RATE_LIMIT_MAX_CLIENTS + 50):
        over_rate_limit(f"6.6.6.{i}", now)

    import app
    assert len(app._rate_hits) <= RATE_LIMIT_MAX_CLIENTS + 1  # + overflow bucket


def test_limits_are_above_dashboard_usage_and_below_pool_saturation():
    """index.html makes ~6 calls per load; the pool has 8 slots at a 10s timeout."""
    assert RATE_LIMIT_MAX_REQUESTS >= 60
    assert RATE_LIMIT_WINDOW_SECONDS <= 60


def test_the_middleware_is_wired_and_off_under_the_test_client():
    """The counter being right is worthless if the middleware never consults it,
    and the skip being wrong would 429 the rest of this suite as it grows.

    So both halves: outside the lifespan (every other test's state) a flood stays
    200, and under it the same flood ends in a 429 carrying Retry-After.

    Keyed on `_under_lifespan`, not on `_pool`: the limiter must stay on in a
    deployment whose pool failed to open, which is precisely when one runaway
    client does the most damage.
    """
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)
    flood = RATE_LIMIT_MAX_REQUESTS + 2

    assert app_module._under_lifespan is False
    assert all(client.get("/").status_code == 200 for _ in range(flood))

    # No pool, but under the lifespan: still throttled.
    with patch.object(app_module, "_under_lifespan", True):
        app_module._rate_window_start, app_module._rate_hits = 0.0, {}
        codes = [client.get("/").status_code for _ in range(flood)]

    app_module._rate_window_start, app_module._rate_hits = 0.0, {}

    assert codes[0] == 200
    assert codes[-1] == 429
    limited = client.get("/")  # window state was reset, so this one is fine again
    assert limited.status_code == 200


def test_a_missing_pool_under_the_lifespan_is_a_503_not_a_silent_connect():
    """The flaw the `_under_lifespan` split closes.

    With one flag meaning both "no pool" and "not under the lifespan", a lifespan
    that failed to build a pool served every request on a fresh direct connection,
    unthrottled, until Neon's connection cap turned it into a 500 for everyone.
    """
    from unittest.mock import patch

    import pytest
    from fastapi import HTTPException

    import app as app_module

    with patch.object(app_module, "_under_lifespan", True), \
         patch.object(app_module, "_pool", None):
        with pytest.raises(HTTPException) as excinfo:
            app_module.get_db_connection()

    assert excinfo.value.status_code == 503
