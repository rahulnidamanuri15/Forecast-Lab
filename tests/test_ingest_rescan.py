"""The ingest re-scan must only ever move the start date EARLIER.

Both ingesters resume from MAX(as_of) + 1 day. That is correct for the steady
state and permanent for a hole: a date the upstream skipped or served as NULL
sits behind the resume point forever, which is how Maharashtra's
2025-05-21 -> 05-24 gap became permanent. `resume_start()` bounds a re-scan of
the last RESCAN_DAYS days against that floor.

The direction is the whole safety property. min() moves the start earlier, so a
table whose newest row predates the re-scan window still resumes from its own
MAX; max() would jump the start forward over the months in between and create
the hole it exists to fill. Nothing else in the pipeline would notice - the
window frames would just null out across a wider gap - so this is the check.
"""
from datetime import date

from vericast import RESCAN_DAYS, resume_start


def test_no_hole_resumes_the_day_after_the_latest_row():
    assert resume_start(date(2026, 8, 28), None) == date(2026, 8, 29)


def test_a_hole_pulls_the_start_back_to_it():
    assert resume_start(date(2026, 8, 28), date(2026, 8, 24)) == date(2026, 8, 24)


def test_a_hole_after_the_resume_point_never_pushes_the_start_forward():
    """The failure this file exists for: a stale table (newest row months old)
    with a hole inside the recent window must still resume from its own MAX,
    or the re-scan skips every day in between and makes the gap permanent."""
    assert resume_start(date(2026, 1, 10), date(2026, 8, 24)) == date(2026, 1, 11)


def test_empty_table_defers_to_the_caller():
    """None means "no rows yet", which is INITIAL_START's job, not this one's."""
    assert resume_start(None, date(2026, 8, 24)) is None


def test_rescan_window_is_bounded():
    """Unbounded would re-read the whole series every run."""
    assert 0 < RESCAN_DAYS <= 90
