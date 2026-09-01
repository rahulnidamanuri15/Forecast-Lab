"""VeriCast pipeline package: one subpackage per target, plus shared `local_time`.

Run every script as a module from the repo root, e.g.
    python -m vericast.pm25.ingest

Model artifact paths are resolved from this file, not from the working
directory, so `python -m ...`, a cron job and the container all find the same
file. They stay committed to the repo (weekly-retrain.yml pushes them back).
"""
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PM25 = str(ROOT / "models" / "lightgbm_model.txt")
MODEL_ELEC = str(ROOT / "models" / "lightgbm_elec_model.txt")

# How many days behind an upstream source may fall before it counts as stalled.
# One definition per target, imported by every consumer, because the number is
# read in three different roles that have to agree: the diagnostic gate that
# stops the daily pipeline, the /health flag the dashboard's LIVE pill shows, and
# verify_deployment_readiness.py's go-live check. When those drifted apart the
# go-live check failed on data the pipeline gate passed and /health called fresh.
#
# The two numbers differ because the sources differ, and that is the whole reason
# they are named separately rather than folded into one STALE_LIMIT_DAYS:
#   PM2.5 - Open-Meteo publishes yesterday by ~05:00 UTC, so 1 stale day is the
#           steady state and 2 allows one dropped cron run.
#   Elec  - the HalcyonVector demand mirror normally runs 2-4 days behind real
#           time, so only past 5 has it actually stalled.
#
# Why any limit at all: both predict.py modules anchor forecast_date to
# latest_obs + 1 day, so a stalled source slides the anchor with it and every
# internal-consistency check downstream still passes. These are the only checks
# that look at the clock.
PM25_STALE_LIMIT_DAYS = 2
ELEC_STALE_LIMIT_DAYS = 5

# model_performance is keyed UNIQUE(score_date, model) with no city column - it
# predates the second target, and its electricity counterpart does have `state`.
# So the PM2.5 leaderboard can hold exactly one city, and running score.py under a
# different CITY would land Pune's rows on Nagpur's (score_date, model) keys and
# overwrite the published record in place, with nothing in the payload to show it.
#
# Naming it makes that a loud failure instead of a silent one: score.py refuses to
# write and app.py refuses to start under any other CITY. Everything else - the
# observations, features and predictions tables - is keyed on city and is fine, so
# CITY stays an env var for those. Changing this needs a real migration: a city
# column plus a UNIQUE(city, score_date, model) swap, which is a constraint change
# rather than an addition and therefore out of scope for vericast/schema.py.
PM25_CITY_OF_RECORD = "Nagpur"

# Physically plausible daily peak demand for a whole state, in MW. Maharashtra's
# observed 2023-2026 range is 20,147-32,419: wide enough here for growth and a mild
# winter, tight enough that a unit change (kW, GW) or a mis-parsed column fails
# instead of entering the record.
#
# Named here rather than inline in elec/ingest.py because it is a claim about the
# target, not about the fetch: the demand mirror is a third-party CSV tracked at a
# mutable `main`, and the column-name guard there catches a rename while this
# catches a value change - the failure a rename check cannot see.
ELEC_MIN_MW, ELEC_MAX_MW = 15_000.0, 40_000.0

# Physically plausible daily-mean PM2.5 for Nagpur, in ug/m3. The observed
# 2023-2026 CAMS range is roughly 4-160: wide enough here for a bad Diwali week,
# tight enough that a unit error or a sign flip fails instead of entering the
# record. Zero is excluded deliberately - a real daily mean over a city is never
# 0.0, so that value means "the field came back empty", not "clean air".
#
# Named here for the same reason as the MW pair, and used by both ends: the ingest
# guard in pm25/ingest.py (a bad *actual* would be scored against and become a
# permanent error nobody can attribute) and the publish gate in pm25/diagnose.py
# (a bad *forecast* would enter the public record).
PM25_MIN, PM25_MAX = 1.0, 500.0

# How far back an ingest run re-checks for holes. Both ingesters resume from
# MAX(as_of) + 1 day, which is correct for the steady state and permanent for a
# hole: a date the upstream skipped or served as NULL is behind the resume point
# forever after. Maharashtra's 2025-05-21 -> 05-24 gap is that, and so are the
# electricity forecast_dates with no daily row (2026-08-24/25/28) - the mirror
# never delivered those observations, so predict.py was never asked for them.
#
# 30 days because the sources revise within days, not months, and the window is
# what bounds the cost: a run with no holes still fetches one day.
RESCAN_DAYS = 30


def resume_start(last_date, gap_date):
    """Earliest date an ingest run must fetch, or None when the table is empty.

    MAX(as_of) + 1 day is the floor, not the answer: min() can only move the
    start EARLIER, so a table whose newest row predates the re-scan window still
    resumes from its own MAX rather than jumping forward over the months in
    between. That direction is the whole safety property - getting it backwards
    would turn a hole-filling re-scan into a hole-creating one.
    """
    if last_date is None:
        return None
    start = last_date + timedelta(days=1)
    return min(start, gap_date) if gap_date else start

