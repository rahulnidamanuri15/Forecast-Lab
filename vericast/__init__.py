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
# One definition per target, imported by every consumer, because three roles have
# to agree on the number: the diagnostic gate that stops the daily pipeline,
# /health's LIVE pill, and verify_deployment_readiness.py's go-live check.
#
# The two differ because the sources differ:
#   PM2.5 - Open-Meteo publishes yesterday by ~05:00 UTC, so 1 stale day is the
#           steady state and 2 allows one dropped cron run.
#   Elec  - the demand mirror normally runs 2-4 days behind, so only past 5 has
#           it actually stalled.
#
# Why any limit: both predict.py modules anchor forecast_date to latest_obs + 1
# day, so a stalled source slides the anchor with it and every
# internal-consistency check downstream still passes.
PM25_STALE_LIMIT_DAYS = 2
ELEC_STALE_LIMIT_DAYS = 5


def refuse_stale(as_of, today, limit_days, target):
    """Age of `as_of` in days, raising past `limit_days`.

    Called by both predict.py modules *before* the forecast is committed. It
    used to be a [WARN] there and a hard failure in diagnose.py at step 6/6,
    which meant a forecast anchored to a stalled source was already public and
    already being served by /forecast by the time the job exited non-zero. A
    published number is the one thing this project cannot take back, so the gate
    moved ahead of the commit.
    """
    stale_days = (today - as_of).days
    if stale_days > limit_days:
        raise RuntimeError(
            f"{target} source has stalled: newest observation is {as_of}, "
            f"{stale_days} day(s) old, limit is {limit_days}. Refusing to "
            f"publish a forecast anchored to it."
        )
    return stale_days

# model_performance is keyed UNIQUE(score_date, model) with no city column - it
# predates the second target, whose counterpart does have `state`. So the PM2.5
# leaderboard holds exactly one city, and score.py under a different CITY would
# land another city's rows on Nagpur's keys and overwrite the published record in
# place, with nothing in the payload to show it. score.py and app.py both refuse to
# run under any other CITY, which makes that loud instead of silent. Everything
# else - observations, features, predictions - is keyed on city and is fine.
# Changing this needs a UNIQUE(city, score_date, model) migration, out of scope
# for vericast/schema.py.
PM25_CITY_OF_RECORD = "Nagpur"


def require_city_of_record(city):
    """Return `city`, refusing any value the published record cannot hold.

    Hoisted out of the three modules that had this check (app.py,
    vericast/pm25/score.py, experiments/save_backtest_results.py) and into the
    six that did not. The invariant is a property of the schema described above,
    so it belongs beside it rather than copied per caller - and it has to fire in
    every module, not just the ones that write model_performance: under another
    CITY, ingest and features and predict would happily populate a second city's
    rows that score.py then refuses to score and app.py refuses to serve. One
    loud failure at import beats a pipeline that half-runs.
    """
    if city != PM25_CITY_OF_RECORD:
        raise RuntimeError(
            f"CITY={city!r} but this deployment is single-city: model_performance "
            f"has no city column, so only {PM25_CITY_OF_RECORD!r} can be ingested, "
            f"published, scored or served. See vericast/__init__.py."
        )
    return city


# Physically plausible daily peak demand for a whole state, in MW. Maharashtra's
# observed 2023-2026 range is 20,147-32,419: wide enough for growth and a mild
# winter, tight enough that a unit change (kW, GW) or a mis-parsed column fails
# instead of entering the record. A claim about the target, not the fetch, so both
# the ingest and publish gates import it.
ELEC_MIN_MW, ELEC_MAX_MW = 15_000.0, 40_000.0

# Physically plausible daily-mean PM2.5 for Nagpur, in ug/m3. The observed
# 2023-2026 CAMS range is roughly 4-160. Zero is excluded deliberately: a real
# daily mean over a city is never 0.0, so that value means "the field came back
# empty", not "clean air". Used at both ends - a bad *actual* gets scored against
# and becomes a permanent error nobody can attribute; a bad *forecast* enters the
# public record.
PM25_MIN, PM25_MAX = 1.0, 500.0


def refuse_implausible(value, low, high, model, unit):
    """Return float(value), raising if it is None or outside [low, high].

    The publish gate. Both diagnose.py modules already range-check what was
    written, but they run as step 6/6 - after predict.py's commit - so a -40
    ug/m3 forecast was public, and served by /forecast, for as long as it took
    the job to fail. This is the same check one step earlier, where refusing
    still means nothing was published.

    Raised, not asserted: python -O erases `assert` and this guard stands
    between a broken model and the permanent record. Same reasoning as
    verify_alignment above.
    """
    if value is None:
        raise RuntimeError(
            f"{model} produced no value - refusing to publish NULL as a forecast."
        )
    value = float(value)
    if not (low <= value <= high):
        raise RuntimeError(
            f"{model} predicted {value:.2f} {unit}, outside the plausible range "
            f"{low}-{high} {unit}. Refusing to publish it. A value this far out "
            f"means a broken artifact, a unit change or corrupt features - not "
            f"weather."
        )
    return value


# How far back an ingest run re-checks for holes. Both ingesters resume from
# MAX(as_of) + 1 day, which is correct for the steady state and permanent for a
# hole: a date the upstream skipped or served as NULL is behind the resume point
# forever after (Maharashtra's 2025-05-21 -> 05-24 gap is one). 30 days because
# the sources revise within days, not months.
RESCAN_DAYS = 30


def resume_start(last_date, gap_date):
    """Earliest date an ingest run must fetch, or None when the table is empty.

    MAX(as_of) + 1 day is the floor, not the answer: min() can only move the
    start EARLIER, so a table whose newest row predates the re-scan window still
    resumes from its own MAX rather than jumping forward over the months in
    between. Backwards, this would create holes instead of filling them.
    """
    if last_date is None:
        return None
    start = last_date + timedelta(days=1)
    return min(start, gap_date) if gap_date else start


# Every feature row must have a next-day observation to be the target of, except
# where the observation series itself has a hole. Both counts come from the same
# shape so they can be compared directly: an orphan is only excusable if a gap
# explains it.
#
# Row existence, not "value IS NOT NULL": a low-coverage day is stored as a row
# with a NULL value (ingest.py's MIN_HOURS_PER_DAY) and train.py filters those.
# Broken-join territory is a *missing row*, which is what these count.
#
# Formatted, not parameterised, because table and column names cannot be bound -
# and the four values are module constants in vericast/{pm25,elec}/features.py,
# never anything a caller supplies. The key *value* is still bound.
_ORPHAN_ROWS_SQL = """
SELECT COUNT(*)
FROM {features} f
WHERE f.{key} = %s
  AND f.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM {observations} WHERE {key} = f.{key})
  AND NOT EXISTS (
      SELECT 1 FROM {observations} o
      WHERE o.{key} = f.{key}
        AND o.as_of = f.as_of + INTERVAL '1 day'
  )
"""

_GAP_DAYS_SQL = """
SELECT COUNT(*)
FROM {observations} o
WHERE o.{key} = %s
  AND o.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM {observations} WHERE {key} = o.{key})
  AND NOT EXISTS (
      SELECT 1 FROM {observations} n
      WHERE n.{key} = o.{key}
        AND n.as_of = o.as_of + INTERVAL '1 day'
  )
"""


def alignment_sql(features, observations, key):
    """The (gap-days, orphan-rows) query pair for one target's two tables."""
    names = {"features": features, "observations": observations, "key": key}
    return _GAP_DAYS_SQL.format(**names), _ORPHAN_ROWS_SQL.format(**names)


def verify_alignment(cur, gap_sql, orphan_sql, key_value):
    """Enforce the features(t) -> target(t+1) contract, gaps accounted for.

    Equality, not a tolerance: hardcoding "<= 1" for Maharashtra's known
    2025-05-21 -> 2025-05-24 hole absorbs the next gap silently, and absorbs a
    genuinely broken join just as quietly. Deriving the expected count from the
    observations means a NEW gap fails here, on the day it appears.

    Raised, not asserted: `assert` is erased by python -O / PYTHONOPTIMIZE=1, so
    the one interpreter flag someone adds for speed would turn every data gate in
    this pipeline into a no-op that still exits 0. Same reasoning in both train.py
    modules; the remaining bare asserts are all in `__main__` self-checks, where
    being erased costs nothing.
    """
    cur.execute(gap_sql, (key_value,))
    gaps = cur.fetchone()[0]
    cur.execute(orphan_sql, (key_value,))
    orphans = cur.fetchone()[0]

    if orphans != gaps:
        raise AssertionError(
            f"{orphans} feature rows have no next-day target but only {gaps} "
            f"observation gap(s) explain it - the features(t) -> target(t+1) "
            f"contract is broken"
        )
    print(f"  Alignment OK: {orphans} orphan row(s), all explained by "
          f"{gaps} observation gap(s)")

