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

# Upstream staleness thresholds (days).
# PM2.5 (Open-Meteo) updates daily (limit: 2 days).
# Electricity (Grid-Sentinel) lags real-time by 2-4 days (limit: 5 days).
PM25_STALE_LIMIT_DAYS = 2
ELEC_STALE_LIMIT_DAYS = 5


def refuse_stale(as_of, today, limit_days, target):
    """Refuse to publish forecasts if the newest observation exceeds limit_days."""
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
# model_performance has no city column; the PM2.5 pipeline is scoped strictly to Nagpur.
PM25_CITY_OF_RECORD = "Nagpur"


def require_city_of_record(city):
    """Enforce single-city invariant for PM2.5 to prevent cross-city key collisions."""
    if city != PM25_CITY_OF_RECORD:
        raise RuntimeError(
            f"CITY={city!r} but this deployment is single-city: model_performance "
            f"has no city column, so only {PM25_CITY_OF_RECORD!r} can be ingested, "
            f"published, scored or served. See vericast/__init__.py."
        )
    return city


# Physical plausibility limits for target values.
# Catch unit errors (kW vs GW) or empty/corrupt fields before they enter the record.
ELEC_MIN_MW, ELEC_MAX_MW = 15_000.0, 40_000.0  # Maharashtra daily peak demand range (MW)
PM25_MIN, PM25_MAX = 1.0, 500.0                # Nagpur daily mean PM2.5 range (ug/m3)


def refuse_implausible(value, low, high, model, unit):
    """Validate that predicted/observed value is non-null and within [low, high]."""
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


# Historical lookback window (days) to re-scan for missing observations or upstream revisions.
RESCAN_DAYS = 30


def resume_start(last_date, gap_date):
    """Return earliest date to ingest, prioritizing historical gaps over monotonic advance."""
    if last_date is None:
        return None
    start = last_date + timedelta(days=1)
    return min(start, gap_date) if gap_date else start


# Floating-point tolerance for detecting ground truth revisions in observations.
ACTUAL_TOLERANCE = 1e-6

# Query to find daily predictions whose scored actuals differ from current observations.
_DIVERGED_SQL = """
SELECT p.forecast_date, p.model, p.{actual}, o.{observed}
FROM {predictions} p
JOIN {observations} o ON o.{key} = p.{key} AND o.as_of = p.forecast_date
WHERE p.{key} = %s
  AND p.source = 'daily'
  AND p.{actual} IS NOT NULL
  AND (o.{observed} IS NULL
       OR ABS(p.{actual} - o.{observed}) > {eps})
ORDER BY p.forecast_date, p.model
"""

# Re-opens diverged predictions by setting actual to NULL, allowing score.py to re-score them atomically.
_REOPEN_SQL = """
UPDATE {predictions} p
SET {actual} = NULL
FROM {observations} o
WHERE o.{key} = p.{key}
  AND o.as_of = p.forecast_date
  AND p.{key} = %s
  AND p.source = 'daily'
  AND p.{actual} IS NOT NULL
  AND p.{predicted} IS NOT NULL
  AND o.{observed} IS NOT NULL
  AND ABS(p.{actual} - o.{observed}) > {eps}
"""


def revision_sql(predictions, observations, key, actual, predicted, observed):
    """Return (diverged_sql, reopen_sql) query pair for detecting and reopening revised actuals."""
    names = {"predictions": predictions, "observations": observations, "key": key,
             "actual": actual, "predicted": predicted, "observed": observed,
             "eps": ACTUAL_TOLERANCE}
    return _DIVERGED_SQL.format(**names), _REOPEN_SQL.format(**names)


def reopen_revised_actuals(cur, diverged_sql, reopen_sql, key_value, unit):
    """Re-open scored rows whose ground truth observation changed upstream. Returns count re-opened."""
    cur.execute(diverged_sql, (key_value,))
    diverged = cur.fetchall()
    if not diverged:
        return 0

    for forecast_date, model, scored_against, observed_now in diverged:
        if observed_now is None:
            print(f"  [frozen] {forecast_date} {model}: scored against "
                  f"{scored_against:.2f} {unit}, but that observation is NULL now - "
                  f"no ground truth to re-score against, so the published actual and "
                  f"its error stand.")
        else:
            print(f"  [revised] {forecast_date} {model}: ground truth moved "
                  f"{scored_against:.2f} -> {observed_now:.2f} {unit} upstream; "
                  f"re-opening it to be re-scored below.")

    # Only a revision to a usable value re-opens anything; a [frozen] row has nothing
    # to re-score against and SCORE_SQL would not refill it.
    if not any(observed_now is not None for *_, observed_now in diverged):
        return 0

    cur.execute(reopen_sql, (key_value,))
    print(f"Re-opened {cur.rowcount} scored row(s) whose ground truth was revised "
          f"upstream; they are re-scored in this same transaction, so the new actual "
          f"and the new error land together.")
    return cur.rowcount


# Queries to verify that every feature row at t has a matching observation at t+1,
# ensuring orphan feature rows are fully accounted for by genuine observation gaps.
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
    """Return (gap_days_sql, orphan_rows_sql) query pair for target tables."""
    names = {"features": features, "observations": observations, "key": key}
    return _GAP_DAYS_SQL.format(**names), _ORPHAN_ROWS_SQL.format(**names)


def verify_alignment(cur, gap_sql, orphan_sql, key_value):
    """Enforce features(t) -> target(t+1) temporal alignment; orphan rows must match gaps."""
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

