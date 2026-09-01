"""Pipeline alignment invariants, checked against a real Postgres.

These are the invariants that silently break when a step is skipped or a
timezone slips: features must be engineered up to the latest observation, and
a forecast must be labelled for the day after the data it was built from.

Two databases, one suite. Against the live database these assert on real
pipeline output. Against CI's empty throwaway Postgres they seed themselves
first (see `_seed` below) so the same SQL still runs - which is the point: these
are the only tests that execute the window-frame queries the leakage
guarantee rests on. `_seed` refuses any non-local host, so pointing
DATABASE_URL at the managed instance can never fabricate observations there.

Still skipped rather than failed when no database is *reachable* at all, so a
bare checkout runs the API-only tests instead of erroring.
"""
import os
import sys
from datetime import timedelta

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")
STATE = os.getenv("STATE", "Maharashtra")

def _unreachable():
    """Why the live database can't be used, or None if it can."""
    if not DATABASE_URL:
        return "DATABASE_URL not set"
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=5).close()
        return None
    except Exception as exc:
        return f"database unreachable: {type(exc).__name__}"


# Called once, not once per pytestmark argument: each call opens a real
# connection, and two connection attempts for one skip decision is two chances
# to hang on a 5s timeout.
_SKIP_REASON = _unreachable()

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


@pytest.fixture(scope="module")
def cur():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as c:
            _seed(conn, c)
            yield c


def _scalar(cur, sql, params=()):
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------
# Seeding, for CI's empty throwaway Postgres. No-op against any database that
# already has rows, so running with the live DSN still asserts on real pipeline
# output rather than on fixtures.
#
# The dates are the real ones on purpose: 2025-05-21 -> 2025-05-24 is the actual
# Maharashtra gap, and reproducing it is the whole point - without it
# test_elec_lag_is_null_across_the_date_gap has nothing to check.
#
# Only the *observations* are synthetic. The features tables are built by calling
# the production engineer_features(), so CI runs the real RANGE ... PRECEDING
# frames and COUNT(<col>) OVER wN guards. Hand-writing feature rows here would make
# the leakage tests assert against this file instead of against the query.
# --------------------------------------------------------------------------

# 40 consecutive days, so the newest features row has a full 30-day window and
# test_latest_features_row_is_complete finds no NULLs.
SEED_PM25 = """
INSERT INTO observations
    (city, as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum)
SELECT %s, d::date,
       45 + 12 * sin(n::float8 / 4),
       80 + 20 * sin(n::float8 / 4),
       31 +  3 * cos(n::float8 / 5),
       11 +  4 * cos(n::float8 / 3),
       GREATEST(0, 5 * sin(n::float8 / 3))
FROM generate_series(DATE '2025-05-02', DATE '2025-06-10', INTERVAL '1 day')
     WITH ORDINALITY AS g(d, n)
ON CONFLICT (city, as_of) DO NOTHING;
"""

# 2025-04-01 -> 2025-06-30 with 05-22 and 05-23 removed: 05-21 is the last day
# before the hole and 05-24 the first day after it, exactly as upstream.
#
# Both ends are sized off the 30-day window. The leading run is >= 30 days so
# test_elec_rolling_30_requires_a_full_window sees its first 29 rows null out on
# window size rather than on the gap. The trailing run is >= 30 days *past* the
# gap (06-30 - 29 days = 06-01) so the newest row's w30 counts 30 real days;
# ending at 06-10 instead leaves demand_roll_30_mean NULL on the newest row,
# because that window still straddles the hole.
SEED_ELEC = """
INSERT INTO electricity_observations
    (state, as_of, peak_demand_mw, energy_met_mu, temperature_2m_mean, temperature_2m_max)
SELECT %s, d::date,
       24000 + 2500 * sin(n::float8 / 6),
       480 + 40 * sin(n::float8 / 6),
       30 + 4 * cos(n::float8 / 7),
       36 + 4 * cos(n::float8 / 7)
FROM generate_series(DATE '2025-04-01', DATE '2025-06-30', INTERVAL '1 day')
     WITH ORDINALITY AS g(d, n)
WHERE d::date NOT BETWEEN DATE '2025-05-22' AND DATE '2025-05-23'
ON CONFLICT (state, as_of) DO NOTHING;
"""

# One published forecast per model at MAX(as_of) + 1 day - the t -> t+1 contract
# the tests below assert. Written directly rather than by running predict.py,
# which needs a trained artifact and a features row this seed cannot guarantee.
# Left unscored: on the daily path nothing fills actual_* except the score.py
# files. (The one exception lives outside that path -
# experiments/save_*_backtest_results.py seed the launch record with actuals a
# walk-forward backtest already knows. They are never run from the pipeline.)
SEED_PM25_PREDICTIONS = """
INSERT INTO predictions (city, forecast_date, predicted_pm2_5, model)
SELECT %s, MAX(as_of) + 1, 47.5, m
FROM observations, unnest(ARRAY['lightgbm', 'naive_baseline']) AS t(m)
WHERE city = %s
GROUP BY m
ON CONFLICT (city, forecast_date, model) DO NOTHING;
"""

SEED_ELEC_PREDICTIONS = """
INSERT INTO electricity_predictions (state, forecast_date, predicted_demand_mw, model)
SELECT %s, MAX(as_of) + 1, 24500, m
FROM electricity_observations,
     unnest(ARRAY['lightgbm', 'naive_baseline', 'seasonal_naive']) AS t(m)
WHERE state = %s
GROUP BY m
ON CONFLICT (state, forecast_date, model) DO NOTHING;
"""


def _seed(conn, c):
    """Populate an empty database with the shape the tests below assert on."""
    if (_scalar(c, "SELECT COUNT(*) FROM observations")
            or _scalar(c, "SELECT COUNT(*) FROM electricity_observations")):
        return

    # An empty table is normally CI's throwaway Postgres - but it is also what a
    # fresh managed instance looks like, and the seed below writes 40 synthetic
    # days plus unscored forecasts. Writing those into anything but a local
    # database would contaminate the published record with fabricated
    # observations, so refuse rather than seed. Skip, not fail: the live DSN
    # having no rows yet is a legitimate state, it just isn't one these tests can
    # bootstrap themselves out of.
    host = conninfo_to_dict(DATABASE_URL).get("host", "")
    if host not in ("localhost", "127.0.0.1", "::1", ""):
        pytest.skip(
            f"refusing to seed synthetic observations into a remote database "
            f"(host {host!r}); run the pipeline against it instead, or point "
            f"DATABASE_URL at a local throwaway Postgres")

    c.execute(SEED_PM25, (CITY,))
    c.execute(SEED_ELEC, (STATE,))
    c.execute(SEED_PM25_PREDICTIONS, (CITY, CITY))
    c.execute(SEED_ELEC_PREDICTIONS, (STATE, STATE))
    # Commit before engineering: both engineer_features() open their own
    # connection and would not see uncommitted rows.
    conn.commit()

    # Imported here, not at module scope: a bare checkout with no reachable
    # database skips this whole module and must not pay for the import.
    from vericast.elec.features import engineer_features as engineer_elec
    from vericast.pm25.features import engineer_features as engineer_pm25

    engineer_pm25()
    engineer_elec()
    print("[seed] empty database populated with synthetic observations")


def test_features_reach_latest_observation(cur):
    """vericast/pm25/features.py must have run for the newest observation."""
    latest_obs = _scalar(cur, "SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
    latest_feat = _scalar(cur, "SELECT MAX(as_of) FROM features WHERE city = %s", (CITY,))
    assert latest_obs is not None, "no observations at all"
    assert latest_feat == latest_obs, f"features stop at {latest_feat}, observations at {latest_obs}"


def test_latest_features_row_is_complete(cur):
    """A NULL in the newest features row means LightGBM would predict garbage."""
    cur.execute("""
        SELECT * FROM features WHERE city = %s ORDER BY as_of DESC LIMIT 1
    """, (CITY,))
    row = cur.fetchone()
    assert row is not None, "no features rows"
    names = [d[0] for d in cur.description]
    nulls = [n for n, v in zip(names, row) if v is None and n not in ("city", "as_of", "id")]
    assert not nulls, f"NULL feature columns: {nulls}"


def test_forecast_date_is_latest_observation_plus_one(cur):
    """The core t -> t+1 contract: feature_date + 1 day == target_date."""
    latest_obs = _scalar(cur, "SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
    expected = latest_obs + timedelta(days=1)

    for model in ("lightgbm", "naive_baseline"):
        got = _scalar(cur, """
            SELECT MAX(forecast_date) FROM predictions WHERE city = %s AND model = %s
        """, (CITY, model))
        assert got == expected, f"{model} latest forecast_date is {got}, expected {expected}"


def test_both_models_present(cur):
    """Naive baseline has to exist for every LightGBM forecast, or there is
    nothing to compare against - the whole point of the leaderboard."""
    cur.execute("""
        SELECT DISTINCT model FROM predictions WHERE city = %s
    """, (CITY,))
    models = {r[0] for r in cur.fetchall()}
    assert {"lightgbm", "naive_baseline"} <= models, f"models present: {sorted(models)}"


def test_no_forecast_precedes_its_own_features(cur):
    """A prediction for date D may only exist if features exist for D-1.
    Catches a forecast written from a future features row (leakage) or from
    no features row at all."""
    orphans = _scalar(cur, """
        SELECT COUNT(*)
        FROM predictions p
        WHERE p.city = %s
          AND p.model = 'lightgbm'
          AND NOT EXISTS (
              SELECT 1 FROM features f
              WHERE f.city = p.city
                AND f.as_of = p.forecast_date - INTERVAL '1 day'
          )
    """, (CITY,))
    assert orphans == 0, f"{orphans} lightgbm prediction(s) have no features row for the prior day"


def test_scored_predictions_match_observations(cur):
    """actual_pm2_5 must equal the observation for that date - not a
    neighbouring day, which is what a timezone slip would produce."""
    mismatches = _scalar(cur, """
        SELECT COUNT(*)
        FROM predictions p
        JOIN observations o ON o.city = p.city AND o.as_of = p.forecast_date
        WHERE p.city = %s
          AND p.actual_pm2_5 IS NOT NULL
          AND ABS(p.actual_pm2_5 - o.pm2_5) > 1e-6
    """, (CITY,))
    assert mismatches == 0, f"{mismatches} scored prediction(s) disagree with observations"


def test_pm25_every_feature_row_has_a_next_day_target(cur):
    """The features(t) -> target(t+1) contract for PM2.5, as an orphan count.

    Delegates to the production verify_alignment(), same as its electricity
    counterpart below, so the daily job and this test share one definition. The
    PM2.5 seed has no date gap, so here it asserts 0 == 0 - the regression it
    guards is a *future* orphan appearing, which is what the equality catches and
    a hardcoded tolerance would not.
    """
    from vericast.pm25.features import verify_alignment

    verify_alignment(cur)


# --------------------------------------------------------------------------
# Electricity pipeline. The lag/rolling columns come from Postgres date-addressed
# window frames rather than a Python loop, so what needs verifying is that the
# frames really do null themselves out across the known date gap instead of
# quietly reaching over it.
# --------------------------------------------------------------------------

def test_elec_every_feature_row_has_a_next_day_target(cur):
    """The features(t) -> target(t+1) contract, as an orphan count.

    Delegates to the production verify_alignment(), which the daily job also runs
    at the end of engineer_features() - one definition of the contract, exercised
    both here against the seeded 2025-05-22..23 hole and daily against live data.
    It derives the expected orphan count from the observed gaps rather than
    tolerating a hardcoded 1, so a new gap fails instead of being pre-forgiven.
    """
    from vericast.elec.features import verify_alignment

    verify_alignment(cur)


def test_elec_lag_is_null_across_the_date_gap(cur):
    """2025-05-24 follows a 2-day hole, so demand_lag_1 must be NULL.

    This is the leakage guarantee: `RANGE BETWEEN INTERVAL '1 day' PRECEDING AND
    INTERVAL '1 day' PRECEDING` is addressed by date, not row position, so it
    returns NULL rather than silently grabbing 2025-05-21's value.
    """
    cur.execute("""
        SELECT demand_lag_1, demand_lag_2, demand_lag_6
        FROM electricity_features
        WHERE state = %s AND as_of = DATE '2025-05-24'
    """, (STATE,))
    row = cur.fetchone()
    if row is None:
        pytest.skip("2025-05-24 not in the ingested range")
    lag_1, lag_2, lag_6 = row
    assert lag_1 is None, f"demand_lag_1 should be NULL across the gap, got {lag_1}"
    assert lag_2 is None, f"demand_lag_2 should be NULL across the gap, got {lag_2}"
    assert lag_6 is not None, "demand_lag_6 reaches back past the gap and should be set"


def test_elec_rolling_30_requires_a_full_window(cur):
    """demand_roll_30_mean must be NULL until 30 days of history exist.

    COUNT(peak_demand_mw) OVER w30 = 30 is a stricter guard than a row-index
    check: it also nulls out any window that a date gap left short, or that a row
    with a NULL value left one value short.
    """
    first_29_nonnull = _scalar(cur, """
        SELECT COUNT(*) FROM (
            SELECT demand_roll_30_mean
            FROM electricity_features
            WHERE state = %s
            ORDER BY as_of
            LIMIT 29
        ) head
        WHERE demand_roll_30_mean IS NOT NULL
    """, (STATE,))
    assert first_29_nonnull == 0, (
        f"{first_29_nonnull} of the first 29 rows have a 30-day mean with under 30 days of data"
    )
