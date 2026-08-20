"""Pipeline alignment invariants, checked against the live database.

These are the invariants that silently break when a step is skipped or a
timezone slips: features must be engineered up to the latest observation, and
a forecast must be labelled for the day after the data it was built from.
Skipped (not failed) when DATABASE_URL is absent, so the API-only tests still
run in a bare checkout.
"""
import os
import sys
from datetime import timedelta

import psycopg
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def cur():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as c:
            yield c


def _scalar(cur, sql, params=()):
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


def test_features_reach_latest_observation(cur):
    """engineer_features.py must have run for the newest observation."""
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
