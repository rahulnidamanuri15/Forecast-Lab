import os
import psycopg
import subprocess
import sys
from datetime import timedelta
from dotenv import load_dotenv
import httpx

from vericast import (
    MODEL_ELEC,
    MODEL_PM25,
    PM25_STALE_LIMIT_DAYS,
    local_time,
    require_city_of_record,
)
from vericast.elec.train import FEATURE_COLUMNS as ELEC_FEATURE_COLUMNS
from vericast.pm25.train import FEATURE_COLUMNS

load_dotenv()

# Override to smoke-test a deployed instance instead of the local process.
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

# Same env read as app.py and every vericast module: with the city hardcoded, the
# one gate meant to catch a misconfiguration was the only place that could not see
# it - point CITY at a city with no rows and this file still queried Nagpur. And
# the same refusal as app.py: a go-live gate that passes under a CITY the API will
# refuse to boot under is worse than no gate.
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))
STATE = os.getenv("STATE", "Maharashtra")

# The two targets differ only in artifact path, table names, key column and value
# column, so the three DB checks below take a descriptor instead of being written
# twice. Before this the elec half had no DB check at all - its artifact, its daily
# prediction and its forecast-date anchor were unverified by the go-live gate, and
# only its HTTP endpoints were covered.
#
# seasonal_naive is deliberately absent from `models`: it legitimately skips when
# demand_lag_6 is NULL across a date gap, so requiring it would FAIL a correct
# deployment. check_api_electricity_endpoints() covers it over HTTP instead.
TARGETS = {
    "PM2.5": {
        "artifact": MODEL_PM25,
        "observations": "observations",
        "features": "features",
        "feature_columns": FEATURE_COLUMNS,
        "predictions": "predictions",
        "key": "city",
        "key_value": CITY,
        "value": "predicted_pm2_5",
        "fmt": ".2f",
        "unit": "ug/m3",
        "models": ("lightgbm", "naive_baseline"),
    },
    "electricity": {
        "artifact": MODEL_ELEC,
        "observations": "electricity_observations",
        "features": "electricity_features",
        "feature_columns": ELEC_FEATURE_COLUMNS,
        "predictions": "electricity_predictions",
        "key": "state",
        "key_value": STATE,
        "value": "predicted_demand_mw",
        "fmt": ".0f",
        "unit": "MW",
        "models": ("lightgbm", "naive_baseline"),
    },
}

def check_database_url():
    """Check that DATABASE_URL exists"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL environment variable is not set")
        return False
    print("PASS: DATABASE_URL is set")
    return True

def check_postgres_connectivity():
    """Check that PostgreSQL is reachable"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping PostgreSQL connectivity check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result[0] == 1:
                    print("PASS: PostgreSQL is reachable")
                    return True
                else:
                    print("FAIL: PostgreSQL connectivity test failed")
                    return False
    except Exception as e:
        print(f"FAIL: Error connecting to PostgreSQL: {e}")
        return False

def check_observations_freshness():
    """Check that observations are within the shared PM2.5 staleness limit.

    The limit is PM25_STALE_LIMIT_DAYS from vericast/__init__.py, not a local
    number: a stricter one here FAILs go-live on data the daily pipeline passed.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping observations freshness check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print("FAIL: No observations found in database")
                    return False

                # Same "today" the pipeline uses (Asia/Kolkata), not UTC, or this
                # gate disagrees with the scripts it is gating.
                today = local_time.today()
                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs

                days_stale = (today - latest_obs_date).days

                if days_stale <= PM25_STALE_LIMIT_DAYS:
                    print(f"PASS: Observations are fresh (latest: {latest_obs_date}, stale days: {days_stale})")
                    return True
                else:
                    print(f"FAIL: Observations are stale (latest: {latest_obs_date}, "
                          f"stale days: {days_stale}, limit: {PM25_STALE_LIMIT_DAYS})")
                    return False
    except Exception as e:
        print(f"FAIL: Error checking observations freshness: {e}")
        return False

def check_features_match_observations():
    """Check that features latest date == observations latest date"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping features/observations match check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
                latest_obs = cur.fetchone()[0]

                cur.execute("SELECT MAX(as_of) FROM features WHERE city = %s", (CITY,))
                latest_feat = cur.fetchone()[0]

                if latest_obs is None or latest_feat is None:
                    print("FAIL: Missing observations or features data")
                    return False

                obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                feat_date = latest_feat.date() if hasattr(latest_feat, 'date') else latest_feat

                if obs_date == feat_date:
                    print(f"PASS: Features and observations dates match ({obs_date})")
                    return True
                else:
                    print(f"FAIL: Features and observations dates mismatch (obs: {obs_date}, feat: {feat_date})")
                    return False
    except Exception as e:
        print(f"FAIL: Error checking features/observations match: {e}")
        return False

def check_features_no_nulls(target="PM2.5"):
    """Check that `target`'s latest features row has no NULLs in the model's columns.

    The target's own FEATURE_COLUMNS, not SELECT * minus a denylist: the model reads
    exactly those columns, so those are the ones whose NULL stops a forecast. A
    denylist FAILs go-live on any new nullable column the model never looks at, and
    drifts the moment one is added. Same columns the target's diagnose.py checks.

    Parameterised for the same reason as check_prediction_exists: electricity_features
    has its own date gaps - the mirror runs 2-4 days behind - and a NULL there stops
    /electricity/forecast?model=lightgbm exactly as this one stops PM2.5's.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print(f"FAIL: DATABASE_URL not set, skipping {target} features NULL check")
        return False

    t = TARGETS[target]
    columns = t["feature_columns"]

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # ponytail: f-string table/column names as in check_prediction_exists
                # - every value comes from the TARGETS literal above, never a caller.
                cur.execute(f"""
                    SELECT {", ".join(columns)} FROM {t['features']}
                    WHERE {t['key']} = %s
                    ORDER BY as_of DESC
                    LIMIT 1
                """, (t["key_value"],))
                row = cur.fetchone()

                if row is None:
                    print(f"FAIL: No {target} features found")
                    return False

                null_cols = [col for col, val in zip(columns, row) if val is None]

                if not null_cols:
                    print(f"PASS: Latest {target} features have no NULL values")
                    return True
                else:
                    print(f"FAIL: Latest {target} features have NULL values in "
                          f"columns: {null_cols}")
                    return False
    except Exception as e:
        print(f"FAIL: Error checking {target} features for NULLs: {e}")
        return False

def check_leakage_test(module='vericast.pm25.leakage_test'):
    """Run one target's leakage test and require it to pass.

    Parameterised rather than duplicated: the two targets' tests run identically,
    differing only in the module name. The default keeps the PM2.5 call site short.
    """
    print(f"Running leakage test ({module}):")
    try:
        result = subprocess.run([
            sys.executable, '-m', module
        ], capture_output=True, text=True, cwd=os.getcwd(), timeout=180)

        if result.returncode == 0:
            print("PASS: Leakage test PASSED")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print("FAIL: Leakage test FAILED:")
            if result.stdout.strip():
                print(f"   STDOUT: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"   STDERR: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("FAIL: Leakage test timed out (>180 seconds)")
        return False
    except Exception as e:
        print(f"FAIL: Error running leakage test: {e}")
        return False

def check_model_artifact(target="PM2.5"):
    """Check that `target`'s LightGBM artifact exists and is not empty."""
    model_path = TARGETS[target]["artifact"]
    if os.path.exists(model_path):
        if os.path.getsize(model_path) > 0:
            print(f"PASS: {target} LightGBM model artifact exists and is not empty")
            return True
        else:
            print(f"FAIL: {target} LightGBM model artifact exists but is empty")
            return False
    else:
        print(f"FAIL: {target} LightGBM model artifact not found at {model_path}")
        return False

def check_prediction_exists(model="lightgbm", target="PM2.5"):
    """Check that `model` has a daily prediction for `target`'s next day.

    Parameterised like check_leakage_test above: the two models' checks differed
    only in the model literal, and check_forecast_date_logic already loops the
    same query over the same pair. `target` extends that to the second target,
    which had no DB check at all.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print(f"FAIL: DATABASE_URL not set, skipping {target} {model} prediction check")
        return False

    t = TARGETS[target]

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # ponytail: f-string table/column names - every value comes from the
                # TARGETS literal above, never from a caller. Key values are bound.
                cur.execute(
                    f"SELECT MAX(as_of) FROM {t['observations']} WHERE {t['key']} = %s",
                    (t["key_value"],),
                )
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print(f"FAIL: No {target} observations found to determine forecast date")
                    return False

                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                expected_forecast_date = latest_obs_date + timedelta(days=1)

                # source = 'daily' only: a backtest row on that date was written with
                # the actual in hand, so it proves nothing about today's pipeline.
                cur.execute(f"""
                    SELECT forecast_date, {t['value']}
                    FROM {t['predictions']}
                    WHERE {t['key']} = %s AND model = %s AND forecast_date = %s
                      AND source = 'daily'
                """, (t["key_value"], model, expected_forecast_date))

                row = cur.fetchone()

                if row is None:
                    print(f"FAIL: No {target} {model} prediction found for forecast "
                          f"date {expected_forecast_date}")
                    return False

                forecast_date, predicted_value = row
                if predicted_value is None:
                    print(f"FAIL: {target} {model} prediction exists for {forecast_date} "
                          f"but {t['value']} is NULL")
                    return False

                print(f"PASS: {target} {model} prediction exists for {forecast_date} "
                      f"({predicted_value:{t['fmt']}} {t['unit']})")
                return True
    except Exception as e:
        print(f"FAIL: Error checking {target} {model} prediction: {e}")
        return False

def check_forecast_date_logic(target="PM2.5"):
    """Check that forecast_date == latest_observation + 1 for `target`'s models."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print(f"FAIL: DATABASE_URL not set, skipping {target} forecast date logic check")
        return False

    t = TARGETS[target]

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT MAX(as_of) FROM {t['observations']} WHERE {t['key']} = %s",
                    (t["key_value"],),
                )
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print(f"FAIL: No {target} observations found")
                    return False

                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                expected_forecast_date = latest_obs_date + timedelta(days=1)

                all_good = True

                for model in t["models"]:
                    cur.execute(f"""
                        SELECT forecast_date
                        FROM {t['predictions']}
                        WHERE {t['key']} = %s AND model = %s AND source = 'daily'
                        ORDER BY forecast_date DESC
                        LIMIT 1
                    """, (t["key_value"], model))

                    row = cur.fetchone()

                    if row is None:
                        print(f"FAIL: No {target} {model} prediction found")
                        all_good = False
                        continue

                    forecast_date = row[0]
                    forecast_date_only = forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date

                    if forecast_date_only == expected_forecast_date:
                        print(f"PASS: {target} {model} forecast date is correct ({forecast_date_only})")
                    else:
                        print(f"FAIL: {target} {model} forecast date mismatch "
                              f"(expected: {expected_forecast_date}, got: {forecast_date_only})")
                        all_good = False

                return all_good
    except Exception as e:
        print(f"FAIL: Error checking {target} forecast date logic: {e}")
        return False

def check_api_health_endpoint():
    """Check that /health endpoint returns 200"""
    try:
        response = httpx.get(f'{API_BASE}/health', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: /health endpoint returns 200 (status: {data.get('status', 'unknown')})")
            return True
        else:
            print(f"FAIL: /health endpoint returns status {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Error checking /health endpoint: {e}")
        return False

def check_api_forecast_endpoint():
    """Check that /forecast endpoint returns 200"""
    try:
        response = httpx.get(f'{API_BASE}/forecast?model=lightgbm', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: /forecast?model=lightgbm returns 200 (forecast: {data.get('forecast_pm2_5', 'N/A')} PM2.5)")
        else:
            print(f"FAIL: /forecast?model=lightgbm returns status {response.status_code}")
            return False

        # Naive baseline too: a missing artifact 404s one model, not both.
        response = httpx.get(f'{API_BASE}/forecast?model=naive_baseline', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: /forecast?model=naive_baseline returns 200 (forecast: {data.get('forecast_pm2_5', 'N/A')} PM2.5)")
            return True
        else:
            print(f"FAIL: /forecast?model=naive_baseline returns status {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Error checking /forecast endpoint: {e}")
        return False

def check_api_leaderboard_endpoint():
    """Check that /leaderboard endpoint returns 200"""
    try:
        response = httpx.get(f'{API_BASE}/leaderboard', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            leaderboard_count = len(data.get('leaderboard', []))
            print(f"PASS: /leaderboard endpoint returns 200 ({leaderboard_count} models)")
            return True
        else:
            print(f"FAIL: /leaderboard endpoint returns status {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Error checking /leaderboard endpoint: {e}")
        return False

def check_api_evaluation_endpoint():
    """Check that /evaluation returns 200 with metrics nested under a provenance key.

    The status code is not enough: a deploy serving the old flat `mae` per model
    averages the launch backtest into the published record and still returns 200
    with the right number of models. Assert the `verified` / `backtest` nesting.
    """
    try:
        response = httpx.get(f'{API_BASE}/evaluation?days=7', timeout=10.0)
        if response.status_code != 200:
            print(f"FAIL: /evaluation endpoint returns status {response.status_code}")
            return False
        entries = response.json().get('evaluation', [])
        flat = [e['model'] for e in entries if 'mae' in e]
        if flat:
            print(f"FAIL: /evaluation still reports a combined mae for {flat} - "
                  "verified and backtest rows are being averaged together")
            return False
        print(f"PASS: /evaluation endpoint returns 200 ({len(entries)} models, "
              "metrics split by provenance)")
        return True
    except Exception as e:
        print(f"FAIL: Error checking /evaluation endpoint: {e}")
        return False

def check_api_predictions_endpoint():
    """Check that /predictions endpoint returns 200"""
    try:
        response = httpx.get(f'{API_BASE}/predictions?model=lightgbm&limit=5', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            prediction_count = len(data.get('predictions', []))
            print(f"PASS: /predictions endpoint returns 200 ({prediction_count} predictions)")
            return True
        else:
            print(f"FAIL: /predictions endpoint returns status {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Error checking /predictions endpoint: {e}")
        return False

def check_api_electricity_endpoints():
    """Check that all six /electricity/* endpoints return 200.

    Six paths, seven requests: /electricity/forecast is called once per published
    model, because a missing artifact or an unseeded model 404s on that model
    alone and a single lightgbm call would pass while seasonal_naive was broken.

    One function rather than six: they either all work or the router is broken.
    No DB-freshness check here - this file's allows PM25_STALE_LIMIT_DAYS, and the
    demand mirror normally runs 2-4 days behind, so it would fail a correct
    deployment every day. vericast/elec/diagnose.py owns elec freshness, at
    ELEC_STALE_LIMIT_DAYS.
    """
    endpoints = [
        ('/electricity/health', 'latest_observation'),
        ('/electricity/forecast?model=lightgbm', 'forecast_demand_mw'),
        ('/electricity/forecast?model=seasonal_naive', 'forecast_demand_mw'),
        ('/electricity/history?days=7', 'days_returned'),
        ('/electricity/leaderboard', 'leaderboard'),
        ('/electricity/evaluation', 'evaluation'),
        ('/electricity/predictions?model=lightgbm&limit=5', 'count'),
    ]
    try:
        for path, key in endpoints:
            response = httpx.get(f'{API_BASE}{path}', timeout=10.0)
            if response.status_code != 200:
                print(f"FAIL: {path} returns status {response.status_code}")
                return False
            value = response.json().get(key, 'N/A')
            if isinstance(value, list):
                value = f"{len(value)} models"
            print(f"PASS: {path} returns 200 ({key}: {value})")
        return True
    except Exception as e:
        print(f"FAIL: Error checking /electricity endpoints: {e}")
        return False

# Module level, not a local in main(), so tests/test_readiness_gate.py can assert
# the list is intact - every entry callable, every TARGETS key the parameterised
# checks read present - with no database and no running server.
#
# That test is the wiring half of this file's coverage. The checks themselves need
# a populated instance and a live API_BASE, which ci.yml has neither of, so they
# run in .github/workflows/readiness-gate.yml against the live database and the
# deployed API instead. Without the test, a renamed check or a dropped descriptor
# key surfaces as a traceback on the one run that is supposed to catch problems.
CHECKS = [
    ("DATABASE_URL exists", check_database_url),
    ("PostgreSQL connectivity", check_postgres_connectivity),
    # Interpolated, not spelled out: a hardcoded label drifts from the
    # threshold the check actually enforces.
    (f"Observations freshness (<={PM25_STALE_LIMIT_DAYS} days stale)",
     check_observations_freshness),
    ("Features match observations date", check_features_match_observations),
    ("Latest features have no NULLs (PM2.5)", check_features_no_nulls),
    # The elec store gaps on its own schedule - the mirror runs 2-4 days behind -
    # and a NULL there stops /electricity/forecast?model=lightgbm unnoticed.
    ("Latest features have no NULLs (electricity)",
     lambda: check_features_no_nulls("electricity")),
    ("Leakage test passes (PM2.5)", check_leakage_test),
    # The elec store has its own values to get wrong, and its own calendar
    # columns, which the PM2.5 twin has none of.
    ("Leakage test passes (electricity)",
     lambda: check_leakage_test('vericast.elec.leakage_test')),
    ("LightGBM model artifact exists (PM2.5)", check_model_artifact),
    # The elec artifact was unchecked: a truncated or missing
    # lightgbm_elec_model.txt passed go-live and only showed up as a 404 on
    # /electricity/forecast?model=lightgbm.
    ("LightGBM model artifact exists (electricity)",
     lambda: check_model_artifact("electricity")),
    ("LightGBM prediction exists (PM2.5)", check_prediction_exists),
    ("Naive baseline prediction exists (PM2.5)",
     lambda: check_prediction_exists("naive_baseline")),
    ("Forecast date logic, latest_obs + 1 (PM2.5)", check_forecast_date_logic),
    ("LightGBM prediction exists (electricity)",
     lambda: check_prediction_exists("lightgbm", "electricity")),
    ("Naive baseline prediction exists (electricity)",
     lambda: check_prediction_exists("naive_baseline", "electricity")),
    ("Forecast date logic, latest_obs + 1 (electricity)",
     lambda: check_forecast_date_logic("electricity")),
    ("API /health endpoint", check_api_health_endpoint),
    ("API /forecast endpoint", check_api_forecast_endpoint),
    ("API /leaderboard endpoint", check_api_leaderboard_endpoint),
    ("API /evaluation endpoint", check_api_evaluation_endpoint),
    ("API /predictions endpoint", check_api_predictions_endpoint),
    ("API /electricity/* endpoints", check_api_electricity_endpoints),
]


def main():
    """Run all deployment readiness checks"""
    print("Running VeriCast Deployment Readiness Check")
    print("=" * 60)

    checks = CHECKS

    results = []
    for name, check_func in checks:
        print(f"\n[{len(results)+1:2d}/{len(checks)}] {name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"FAIL: Unexpected error in check: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("DEPLOYMENT READINESS SUMMARY:")
    print("=" * 60)

    all_passed = True
    passed_count = 0

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:<10} {name}")
        if not passed:
            all_passed = False
        else:
            passed_count += 1

    print("-" * 60)
    print(f"Total: {passed_count}/{len(checks)} checks passed")

    if all_passed:
        print("\nALL CHECKS PASSED - SYSTEM IS READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("  1. Deploy to your chosen platform (Render, Fly.io, etc.)")
        print("  2. Set environment variables: DATABASE_URL (required), STATE, "
              "FRONTEND_ORIGIN. CITY must stay Nagpur - model_performance has no "
              "city column, so require_city_of_record refuses anything else at "
              "import, here and in app.py.")
        print("  3. Verify GitHub Actions workflow runs successfully")
        print("  4. Monitor the system for 24-48 hours before marking as production")
        return True
    else:
        print("\nSOME CHECKS FAILED - PLEASE FIX ISSUES BEFORE DEPLOYMENT")
        print("\nFailed checks:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)