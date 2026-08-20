import os
import psycopg
import subprocess
import sys
from datetime import timedelta
from dotenv import load_dotenv
import httpx

import local_time

load_dotenv()

# Override to smoke-test a deployed instance instead of the local process.
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

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
    """Check that observations latest date <= 1 day stale"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping observations freshness check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", ("Nagpur",))
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print("FAIL: No observations found in database")
                    return False

                # Same "today" the pipeline uses (Asia/Kolkata), not UTC -
                # otherwise this gate disagrees with the scripts it is gating.
                today = local_time.today()
                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs

                days_stale = (today - latest_obs_date).days

                if days_stale <= 1:
                    print(f"PASS: Observations are fresh (latest: {latest_obs_date}, stale days: {days_stale})")
                    return True
                else:
                    print(f"FAIL: Observations are stale (latest: {latest_obs_date}, stale days: {days_stale})")
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
                # Get latest observation date
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", ("Nagpur",))
                latest_obs = cur.fetchone()[0]

                # Get latest features date
                cur.execute("SELECT MAX(as_of) FROM features WHERE city = %s", ("Nagpur",))
                latest_feat = cur.fetchone()[0]

                if latest_obs is None or latest_feat is None:
                    print("FAIL: Missing observations or features data")
                    return False

                # Compare dates
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

def check_features_no_nulls():
    """Check that latest features have no NULLs"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping features NULL check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Get the latest features record and check for NULLs
                cur.execute("""
                    SELECT * FROM features
                    WHERE city = %s
                    ORDER BY as_of DESC
                    LIMIT 1
                """, ("Nagpur",))
                row = cur.fetchone()

                if row is None:
                    print("FAIL: No features found")
                    return False

                # Get column names
                colnames = [desc[0] for desc in cur.description]

                # Check for NULLs (skip city and as_of as they're identifiers)
                null_cols = []
                for i, val in enumerate(row):
                    colname = colnames[i]
                    if colname not in ['city', 'as_of'] and val is None:
                        null_cols.append(colname)

                if not null_cols:
                    print("PASS: Latest features have no NULL values")
                    return True
                else:
                    print(f"FAIL: Latest features have NULL values in columns: {null_cols}")
                    return False
    except Exception as e:
        print(f"FAIL: Error checking features for NULLs: {e}")
        return False

def check_leakage_test():
    """Run the leakage test to ensure it passes"""
    print("Running leakage test:")
    try:
        result = subprocess.run([
            sys.executable, 'leakage_test.py'
        ], capture_output=True, text=True, cwd=os.getcwd(), timeout=180)

        if result.returncode == 0:
            print("PASS: Leakage test PASSED")
            # Print any output from the test for visibility
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

def check_model_artifact():
    """Check that LightGBM model artifact exists"""
    model_path = "lightgbm_model.txt"
    if os.path.exists(model_path):
        # Check if it's not empty
        if os.path.getsize(model_path) > 0:
            print("PASS: LightGBM model artifact exists and is not empty")
            return True
        else:
            print("FAIL: LightGBM model artifact exists but is empty")
            return False
    else:
        print(f"FAIL: LightGBM model artifact not found at {model_path}")
        return False

def check_lightgbm_prediction_exists():
    """Check that LightGBM prediction exists for tomorrow"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping LightGBM prediction check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Get latest observation date to determine what tomorrow should be
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", ("Nagpur",))
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print("FAIL: No observations found to determine forecast date")
                    return False

                # Calculate expected forecast date (tomorrow)
                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                expected_forecast_date = latest_obs_date + timedelta(days=1)

                # Check for LightGBM prediction for this date
                cur.execute("""
                    SELECT forecast_date, predicted_pm2_5
                    FROM predictions
                    WHERE city = %s AND model = %s AND forecast_date = %s
                """, ("Nagpur", "lightgbm", expected_forecast_date))

                row = cur.fetchone()

                if row is None:
                    print(f"FAIL: No LightGBM prediction found for forecast date {expected_forecast_date}")
                    return False

                forecast_date, predicted_value = row
                if predicted_value is None:
                    print(f"FAIL: LightGBM prediction exists for {forecast_date} but predicted_pm2_5 is NULL")
                    return False

                print(f"PASS: LightGBM prediction exists for {forecast_date} (PM2.5: {predicted_value:.2f})")
                return True
    except Exception as e:
        print(f"FAIL: Error checking LightGBM prediction: {e}")
        return False

def check_naive_prediction_exists():
    """Check that naive baseline prediction exists for tomorrow"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping naive prediction check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Get latest observation date to determine what tomorrow should be
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", ("Nagpur",))
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print("FAIL: No observations found to determine forecast date")
                    return False

                # Calculate expected forecast date (tomorrow)
                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                expected_forecast_date = latest_obs_date + timedelta(days=1)

                # Check for naive prediction for this date
                cur.execute("""
                    SELECT forecast_date, predicted_pm2_5
                    FROM predictions
                    WHERE city = %s AND model = %s AND forecast_date = %s
                """, ("Nagpur", "naive_baseline", expected_forecast_date))

                row = cur.fetchone()

                if row is None:
                    print(f"FAIL: No naive baseline prediction found for forecast date {expected_forecast_date}")
                    return False

                forecast_date, predicted_value = row
                if predicted_value is None:
                    print(f"FAIL: Naive baseline prediction exists for {forecast_date} but predicted_pm2_5 is NULL")
                    return False

                print(f"PASS: Naive baseline prediction exists for {forecast_date} (PM2.5: {predicted_value:.2f})")
                return True
    except Exception as e:
        print(f"FAIL: Error checking naive baseline prediction: {e}")
        return False

def check_forecast_date_logic():
    """Check that forecast_date == latest_observation + 1 for both models"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL not set, skipping forecast date logic check")
        return False

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Get latest observation date
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", ("Nagpur",))
                latest_obs = cur.fetchone()[0]

                if latest_obs is None:
                    print("FAIL: No observations found")
                    return False

                latest_obs_date = latest_obs.date() if hasattr(latest_obs, 'date') else latest_obs
                expected_forecast_date = latest_obs_date + timedelta(days=1)

                # Check both models
                models_to_check = ["lightgbm", "naive_baseline"]
                all_good = True

                for model in models_to_check:
                    cur.execute("""
                        SELECT forecast_date
                        FROM predictions
                        WHERE city = %s AND model = %s
                        ORDER BY forecast_date DESC
                        LIMIT 1
                    """, ("Nagpur", model))

                    row = cur.fetchone()

                    if row is None:
                        print(f"FAIL: No {model} prediction found")
                        all_good = False
                        continue

                    forecast_date = row[0]
                    forecast_date_only = forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date

                    if forecast_date_only == expected_forecast_date:
                        print(f"PASS: {model} forecast date is correct ({forecast_date_only})")
                    else:
                        print(f"FAIL: {model} forecast date mismatch (expected: {expected_forecast_date}, got: {forecast_date_only})")
                        all_good = False

                return all_good
    except Exception as e:
        print(f"FAIL: Error checking forecast date logic: {e}")
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
        # Test lightgbm model
        response = httpx.get(f'{API_BASE}/forecast?model=lightgbm', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print(f"PASS: /forecast?model=lightgbm returns 200 (forecast: {data.get('forecast_pm2_5', 'N/A')} PM2.5)")
        else:
            print(f"FAIL: /forecast?model=lightgbm returns status {response.status_code}")
            return False

        # Test naive_baseline model
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
    """Check that /evaluation endpoint returns 200"""
    try:
        response = httpx.get(f'{API_BASE}/evaluation?days=7', timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            evaluation_count = len(data.get('evaluation', []))
            print(f"PASS: /evaluation endpoint returns 200 ({evaluation_count} models)")
            return True
        else:
            print(f"FAIL: /evaluation endpoint returns status {response.status_code}")
            return False
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

def main():
    """Run all deployment readiness checks"""
    print("Running VeriCast Deployment Readiness Check")
    print("=" * 60)

    checks = [
        ("DATABASE_URL exists", check_database_url),
        ("PostgreSQL connectivity", check_postgres_connectivity),
        ("Observations freshness (<=1 day stale)", check_observations_freshness),
        ("Features match observations date", check_features_match_observations),
        ("Latest features have no NULLs", check_features_no_nulls),
        ("Leakage test passes", check_leakage_test),
        ("LightGBM model artifact exists", check_model_artifact),
        ("LightGBM prediction exists", check_lightgbm_prediction_exists),
        ("Naive baseline prediction exists", check_naive_prediction_exists),
        ("Forecast date logic (latest_obs + 1)", check_forecast_date_logic),
        ("API /health endpoint", check_api_health_endpoint),
        ("API /forecast endpoint", check_api_forecast_endpoint),
        ("API /leaderboard endpoint", check_api_leaderboard_endpoint),
        ("API /evaluation endpoint", check_api_evaluation_endpoint),
        ("API /predictions endpoint", check_api_predictions_endpoint),
    ]

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
        print("  2. Set environment variables: DATABASE_URL, CITY, FRONTEND_ORIGIN")
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