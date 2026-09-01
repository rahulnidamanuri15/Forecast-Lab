"""
Run this locally (where DATABASE_URL and the model artifact are actually
reachable) to figure out why /forecast?model=lightgbm might be 404ing.

Usage: python -m vericast.pm25.diagnose
"""
import os
import sys
from datetime import timedelta

import psycopg
from dotenv import load_dotenv

from vericast import (
    MODEL_PM25 as MODEL_PATH,
    PM25_MAX,
    PM25_MIN,
    PM25_STALE_LIMIT_DAYS,
    local_time,
)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

# Imported, not defined here, for the same reason as STALE_LIMIT_DAYS below:
# pm25/ingest.py gates the incoming *observation* on these bounds and this gates
# the outgoing *forecast*, and two copies could drift into a publish gate looser
# than the ingest one. Still the only check on the forecast's value - nothing in
# app.py or verify_deployment_readiness.py looks at it - so this stays the gate
# between a -40 ug/m3 prediction and the public record.
MIN_PM25, MAX_PM25 = PM25_MIN, PM25_MAX

SIGMA_LIMIT = 3.0   # forecast must sit within 3 sd of the trailing 30-day mean

# Imported, not defined here: /health and verify_deployment_readiness.py read the
# same number, and when this file owned it they drifted. See vericast/__init__.py.
STALE_LIMIT_DAYS = PM25_STALE_LIMIT_DAYS

# Both models vericast/pm25/predict.py publishes. Same role as
# vericast/elec/diagnose.py:36: predict.py warns and skips a model rather than
# raising, so without a completeness check here a day that published only the
# naive baseline exits 0 and the pipeline treats it as a full run.
EXPECTED_MODELS = {"naive_baseline", "lightgbm"}


def expected_models(latest_pm25):
    """The models predict.py would actually have published for this run.

    It skips naive_baseline when the latest observation carries a NULL pm2_5 - a
    thin-hours day under ingest.py's MIN_HOURS_PER_DAY - rather than publishing a
    NULL forecast, so demanding it here would turn a deliberate degradation into a
    red pipeline. lightgbm is NOT excused for NULL features: check 4 already fails
    on that, and one report of a problem is the right number.
    """
    return EXPECTED_MODELS - ({"naive_baseline"} if latest_pm25 is None else set())


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    print("=" * 60)
    print("LightGBM forecast diagnostic")
    print("=" * 60)

    all_ok = True

    # 1. Model file present at the path the script actually uses
    exists = os.path.exists(MODEL_PATH)
    all_ok &= check(
        f"Model file found at ./{MODEL_PATH}",
        exists,
        f"cwd={os.getcwd()}" if not exists else "",
    )

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # 2. Latest observation date, and its value: a thin-hours day is a
            #    present row with a NULL pm2_5 (ingest.py's MIN_HOURS_PER_DAY),
            #    which is exactly when predict.py skips naive_baseline.
            cur.execute(
                """
                SELECT as_of, pm2_5 FROM observations
                WHERE city = %s ORDER BY as_of DESC LIMIT 1
                """,
                (CITY,),
            )
            obs_row = cur.fetchone()
            latest_obs, latest_pm25 = obs_row if obs_row else (None, None)
            print(f"  Latest observation date: {latest_obs}")

            # 2b. Is the upstream source still moving? Ported from
            #     vericast/elec/diagnose.py:63-75. Everything below this point
            #     is an internal-consistency check anchored on latest_obs, so
            #     all of them pass while Open-Meteo is a week stale - the anchor
            #     just slides. This is the only check that looks at the clock.
            #     Early return rather than `all_ok &=`: with no observations at
            #     all, expected_date is None and the checks below compare
            #     against nothing.
            if latest_obs is None:
                check("Observations exist", False, f"no observations for {CITY}")
                print("=" * 60)
                return False

            stale_days = (local_time.today() - latest_obs).days
            all_ok &= check(
                f"Source freshness within {STALE_LIMIT_DAYS} days",
                stale_days <= STALE_LIMIT_DAYS,
                f"latest observation is {stale_days} days old; Open-Meteo has "
                f"stalled and forecast_date is sliding with it"
                if stale_days > STALE_LIMIT_DAYS
                else f"{stale_days} day(s) behind",
            )

            # 3. Latest features row date
            cur.execute(
                "SELECT MAX(as_of) FROM features WHERE city = %s", (CITY,)
            )
            latest_feat = cur.fetchone()[0]
            print(f"  Latest features date:    {latest_feat}")
            all_ok &= check(
                "features as_of matches observations as_of",
                latest_feat == latest_obs,
                f"features={latest_feat}, observations={latest_obs} "
                "vericast/pm25/features.py needs to run" if latest_feat != latest_obs else "",
            )

            # 4. Are there NULLs in the latest features row?
            cur.execute(
                """
                SELECT pm2_5_lag_1, pm10_lag_1, temperature_lag_1, wind_speed_lag_1,
                       precipitation_lag_1, pm2_5_roll_7, pm2_5_roll_30, pm10_roll_7,
                       pm10_roll_30, day_of_week, month, is_weekend,
                       temperature_2m_mean, wind_speed_10m_max, precipitation_sum
                FROM features WHERE city = %s ORDER BY as_of DESC LIMIT 1
                """,
                (CITY,),
            )
            row = cur.fetchone()
            has_nulls = row is None or any(v is None for v in row)
            all_ok &= check(
                "Latest features row has no NULLs",
                not has_nulls,
                str(row) if has_nulls else "",
            )

            # 5. Does a lightgbm prediction row actually exist?
            cur.execute(
                """
                SELECT forecast_date, predicted_pm2_5, created_at
                FROM predictions
                WHERE city = %s AND model = 'lightgbm'
                ORDER BY forecast_date DESC LIMIT 1
                """,
                (CITY,),
            )
            pred_row = cur.fetchone()
            all_ok &= check(
                "A lightgbm row exists in predictions",
                pred_row is not None,
                "no lightgbm rows found at all" if pred_row is None else str(pred_row),
            )
            # No staleness sub-check on MAX(forecast_date) here. `fdate < latest_obs`
            # passed at fdate == latest_obs - a forecast published for today rather
            # than tomorrow, which is exactly the failure it looked like it caught.
            # Check 6 below tests the real contract (forecast_date == latest_obs + 1)
            # against the *expected* date, so the weak version was redundant when it
            # was right and wrong when it disagreed. elec/diagnose.py never had one.

            # 6. Is every value published for that date physically plausible?
            #    Ported from vericast/elec/diagnose.py:91-117. Anchored to
            #    latest_obs + 1 day, not MAX(forecast_date): if predict.py failed
            #    today, MAX() silently falls back to yesterday's already-published
            #    row, which passes every check below and clears the pipeline.
            expected_date = latest_obs + timedelta(days=1)
            cur.execute(
                """
                SELECT model, predicted_pm2_5 FROM predictions
                WHERE city = %s AND forecast_date = %s AND source = 'daily'
                """,
                (CITY, expected_date),
            )
            published = dict(cur.fetchall())
            print(f"  Expected forecast_date:  {expected_date}")

            # Expect only what predict.py would actually have published; the rule
            # lives in expected_models() above so a test can reach it without a DB.
            expect = expected_models(latest_pm25)
            if "naive_baseline" not in expect:
                print(f"  naive_baseline not expected: {latest_obs} has NULL pm2_5 "
                      f"(too few hours ingested), so predict.py skipped it")

            missing = expect - published.keys()
            all_ok &= check(
                f"All {len(expect)} expected models published for {expected_date}",
                not missing,
                f"missing: {sorted(missing)}" if missing else
                ", ".join(f"{m}={published[m]:.1f}" for m in sorted(published)
                          if published[m] is not None),
            )

            out_of_range = {m: v for m, v in published.items()
                            if v is None or not (MIN_PM25 <= v <= MAX_PM25)}
            all_ok &= check(
                f"Forecasts within {MIN_PM25:,.0f}-{MAX_PM25:,.0f} ug/m3",
                not out_of_range,
                str(out_of_range) if out_of_range else str(published),
            )

            # 7. Trend sanity on the trailing 30 days of actuals rather than a
            #    fixed band, so it tightens as the series grows.
            cur.execute(
                """
                SELECT AVG(pm2_5), STDDEV_SAMP(pm2_5)
                FROM (SELECT pm2_5 FROM observations
                      WHERE city = %s AND pm2_5 IS NOT NULL
                      ORDER BY as_of DESC LIMIT 30) recent
                """,
                (CITY,),
            )
            mean_pm, sd_pm = cur.fetchone()

            if published.get("lightgbm") is None:
                all_ok &= check("LightGBM forecast within trend", False,
                                "no lightgbm row to check")
            elif not sd_pm:
                check("LightGBM forecast within trend", True,
                      "not enough history for a sd; skipped")
            else:
                sigma = abs(published["lightgbm"] - float(mean_pm)) / float(sd_pm)
                all_ok &= check(
                    f"LightGBM forecast within {SIGMA_LIMIT} sd of 30-day mean",
                    sigma <= SIGMA_LIMIT,
                    f"{sigma:.2f} sd from mean={float(mean_pm):.1f} "
                    f"(sd={float(sd_pm):.1f})",
                )

    print("=" * 60)
    print("Overall:", "READY" if all_ok else "NEEDS ATTENTION (see FAILs above)")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    # Non-zero exit so the daily pipeline stops instead of publishing a
    # forecast this script just declared unfit.
    sys.exit(0 if main() else 1)