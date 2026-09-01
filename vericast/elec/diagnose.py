"""Quality gate for the published electricity forecast.

Runs last in the daily job; a non-zero exit stops the pipeline rather than
letting an unfit forecast stand.

Thresholds are NOT copied from vericast/pm25/diagnose.py. The demand mirror
lags real time by 2-4 days, so PM2.5's "must be current" freshness check would
fail here every single day and block a forecast that is in fact correct for the
data available. What this checks instead is internal consistency: the forecast is
labelled one day after the newest observation, sits in a physically plausible
range, and is not wildly off the recent trend.

Usage: python -m vericast.elec.diagnose
"""
import os
import sys
from datetime import timedelta

import psycopg
from dotenv import load_dotenv

from vericast import (
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    ELEC_STALE_LIMIT_DAYS,
    MODEL_ELEC as MODEL_PATH,
    local_time,
)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

# Imported, not re-stated: elec/ingest.py rejects an implausible *observation* and
# this rejects an implausible *forecast* against the same claim about the target.
# When this file owned a second copy of the numbers, tests/test_mirror_guards.py
# was pinning only the ingest pair - so a change here could have loosened the
# publish gate with the suite still green. Same fix as STALE_LIMIT_DAYS below.
MIN_MW, MAX_MW = ELEC_MIN_MW, ELEC_MAX_MW

STALE_LIMIT_DAYS = ELEC_STALE_LIMIT_DAYS  # defined in vericast/__init__.py
SIGMA_LIMIT = 3.0      # forecast must sit within 3 sd of the trailing 30-day mean

EXPECTED_MODELS = {"naive_baseline", "seasonal_naive", "lightgbm"}


def expected_models(features_current, demand_lag_6):
    """The models predict.py would actually have published for this run.

    It skips seasonal_naive when demand_lag_6 is NULL - a date gap inside the last
    week, which is the mirror's problem and not this run's - rather than publishing
    a NULL forecast, so demanding it here would turn a deliberate degradation into
    a red pipeline. lightgbm is NOT excused for a stale or missing features row:
    the as_of check already fails on that, and one report of a problem is the right
    number. Same split as vericast/pm25/diagnose.py.
    """
    excused = {"seasonal_naive"} if features_current and demand_lag_6 is None else set()
    return EXPECTED_MODELS - excused


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    print("=" * 62)
    print("Electricity forecast diagnostic (Maharashtra peak demand)")
    print("=" * 62)

    all_ok = True

    exists = os.path.exists(MODEL_PATH)
    all_ok &= check(f"Model file found at ./{MODEL_PATH}", exists,
                    f"cwd={os.getcwd()}" if not exists else "")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(as_of) FROM electricity_observations WHERE state = %s",
                        (STATE,))
            latest_obs = cur.fetchone()[0]
            print(f"  Latest observation date: {latest_obs}")

            if latest_obs is None:
                check("Observations exist", False, "electricity_observations is empty")
                print("=" * 62)
                return False

            stale_days = (local_time.today() - latest_obs).days
            all_ok &= check(
                f"Source freshness within {STALE_LIMIT_DAYS} days",
                stale_days <= STALE_LIMIT_DAYS,
                f"latest observation is {stale_days} days old; the mirror has stalled"
                if stale_days > STALE_LIMIT_DAYS
                else f"{stale_days} days behind (normal for this source)",
            )

            # demand_lag_6 comes back alongside as_of because it is the one column
            # predict.py checks on its own: a date gap inside the last week leaves
            # it NULL and seasonal_naive is skipped while the other two publish.
            cur.execute("""
                SELECT as_of, demand_lag_6 FROM electricity_features
                WHERE state = %s ORDER BY as_of DESC LIMIT 1
            """, (STATE,))
            feat_row = cur.fetchone()
            latest_feat, demand_lag_6 = feat_row if feat_row else (None, None)
            print(f"  Latest features date:    {latest_feat}")
            all_ok &= check(
                "features as_of matches observations as_of",
                latest_feat == latest_obs,
                f"features={latest_feat}, observations={latest_obs}; "
                "vericast/elec/features.py needs to run"
                if latest_feat != latest_obs else "",
            )

            # Every model should have published for the day after the newest
            # observation - that is what vericast/elec/predict.py labels it.
            # source = 'daily' so a launch-backtest row seeded on that date, which
            # was written with the actual already in hand, can't satisfy the gate.
            expected_date = latest_obs + timedelta(days=1)
            cur.execute("""
                SELECT model, predicted_demand_mw
                FROM electricity_predictions
                WHERE state = %s AND forecast_date = %s AND source = 'daily'
            """, (STATE, expected_date))
            published = dict(cur.fetchall())
            print(f"  Expected forecast_date:  {expected_date}")

            # Expect only what predict.py would actually have published; the rule
            # lives in expected_models() above so a test can reach it without a DB.
            expect = expected_models(latest_feat == latest_obs, demand_lag_6)
            if "seasonal_naive" not in expect:
                print(f"  seasonal_naive not expected: demand_lag_6 is NULL at "
                      f"{latest_feat} (date gap in the last week), so predict.py "
                      f"skipped it")

            missing = expect - published.keys()
            all_ok &= check(
                f"All {len(expect)} expected models published for {expected_date}",
                not missing,
                f"missing: {sorted(missing)}" if missing else
                # `is not None` for the same reason as the `v is None or` below,
                # and matching vericast/pm25/diagnose.py: a NULL forecast has to be
                # reported by the out-of-range check underneath, not crash this
                # format string on the way there.
                ", ".join(f"{m}={published[m]:.0f} MW" for m in sorted(published)
                          if published[m] is not None),
            )

            # `v is None or` matches vericast/pm25/diagnose.py: predicted_demand_mw
            # is NOT NULL today, but the comparison must not raise if that ever
            # changes - the gate has to report the bad forecast, not crash on it.
            out_of_range = {m: v for m, v in published.items()
                            if v is None or not (MIN_MW <= v <= MAX_MW)}
            all_ok &= check(
                f"Forecasts within {MIN_MW:,.0f}-{MAX_MW:,.0f} MW",
                not out_of_range,
                str(out_of_range) if out_of_range else "",
            )

            # Trend sanity, on the trailing 30 days of actuals rather than a
            # fixed band, so it tightens as the series grows.
            cur.execute("""
                SELECT AVG(peak_demand_mw), STDDEV_SAMP(peak_demand_mw)
                FROM (SELECT peak_demand_mw FROM electricity_observations
                      WHERE state = %s ORDER BY as_of DESC LIMIT 30) recent
            """, (STATE,))
            mean_mw, sd_mw = cur.fetchone()

            if published.get("lightgbm") is None:
                all_ok &= check("LightGBM forecast within trend", False,
                                "no lightgbm row to check")
            elif not sd_mw:
                check("LightGBM forecast within trend", True,
                      "not enough history for a sd; skipped")
            else:
                sigma = abs(published["lightgbm"] - float(mean_mw)) / float(sd_mw)
                all_ok &= check(
                    f"LightGBM forecast within {SIGMA_LIMIT} sd of 30-day mean",
                    sigma <= SIGMA_LIMIT,
                    f"{sigma:.2f} sd from mean={float(mean_mw):.0f} MW "
                    f"(sd={float(sd_mw):.0f} MW)",
                )

    print("=" * 62)
    print("Overall:", "READY" if all_ok else "NEEDS ATTENTION (see FAILs above)")
    print("=" * 62)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
