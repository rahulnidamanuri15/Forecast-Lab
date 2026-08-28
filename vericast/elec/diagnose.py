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

from vericast import MODEL_ELEC as MODEL_PATH, local_time

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

# Observed 2023-2026 Maharashtra range is 20,147-32,419 MW. Bounds are wide
# enough for growth and a mild winter, tight enough that a unit error or a
# mis-parsed column fails instead of publishing.
MIN_MW, MAX_MW = 15_000.0, 40_000.0

STALE_LIMIT_DAYS = 5   # mirror normally lags 2-4 days; past this it has stalled
SIGMA_LIMIT = 3.0      # forecast must sit within 3 sd of the trailing 30-day mean

EXPECTED_MODELS = {"naive_baseline", "seasonal_naive", "lightgbm"}


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

            cur.execute("SELECT MAX(as_of) FROM electricity_features WHERE state = %s",
                        (STATE,))
            latest_feat = cur.fetchone()[0]
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
            expected_date = latest_obs + timedelta(days=1)
            cur.execute("""
                SELECT model, predicted_demand_mw
                FROM electricity_predictions
                WHERE state = %s AND forecast_date = %s
            """, (STATE, expected_date))
            published = dict(cur.fetchall())
            print(f"  Expected forecast_date:  {expected_date}")

            missing = EXPECTED_MODELS - published.keys()
            all_ok &= check(
                f"All {len(EXPECTED_MODELS)} models published for {expected_date}",
                not missing,
                f"missing: {sorted(missing)}" if missing else
                ", ".join(f"{m}={published[m]:.0f} MW" for m in sorted(published)),
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
                check("LightGBM forecast within trend", False,
                      "no lightgbm row to check")
                all_ok = False
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
