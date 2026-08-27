"""
Run this locally (where DATABASE_URL and the model artifact are actually
reachable) to figure out why /forecast?model=lightgbm might be 404ing.

Usage: python -m vericast.pm25.diagnose
"""
import os
import sys

import psycopg
from dotenv import load_dotenv

from vericast import MODEL_PM25 as MODEL_PATH

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

# Nagpur's observed 2023-2026 daily CAMS range is roughly 4-160 ug/m3. Bounds are
# wide enough for a bad Diwali week, tight enough that a unit error or a sign flip
# fails instead of publishing. Nothing anywhere else in the stack looks at the
# forecast's *value* - not app.py, not verify_deployment_readiness.py - so this is
# the only gate between a -40 ug/m3 prediction and the public record.
MIN_PM25, MAX_PM25 = 1.0, 500.0

SIGMA_LIMIT = 3.0   # forecast must sit within 3 sd of the trailing 30-day mean


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
            # 2. Latest observation date
            cur.execute(
                "SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,)
            )
            latest_obs = cur.fetchone()[0]
            print(f"  Latest observation date: {latest_obs}")

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
            if pred_row:
                fdate = pred_row[0]
                stale = (latest_obs and (fdate < latest_obs)) if fdate else True
                check(
                    "Latest lightgbm forecast_date is current",
                    not stale,
                    f"forecast_date={fdate}, latest_obs={latest_obs}" if stale else "",
                )

            # 6. Is every value published for that date physically plausible?
            #    Ported from vericast/elec/diagnose.py:108-114.
            cur.execute(
                """
                SELECT model, predicted_pm2_5 FROM predictions
                WHERE city = %s AND forecast_date = (
                    SELECT MAX(forecast_date) FROM predictions WHERE city = %s)
                """,
                (CITY, CITY),
            )
            published = {m: v for m, v in cur.fetchall()}
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