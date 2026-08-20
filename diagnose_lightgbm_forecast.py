"""
Run this locally (where DATABASE_URL and lightgbm_model.txt are actually
reachable) to figure out why /forecast?model=lightgbm might be 404ing.

Usage: python diagnose_lightgbm_forecast.py
"""
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
CITY = "Nagpur"
MODEL_PATH = "lightgbm_model.txt"


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
                "engineer_features.py needs to run" if latest_feat != latest_obs else "",
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

    print("=" * 60)
    print("Overall:", "READY" if all_ok else "NEEDS ATTENTION (see FAILs above)")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    # Non-zero exit so the daily pipeline stops instead of publishing a
    # forecast this script just declared unfit.
    sys.exit(0 if main() else 1)