"""Verify the PM2.5 feature store is point-in-time correct.

Checks three invariants against `observations`, deriving expectations from
*calendar dates* rather than row positions:

  1. same-day weather columns equal that day's observation exactly;
  2. `*_lag_1` equals the previous calendar day, and is NULL when that day is
     missing (a date gap) or when the row is the first in the series;
  3. `*_roll_7` / `*_roll_30` are NULL unless every one of the preceding 6 / 29
     calendar days is present, and equal the mean over that full window when it is.

Check 3 used to re-implement features.py's old `if i >= 6:` row-index logic,
which meant it ratified the bug it was supposed to catch. It now asserts the
full-window property that vericast/pm25/features.py's `COUNT(*) OVER wN`
guarantees, so the two disagree if either drifts.
"""
import os
import psycopg
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

TOLERANCE = 1e-9

# (feature column, observation column, window length in days)
ROLLING = [
    ("pm2_5_roll_7", "pm2_5", 7),
    ("pm2_5_roll_30", "pm2_5", 30),
    ("pm10_roll_7", "pm10", 7),
    ("pm10_roll_30", "pm10", 30),
]

# (feature column, observation column)
LAGS = [
    ("pm2_5_lag_1", "pm2_5"),
    ("pm10_lag_1", "pm10"),
    ("temperature_lag_1", "temperature_2m_mean"),
    ("wind_speed_lag_1", "wind_speed_10m_max"),
    ("precipitation_lag_1", "precipitation_sum"),
]

SAME_DAY = ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"]

OBS_COLS = ["pm2_5", "pm10", "temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"]
FEAT_COLS = [c for c, _ in LAGS] + [c for c, _, _ in ROLLING] + SAME_DAY


def mismatch(a, b, tol=TOLERANCE):
    """None-safe comparison. Both None -> match; one None -> mismatch."""
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(a - b) > tol


def run_leakage_test():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT as_of, {', '.join(OBS_COLS)} FROM observations "
                f"WHERE city = %s ORDER BY as_of", (CITY,))
            obs = {r[0]: dict(zip(OBS_COLS, r[1:])) for r in cur.fetchall()}
            print(f"Fetched {len(obs)} observation records")

            cur.execute(
                f"SELECT as_of, {', '.join(FEAT_COLS)} FROM features "
                f"WHERE city = %s ORDER BY as_of", (CITY,))
            feats = [(r[0], dict(zip(FEAT_COLS, r[1:]))) for r in cur.fetchall()]
            print(f"Fetched {len(feats)} feature records")

    errors = []
    for as_of, feat in feats:
        today = obs.get(as_of)
        if today is None:
            errors.append(f"{as_of}: missing in observations")
            continue

        # 1. Same-day weather is copied straight through.
        for col in SAME_DAY:
            if mismatch(feat[col], today[col]):
                errors.append(f"{as_of}: {col} {feat[col]} != observation {today[col]}")

        # 2. Lags come from the previous *calendar* day, or are NULL if it is absent.
        prev = obs.get(as_of - timedelta(days=1))
        for fcol, ocol in LAGS:
            expected = None if prev is None else prev[ocol]
            if mismatch(feat[fcol], expected):
                errors.append(
                    f"{as_of}: {fcol} is {feat[fcol]}, expected {expected}"
                    + ("" if prev else " (NULL - previous day absent)"))

        # 3. Rolling means need every day of the window, else NULL.
        for fcol, ocol, days in ROLLING:
            window = [obs.get(as_of - timedelta(days=d)) for d in range(days)]
            values = [w[ocol] for w in window if w is not None and w[ocol] is not None]
            complete = len(values) == days
            expected = sum(values) / days if complete else None
            if mismatch(feat[fcol], expected):
                errors.append(
                    f"{as_of}: {fcol} is {feat[fcol]}, expected {expected} "
                    f"({len(values)}/{days} days present)")

    if errors:
        print("Leakage test FAILED")
        print(f"Found {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    print("Leakage test PASSED")
    print(f"All {len(feats)} feature records checked successfully.")
    return True


if __name__ == "__main__":
    exit(0 if run_leakage_test() else 1)
