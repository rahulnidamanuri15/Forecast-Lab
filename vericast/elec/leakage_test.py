"""Verify the electricity feature store is point-in-time correct.

The twin of vericast/pm25/leakage_test.py, and it exists for the same reason that
one does: the window frames in features.py make look-ahead leakage structurally
unexpressible, but "the SQL cannot express leakage" is a claim about the SQL as
written, and nothing on the daily path re-read the stored rows to confirm the SQL
in the file is the SQL that produced them. verify_alignment() checks the
features(t) -> target(t+1) *join*; this checks the *values*.

Checks four invariants against `electricity_observations`, deriving expectations
from *calendar dates* rather than row positions:

  1. same-day columns equal that day's observation exactly, and
     cooling_degree_days is GREATEST(0, temp - COOLING_BASE) of it;
  2. calendar columns (day_of_week, month, is_weekend) match the date itself -
     the one place an off-by-one is invisible in every other check;
  3. `demand_lag_1` / `_2` / `_6` and `temp_lag_1` equal that many calendar days
     back, and are NULL when that day is missing (Maharashtra's
     2025-05-21 -> 05-24 gap) or the row is early in the series. lag_6 not lag_7:
     features at as_of = t predict t+1, so same-weekday-last-week is y(t-6);
  4. the rolling columns are NULL unless every one of the preceding 6 / 29
     calendar days is present *with a value*, and equal the mean (or max) over
     that full window when they are. Counting the averaged column rather than `*`
     is what temp_roll_7 needs: temperature_2m_mean is nullable and AVG skips
     NULLs, so a 6-value mean must not be labelled a 7-day one.

Usage: python -m vericast.elec.leakage_test
"""
import os
from datetime import timedelta

import psycopg
from dotenv import load_dotenv

from vericast.elec.features import COOLING_BASE

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STATE = os.getenv("STATE", "Maharashtra")

TOLERANCE = 1e-9

# (feature column, observation column, window length in days, aggregate)
ROLLING = [
    ("demand_roll_7_mean", "peak_demand_mw", 7, "mean"),
    ("demand_roll_7_max", "peak_demand_mw", 7, "max"),
    ("demand_roll_30_mean", "peak_demand_mw", 30, "mean"),
    ("temp_roll_7", "temperature_2m_mean", 7, "mean"),
]

# (feature column, observation column, days back)
LAGS = [
    ("demand_lag_1", "peak_demand_mw", 1),
    ("demand_lag_2", "peak_demand_mw", 2),
    ("demand_lag_6", "peak_demand_mw", 6),
    ("temp_lag_1", "temperature_2m_mean", 1),
]

SAME_DAY = ["temperature_2m_mean", "temperature_2m_max"]

OBS_COLS = ["peak_demand_mw", "energy_met_mu",
            "temperature_2m_mean", "temperature_2m_max"]
FEAT_COLS = ([c for c, _, _ in LAGS] + [c for c, _, _, _ in ROLLING] + SAME_DAY
             + ["cooling_degree_days", "day_of_week", "month", "is_weekend"])


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
                f"SELECT as_of, {', '.join(OBS_COLS)} FROM electricity_observations "
                f"WHERE state = %s ORDER BY as_of", (STATE,))
            obs = {r[0]: dict(zip(OBS_COLS, r[1:])) for r in cur.fetchall()}
            print(f"Fetched {len(obs)} observation records")

            cur.execute(
                f"SELECT as_of, {', '.join(FEAT_COLS)} FROM electricity_features "
                f"WHERE state = %s ORDER BY as_of", (STATE,))
            feats = [(r[0], dict(zip(FEAT_COLS, r[1:]))) for r in cur.fetchall()]
            print(f"Fetched {len(feats)} feature records")

    errors = []
    for as_of, feat in feats:
        today = obs.get(as_of)
        if today is None:
            errors.append(f"{as_of}: missing in electricity_observations")
            continue

        # 1. Same-day columns are copied straight through, and the one derived
        #    same-day column is derived from the observation, not from itself.
        for col in SAME_DAY:
            if mismatch(feat[col], today[col]):
                errors.append(f"{as_of}: {col} {feat[col]} != observation {today[col]}")

        temp = today["temperature_2m_mean"]
        cdd = None if temp is None else max(0.0, temp - COOLING_BASE)
        if mismatch(feat["cooling_degree_days"], cdd):
            errors.append(f"{as_of}: cooling_degree_days is "
                          f"{feat['cooling_degree_days']}, expected {cdd}")

        # 2. Calendar columns. ISODOW is Mon=1, the column is Mon=0, and weekend
        #    is Sat/Sun - an off-by-one here shifts every weekday the model sees
        #    and shows up nowhere else.
        if feat["day_of_week"] != as_of.weekday():
            errors.append(f"{as_of}: day_of_week is {feat['day_of_week']}, "
                          f"expected {as_of.weekday()}")
        if feat["month"] != as_of.month:
            errors.append(f"{as_of}: month is {feat['month']}, expected {as_of.month}")
        if bool(feat["is_weekend"]) != (as_of.weekday() >= 5):
            errors.append(f"{as_of}: is_weekend is {feat['is_weekend']}, "
                          f"expected {as_of.weekday() >= 5}")

        # 3. Lags come from that many *calendar* days back, or are NULL if the
        #    day is absent.
        for fcol, ocol, back in LAGS:
            prev = obs.get(as_of - timedelta(days=back))
            expected = None if prev is None else prev[ocol]
            if mismatch(feat[fcol], expected):
                errors.append(
                    f"{as_of}: {fcol} is {feat[fcol]}, expected {expected}"
                    + ("" if prev else f" (NULL - day -{back} absent)"))

        # 4. Rolling windows need every day of the window with a value, else NULL.
        for fcol, ocol, days, agg in ROLLING:
            window = [obs.get(as_of - timedelta(days=d)) for d in range(days)]
            values = [w[ocol] for w in window if w is not None and w[ocol] is not None]
            complete = len(values) == days
            if not complete:
                expected = None
            else:
                expected = sum(values) / days if agg == "mean" else max(values)
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
