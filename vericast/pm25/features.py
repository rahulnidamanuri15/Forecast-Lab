"""Engineer the point-in-time PM2.5 feature store.

One idempotent INSERT ... SELECT, ported from vericast/elec/features.py, which
replaced the Python row-loop this file used to be. Postgres window frames do the
same job with stronger guarantees:

  * `RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING` is
    date-addressed, not row-addressed, so it returns NULL across a date gap on
    its own - no explicit gap guard to forget.
  * `COUNT(pm2_5) OVER w7 = 7` is stricter than the old row-index check: `if i >= 6`
    means "seven rows exist", not "seven consecutive days exist", so the old
    version averaged 7 rows whether or not they spanned 7 calendar days and
    still called the result a 7-day mean. This refuses instead. It counts the
    averaged column, not `*`: a thin-hours day is stored as a row with a NULL
    value (ingest.py's MIN_HOURS_PER_DAY), which `AVG` skips - so `COUNT(*)`
    would pass a 6-value mean off as a 7-day one, and disagree with
    leakage_test.py's `len(values) == days`.
  * Look-ahead leakage is structurally unexpressible - a `RANGE ... PRECEDING`
    frame cannot reference a future row.

Full recompute every run, so changing a feature definition needs no backfill.
"""
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CITY = os.getenv("CITY", "Nagpur")

# is_weekend is BOOLEAN here (see vericast/schema.py:55), unlike
# electricity_features.is_weekend which is INT - so no ::int cast.
ENGINEER_SQL = """
INSERT INTO features (
    city, as_of,
    pm2_5_lag_1, pm10_lag_1, temperature_lag_1, wind_speed_lag_1, precipitation_lag_1,
    pm2_5_roll_7, pm2_5_roll_30, pm10_roll_7, pm10_roll_30,
    day_of_week, month, is_weekend,
    temperature_2m_mean, wind_speed_10m_max, precipitation_sum
)
SELECT
    city,
    as_of,

    MAX(pm2_5) OVER lag1,
    MAX(pm10) OVER lag1,
    MAX(temperature_2m_mean) OVER lag1,
    MAX(wind_speed_10m_max) OVER lag1,
    MAX(precipitation_sum) OVER lag1,

    CASE WHEN COUNT(pm2_5) OVER w7  = 7  THEN AVG(pm2_5) OVER w7  END,
    CASE WHEN COUNT(pm2_5) OVER w30 = 30 THEN AVG(pm2_5) OVER w30 END,
    CASE WHEN COUNT(pm10)  OVER w7  = 7  THEN AVG(pm10)  OVER w7  END,
    CASE WHEN COUNT(pm10)  OVER w30 = 30 THEN AVG(pm10)  OVER w30 END,

    EXTRACT(ISODOW FROM as_of)::int - 1,
    EXTRACT(MONTH  FROM as_of)::int,
    (EXTRACT(ISODOW FROM as_of) >= 6),

    temperature_2m_mean,
    wind_speed_10m_max,
    precipitation_sum

FROM observations
WHERE city = %s
WINDOW lag1 AS (ORDER BY as_of
            RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING),
       w7   AS (ORDER BY as_of RANGE BETWEEN INTERVAL '6 days'  PRECEDING AND CURRENT ROW),
       w30  AS (ORDER BY as_of RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW)
ON CONFLICT (city, as_of) DO UPDATE SET
    pm2_5_lag_1         = EXCLUDED.pm2_5_lag_1,
    pm10_lag_1          = EXCLUDED.pm10_lag_1,
    temperature_lag_1   = EXCLUDED.temperature_lag_1,
    wind_speed_lag_1    = EXCLUDED.wind_speed_lag_1,
    precipitation_lag_1 = EXCLUDED.precipitation_lag_1,
    pm2_5_roll_7        = EXCLUDED.pm2_5_roll_7,
    pm2_5_roll_30       = EXCLUDED.pm2_5_roll_30,
    pm10_roll_7         = EXCLUDED.pm10_roll_7,
    pm10_roll_30        = EXCLUDED.pm10_roll_30,
    day_of_week         = EXCLUDED.day_of_week,
    month               = EXCLUDED.month,
    is_weekend          = EXCLUDED.is_weekend,
    temperature_2m_mean = EXCLUDED.temperature_2m_mean,
    wind_speed_10m_max  = EXCLUDED.wind_speed_10m_max,
    precipitation_sum   = EXCLUDED.precipitation_sum,
    created_at          = CURRENT_TIMESTAMP;
"""


# Ported from vericast/elec/features.py so both feature stores check the same
# contract on the daily path. leakage_test.py (step 3 of daily-pipeline.yml) is not
# a substitute: it checks feature *values* - lag_1 equals the previous calendar day,
# roll_7/30 NULL unless the window is complete - and never asks whether each
# feature row has a next-day target to be scored against.
#
# Row existence, not `pm2_5 IS NOT NULL`: a low-coverage day is stored as a row
# with a NULL value (ingest.py's MIN_HOURS_PER_DAY), and train.py filters those.
# What is broken-join territory is a *missing row*, which is what these count.
ORPHAN_ROWS_SQL = """
SELECT COUNT(*)
FROM features f
WHERE f.city = %s
  AND f.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM observations WHERE city = f.city)
  AND NOT EXISTS (
      SELECT 1 FROM observations o
      WHERE o.city = f.city
        AND o.as_of = f.as_of + INTERVAL '1 day'
  )
"""

GAP_DAYS_SQL = """
SELECT COUNT(*)
FROM observations o
WHERE o.city = %s
  AND o.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM observations WHERE city = o.city)
  AND NOT EXISTS (
      SELECT 1 FROM observations n
      WHERE n.city = o.city
        AND n.as_of = o.as_of + INTERVAL '1 day'
  )
"""


def verify_alignment(cur):
    """Enforce the features(t) -> target(t+1) contract, gaps accounted for.

    Equality against the observed gap count rather than a tolerance, for the
    reason vericast/elec/features.py gives: a hardcoded "<= 1" pre-forgives the
    next gap, and forgives a genuinely broken join just as quietly. Raised rather
    than asserted for the reason given there too - python -O erases `assert`.
    """
    cur.execute(GAP_DAYS_SQL, (CITY,))
    gaps = cur.fetchone()[0]
    cur.execute(ORPHAN_ROWS_SQL, (CITY,))
    orphans = cur.fetchone()[0]

    if orphans != gaps:
        raise AssertionError(
            f"{orphans} feature rows have no next-day target but only {gaps} "
            f"observation gap(s) explain it - the features(t) -> target(t+1) "
            f"contract is broken"
        )
    print(f"  Alignment OK: {orphans} orphan row(s), all explained by "
          f"{gaps} observation gap(s)")


def engineer_features():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(ENGINEER_SQL, (CITY,))
                written = cur.rowcount
                conn.commit()
                print(f"Upserted {written} feature rows for {CITY}")

                cur.execute("""
                    SELECT COUNT(*),
                           COUNT(*) FILTER (WHERE pm2_5_lag_1 IS NULL),
                           COUNT(*) FILTER (WHERE pm2_5_roll_7 IS NULL),
                           COUNT(*) FILTER (WHERE pm2_5_roll_30 IS NULL),
                           MIN(as_of), MAX(as_of)
                    FROM features WHERE city = %s
                """, (CITY,))
                total, no_lag1, no_roll7, no_roll30, first, last = cur.fetchone()
                print(f"Total {CITY} feature rows: {total} ({first} -> {last})")
                # Expected NULLs: warm-up at the series start, plus the day after
                # each date gap. Anything beyond that is a data problem.
                print(f"  NULL pm2_5_lag_1: {no_lag1} (series start + day after each gap)")
                print(f"  NULL pm2_5_roll_7: {no_roll7} (first 6 days + gap-spanning windows)")
                print(f"  NULL pm2_5_roll_30: {no_roll30} (first 29 days + gap-spanning windows)")

                verify_alignment(cur)
    except Exception as e:
        print(f"Error engineering features: {e}")
        raise


if __name__ == "__main__":
    engineer_features()
