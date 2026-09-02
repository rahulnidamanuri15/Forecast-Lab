"""Engineer the point-in-time PM2.5 feature store.

One idempotent INSERT ... SELECT. Postgres window frames carry the guarantees:

  * `RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING` is
    date-addressed, not row-addressed, so it returns NULL across a date gap on
    its own - no explicit gap guard to forget.
  * `COUNT(pm2_5) OVER w7 = 7` refuses to call an incomplete window a 7-day mean.
    It counts the averaged column, not `*`: a thin-hours day is stored as a row
    with a NULL value (ingest.py's MIN_HOURS_PER_DAY), which `AVG` skips, so
    `COUNT(*)` would pass a 6-value mean off as a 7-day one.
  * Look-ahead leakage is structurally unexpressible - a `RANGE ... PRECEDING`
    frame cannot reference a future row.

Full recompute every run, so changing a feature definition needs no backfill.
"""
import os
import psycopg
from dotenv import load_dotenv

from vericast import (
    alignment_sql,
    require_city_of_record,
    verify_alignment as check_alignment,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))

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


# Its own check, not covered by leakage_test.py: that one re-derives feature
# *values* and never asks whether each feature row has a next-day target to score
# against. Both queries live in vericast/__init__.py - the two targets differed
# only in table and key names.
GAP_DAYS_SQL, ORPHAN_ROWS_SQL = alignment_sql("features", "observations", "city")


def verify_alignment(cur):
    """This city's alignment check. See vericast.verify_alignment for the contract."""
    check_alignment(cur, GAP_DAYS_SQL, ORPHAN_ROWS_SQL, CITY)


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
