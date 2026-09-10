"""Engineer point-in-time electricity feature store in PostgreSQL.

Idempotent INSERT ... SELECT using window frames with strict point-in-time guarantees:
  - RANGE BETWEEN INTERVAL '1 day' PRECEDING: date-addressed to null across gaps.
  - COUNT(col) OVER wN = N: ensures full calendar window coverage before taking averages.
  - No lookahead: all rolling/lag frames strictly precede the target date.
"""
import os
import psycopg
from dotenv import load_dotenv

from vericast import (
    acquire_pipeline_lock,
    alignment_sql,
    require_database_url,
    verify_alignment as check_alignment,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = os.getenv("STATE", "Maharashtra")

COOLING_BASE = 24.0  # degC above which air-conditioning load kicks in

# demand_lag_6, not lag_7: features at as_of = t predict t+1, so the
# same-weekday-last-week value for the target is y(t-6).
ENGINEER_SQL = f"""
INSERT INTO electricity_features (
    state, as_of,
    demand_lag_1, demand_lag_2, demand_lag_6,
    demand_roll_7_mean, demand_roll_7_max, demand_roll_30_mean,
    temp_lag_1, temp_roll_7, cooling_degree_days,
    day_of_week, month, is_weekend,
    temperature_2m_mean, temperature_2m_max
)
SELECT
    state,
    as_of,

    MAX(peak_demand_mw) OVER (ORDER BY as_of
        RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING),
    MAX(peak_demand_mw) OVER (ORDER BY as_of
        RANGE BETWEEN INTERVAL '2 days' PRECEDING AND INTERVAL '2 days' PRECEDING),
    MAX(peak_demand_mw) OVER (ORDER BY as_of
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND INTERVAL '6 days' PRECEDING),

    CASE WHEN COUNT(peak_demand_mw) OVER w7  = 7  THEN AVG(peak_demand_mw) OVER w7  END,
    CASE WHEN COUNT(peak_demand_mw) OVER w7  = 7  THEN MAX(peak_demand_mw) OVER w7  END,
    CASE WHEN COUNT(peak_demand_mw) OVER w30 = 30 THEN AVG(peak_demand_mw) OVER w30 END,

    MAX(temperature_2m_mean) OVER (ORDER BY as_of
        RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING),
    CASE WHEN COUNT(temperature_2m_mean) OVER w7 = 7 THEN AVG(temperature_2m_mean) OVER w7 END,
    GREATEST(0, temperature_2m_mean - {COOLING_BASE}),

    EXTRACT(ISODOW FROM as_of)::int - 1,
    EXTRACT(MONTH  FROM as_of)::int,
    (EXTRACT(ISODOW FROM as_of) >= 6)::int,

    temperature_2m_mean,
    temperature_2m_max

FROM electricity_observations
WHERE state = %s
WINDOW w7  AS (ORDER BY as_of RANGE BETWEEN INTERVAL '6 days'  PRECEDING AND CURRENT ROW),
       w30 AS (ORDER BY as_of RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW)
ON CONFLICT (state, as_of) DO UPDATE SET
    demand_lag_1        = EXCLUDED.demand_lag_1,
    demand_lag_2        = EXCLUDED.demand_lag_2,
    demand_lag_6        = EXCLUDED.demand_lag_6,
    demand_roll_7_mean  = EXCLUDED.demand_roll_7_mean,
    demand_roll_7_max   = EXCLUDED.demand_roll_7_max,
    demand_roll_30_mean = EXCLUDED.demand_roll_30_mean,
    temp_lag_1          = EXCLUDED.temp_lag_1,
    temp_roll_7         = EXCLUDED.temp_roll_7,
    cooling_degree_days = EXCLUDED.cooling_degree_days,
    day_of_week         = EXCLUDED.day_of_week,
    month               = EXCLUDED.month,
    is_weekend          = EXCLUDED.is_weekend,
    temperature_2m_mean = EXCLUDED.temperature_2m_mean,
    temperature_2m_max  = EXCLUDED.temperature_2m_max,
    created_at          = CURRENT_TIMESTAMP
WHERE electricity_features.demand_lag_1        IS DISTINCT FROM EXCLUDED.demand_lag_1
   OR electricity_features.demand_lag_2        IS DISTINCT FROM EXCLUDED.demand_lag_2
   OR electricity_features.demand_lag_6        IS DISTINCT FROM EXCLUDED.demand_lag_6
   OR electricity_features.demand_roll_7_mean  IS DISTINCT FROM EXCLUDED.demand_roll_7_mean
   OR electricity_features.demand_roll_7_max   IS DISTINCT FROM EXCLUDED.demand_roll_7_max
   OR electricity_features.demand_roll_30_mean IS DISTINCT FROM EXCLUDED.demand_roll_30_mean
   OR electricity_features.temp_lag_1          IS DISTINCT FROM EXCLUDED.temp_lag_1
   OR electricity_features.temp_roll_7         IS DISTINCT FROM EXCLUDED.temp_roll_7
   OR electricity_features.cooling_degree_days IS DISTINCT FROM EXCLUDED.cooling_degree_days
   OR electricity_features.temperature_2m_mean IS DISTINCT FROM EXCLUDED.temperature_2m_mean
   OR electricity_features.temperature_2m_max  IS DISTINCT FROM EXCLUDED.temperature_2m_max;
"""


GAP_DAYS_SQL, ORPHAN_ROWS_SQL = alignment_sql(
    "electricity_features", "electricity_observations", "state")


def verify_alignment(cur):
    """This state's alignment check. See vericast.verify_alignment for the contract."""
    check_alignment(cur, GAP_DAYS_SQL, ORPHAN_ROWS_SQL, STATE)


def engineer_features():
    require_database_url(DATABASE_URL)
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Serialize overlapping runs; released on COMMIT/ROLLBACK.
                acquire_pipeline_lock(cur, "elec_features")
                cur.execute(ENGINEER_SQL, (STATE,))
                written = cur.rowcount
                print(f"Upserted {written} feature rows for {STATE}")

                cur.execute("""
                    SELECT COUNT(*),
                           COUNT(*) FILTER (WHERE demand_lag_1 IS NULL),
                           COUNT(*) FILTER (WHERE demand_roll_7_mean IS NULL),
                           COUNT(*) FILTER (WHERE demand_roll_30_mean IS NULL),
                           MIN(as_of), MAX(as_of)
                    FROM electricity_features WHERE state = %s
                """, (STATE,))
                total, no_lag1, no_roll7, no_roll30, first, last = cur.fetchone()
                print(f"Total {STATE} feature rows: {total} ({first} -> {last})")
                # Expected NULLs: warm-up at the series start, plus the day after
                # each date gap. Anything beyond that is a data problem.
                print(f"  NULL demand_lag_1: {no_lag1} (series start + day after each gap)")
                print(f"  NULL demand_roll_7_mean: {no_roll7} (first 6 days + gap-spanning windows)")
                print(f"  NULL demand_roll_30_mean: {no_roll30} (first 29 days + gap-spanning windows)")

                # An AssertionError here exits non-zero, so the pipeline stops
                # before predict.py publishes a forecast built on a broken join.
                # Validated BEFORE commit: a failed alignment rolls everything back
                # instead of leaving bad rows committed.
                verify_alignment(cur)
                conn.commit()
    except Exception as e:
        print(f"Error engineering electricity features: {e}")
        raise


if __name__ == "__main__":
    engineer_features()
