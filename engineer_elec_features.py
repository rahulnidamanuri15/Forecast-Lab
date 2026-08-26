"""Engineer the point-in-time electricity feature store.

One idempotent INSERT ... SELECT instead of the Python row-loop that
engineer_features.py uses for PM2.5, because Postgres window frames do the same
job with stronger guarantees:

  * `RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING` is
    date-addressed, not row-addressed, so it returns NULL across a date gap on
    its own - no explicit gap guard to forget. (Maharashtra has one such gap:
    2025-05-21 -> 2025-05-24.)
  * `COUNT(*) OVER w7 = 7` is stricter than a row-index check: the Python
    version averages the previous 7 *rows* whether or not they are 7 consecutive
    days, silently spanning gaps. This refuses instead.
  * Look-ahead leakage is structurally unexpressible - a `RANGE ... PRECEDING`
    frame cannot reference a future row. That is why there is no
    elec_leakage_test.py mirroring leakage_test.py; what does need asserting is
    the features(t) -> target(t+1) *join*, which lives in
    tests/test_feature_alignment.py.

Full recompute every run, so changing a feature definition needs no backfill.
"""
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

STATE = "Maharashtra"

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

    CASE WHEN COUNT(*) OVER w7  = 7  THEN AVG(peak_demand_mw) OVER w7  END,
    CASE WHEN COUNT(*) OVER w7  = 7  THEN MAX(peak_demand_mw) OVER w7  END,
    CASE WHEN COUNT(*) OVER w30 = 30 THEN AVG(peak_demand_mw) OVER w30 END,

    MAX(temperature_2m_mean) OVER (ORDER BY as_of
        RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING),
    CASE WHEN COUNT(*) OVER w7 = 7 THEN AVG(temperature_2m_mean) OVER w7 END,
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
    created_at          = CURRENT_TIMESTAMP;
"""


def engineer_features():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(ENGINEER_SQL, (STATE,))
                written = cur.rowcount
                conn.commit()
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
    except Exception as e:
        print(f"Error engineering electricity features: {e}")
        raise


if __name__ == "__main__":
    engineer_features()
