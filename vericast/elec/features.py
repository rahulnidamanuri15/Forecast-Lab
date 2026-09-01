"""Engineer the point-in-time electricity feature store.

One idempotent INSERT ... SELECT. This is the design both feature stores use -
vericast/pm25/features.py was ported to it from a Python row-loop - because
Postgres window frames give guarantees a loop cannot:

  * `RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING` is
    date-addressed, not row-addressed, so it returns NULL across a date gap on
    its own - no explicit gap guard to forget. (Maharashtra has one such gap:
    2025-05-21 -> 2025-05-24.)
  * `COUNT(peak_demand_mw) OVER w7 = 7` is stricter than a row-index check:
    counting rows averages the previous 7 *rows* whether or not they are 7
    consecutive days, silently spanning gaps. This refuses instead. It counts the
    averaged column rather than `*` so a NULL value counts as absent, which
    matters for temp_roll_7: temperature_2m_mean is nullable, and `AVG` skips
    NULLs, so `COUNT(*)` would label a 6-value mean a 7-day one.
  * Look-ahead leakage is structurally unexpressible - a `RANGE ... PRECEDING`
    frame cannot reference a future row. That is a claim about this SQL as
    written, though, not about the rows in the table, so
    vericast/elec/leakage_test.py re-derives every stored value from the
    observations by calendar date and the daily job runs it as its own step -
    the same arrangement vericast/pm25/ has. The complementary check is the
    features(t) -> target(t+1) *join*, which verify_alignment() below runs at the
    end of every engineer_features() call, so it fires on the daily cron and not
    only in tests/test_feature_alignment.py on push.

Full recompute every run, so changing a feature definition needs no backfill.
"""
import os
import psycopg
from dotenv import load_dotenv

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
    created_at          = CURRENT_TIMESTAMP;
"""


# Every feature row must have a next-day observation to be the target of, except
# where the observation series itself has a hole. Both counts come from the same
# shape so they can be compared directly: an orphan is only excusable if a gap
# explains it.
#
# The join half of the daily leakage gate; leakage_test.py is the value half, and
# neither substitutes for the other. This one runs inline here rather than as its
# own step so it fires before the SQL above can hand a broken join to predict.py.
ORPHAN_ROWS_SQL = """
SELECT COUNT(*)
FROM electricity_features f
WHERE f.state = %s
  AND f.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM electricity_observations WHERE state = f.state)
  AND NOT EXISTS (
      SELECT 1 FROM electricity_observations o
      WHERE o.state = f.state
        AND o.as_of = f.as_of + INTERVAL '1 day'
  )
"""

GAP_DAYS_SQL = """
SELECT COUNT(*)
FROM electricity_observations o
WHERE o.state = %s
  AND o.as_of < (SELECT MAX(as_of) - INTERVAL '1 day'
                 FROM electricity_observations WHERE state = o.state)
  AND NOT EXISTS (
      SELECT 1 FROM electricity_observations n
      WHERE n.state = o.state
        AND n.as_of = o.as_of + INTERVAL '1 day'
  )
"""


def verify_alignment(cur):
    """Enforce the features(t) -> target(t+1) contract, gaps accounted for.

    Equality, not a tolerance: hardcoding "<= 1" for Maharashtra's known
    2025-05-21 -> 2025-05-24 hole absorbs the next gap silently, and absorbs a
    genuinely broken join just as quietly. Deriving the expected count from the
    observations means a NEW gap fails here - loudly, on the day it appears -
    rather than being pre-forgiven.

    Raised, not asserted. This is a data-integrity gate on the daily path, and
    `assert` is erased by python -O / PYTHONOPTIMIZE=1: the one interpreter flag
    someone adds for speed would turn every gate in this pipeline into a no-op
    that still exits 0. AssertionError keeps the type and message a caller may
    already match on. Same reasoning in vericast/pm25/features.py and both
    train.py modules; the remaining bare asserts are all in `__main__`
    self-checks, where being erased under -O costs nothing.
    """
    cur.execute(GAP_DAYS_SQL, (STATE,))
    gaps = cur.fetchone()[0]
    cur.execute(ORPHAN_ROWS_SQL, (STATE,))
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

                # Runs inside the daily job's "Engineer features" step, right
                # after the SQL that could break it. An AssertionError here exits
                # non-zero, so the pipeline stops before predict.py publishes a
                # forecast built on a broken join.
                verify_alignment(cur)
    except Exception as e:
        print(f"Error engineering electricity features: {e}")
        raise


if __name__ == "__main__":
    engineer_features()
