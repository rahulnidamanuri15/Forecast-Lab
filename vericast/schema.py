"""DDL for every table in both targets. Idempotent: CREATE TABLE IF NOT EXISTS
plus ADD COLUMN IF NOT EXISTS only, so running it against the live database
cannot touch the published record.

    python -m vericast.schema

Replaces the four one-table-per-file create_*_table.py scripts. ponytail: no
migration tool - MIGRATIONS below is an idempotent tuple of statements, not a
versioned history, so it has no down-migration and no ordering guarantees beyond
"top to bottom". Add Alembic when a column needs to change rather than be added.
"""
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Note demand_lag_6, not lag_7. Features at as_of = t predict t+1, so the
# same-weekday-last-week value for the *target* is y(t-6), not y(t-7). The same
# column feeds LightGBM's weekly signal and the seasonal_naive baseline.
TABLES = {
    "observations": """
        CREATE TABLE IF NOT EXISTS observations (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,
            pm2_5 FLOAT,
            pm10 FLOAT,
            temperature_2m_mean FLOAT,
            wind_speed_10m_max FLOAT,
            precipitation_sum FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, as_of)
        );
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,

            pm2_5_lag_1 FLOAT,
            pm10_lag_1 FLOAT,
            temperature_lag_1 FLOAT,
            wind_speed_lag_1 FLOAT,
            precipitation_lag_1 FLOAT,

            pm2_5_roll_7 FLOAT,
            pm2_5_roll_30 FLOAT,
            pm10_roll_7 FLOAT,
            pm10_roll_30 FLOAT,

            day_of_week INTEGER,  -- 0=Monday, 6=Sunday
            month INTEGER,        -- 1-12
            is_weekend BOOLEAN,

            temperature_2m_mean FLOAT,
            wind_speed_10m_max FLOAT,
            precipitation_sum FLOAT,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, as_of)
        );
    """,
    # `source` splits the record: a 'daily' row was published before its
    # actual_pm2_5 existed, a 'backtest' row was written by
    # experiments/save_backtest_results.py with the actual already in hand.
    # /evaluation aggregates these rows directly and is the headline accuracy
    # claim, so averaging the two answers a different question than the one this
    # project exists to answer.
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_pm2_5 FLOAT,
            actual_pm2_5 FLOAT,
            model VARCHAR(50) DEFAULT 'naive_baseline',
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, forecast_date, model)
        );
    """,
    # No city column, unlike its electricity counterpart: this table predates the
    # second target, so changing CITY needs a migration, not just an env var -
    # two cities' rows would land on the same (score_date, model) key.
    #
    # Rows come from vericast/pm25/score.py (one scored day, sample_size normally
    # 1) and, once at launch, from experiments/save_backtest_results.py (a whole
    # backtest, sample_size in the hundreds). /leaderboard reads the latest
    # score_date per model, so without `source` a backtest re-run today would
    # become the published leaderboard.
    "model_performance": """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            score_date DATE NOT NULL,
            model VARCHAR(50) NOT NULL,
            mae FLOAT,
            rmse FLOAT,
            sample_size INTEGER,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(score_date, model)
        );
    """,
    "electricity_observations": """
        CREATE TABLE IF NOT EXISTS electricity_observations (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,
            peak_demand_mw FLOAT NOT NULL,
            energy_met_mu FLOAT,
            temperature_2m_mean FLOAT,
            temperature_2m_max FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, as_of)
        );
    """,
    "electricity_features": """
        CREATE TABLE IF NOT EXISTS electricity_features (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,

            demand_lag_1 FLOAT,
            demand_lag_2 FLOAT,
            demand_lag_6 FLOAT,

            demand_roll_7_mean FLOAT,
            demand_roll_7_max FLOAT,
            demand_roll_30_mean FLOAT,

            temp_lag_1 FLOAT,
            temp_roll_7 FLOAT,
            cooling_degree_days FLOAT,

            day_of_week INT,
            month INT,
            is_weekend INT,

            temperature_2m_mean FLOAT,
            temperature_2m_max FLOAT,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, as_of)
        );
    """,
    # `source` as on predictions above: 'daily' means published before the actual
    # existed, 'backtest' means seeded by experiments/save_elec_backtest_results.py.
    "electricity_predictions": """
        CREATE TABLE IF NOT EXISTS electricity_predictions (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_demand_mw FLOAT NOT NULL,
            actual_demand_mw FLOAT,
            model VARCHAR(50) NOT NULL,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, forecast_date, model)
        );
    """,
    # mape lives here and not on model_performance: adding a column to a live table
    # the published record reads from is unrequested risk. `source` is the one
    # exception, and it went on both tables together.
    "electricity_model_performance": """
        CREATE TABLE IF NOT EXISTS electricity_model_performance (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            score_date DATE NOT NULL,
            model VARCHAR(50) NOT NULL,
            mae FLOAT NOT NULL,
            rmse FLOAT NOT NULL,
            mape FLOAT,
            sample_size INT NOT NULL,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, score_date, model)
        );
    """,
}

# Columns added after the tables were already live. ADD COLUMN IF NOT EXISTS keeps
# this as idempotent as the CREATEs above, so `python -m vericast.schema` stays safe
# to run against production - which is what makes this the migration rather than a
# psql snippet only one machine ever ran.
#
# The backfill is the half that is easy to miss: ADD COLUMN ... DEFAULT 'daily'
# marks the existing backtest rows 'daily' too.
#
# All four backfills are bounded by created_at, because an unbounded one is a
# standing rule rather than a migration. Every writer now labels its own rows, so
# only rows predating the column need fixing - and those are exactly what the bound
# selects. Unbounded, a legitimate multi-day daily score would be relabelled
# 'backtest' and drop off /leaderboard permanently, with no down-migration to undo
# it. The other available bound (MAX(score_date) WHERE source = 'backtest') is a
# value the seeders *advance*, so "re-run the seeder, then run this module" was two
# routine operations that together relabelled verified rows.
#
# The date is in the past and must stay there: a future cutoff leaves the window
# open on rows being written today, which is the whole failure mode.
MIGRATION_CUTOFF = "2026-08-29"  # `source` landed this day

MIGRATIONS = (
    "ALTER TABLE model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "ALTER TABLE electricity_model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    # Aggregate tables. A daily row scores one date against a UNIQUE(city,
    # forecast_date, model) predictions table, so its sample_size is 1 by
    # construction - anything larger came from a backtest.
    "UPDATE model_performance SET source = 'backtest' "
    f"WHERE sample_size > 1 AND source = 'daily' AND created_at < '{MIGRATION_CUTOFF}';",
    "UPDATE electricity_model_performance SET source = 'backtest' "
    f"WHERE sample_size > 1 AND source = 'daily' AND created_at < '{MIGRATION_CUTOFF}';",

    # Row-level tables. /evaluation aggregates these directly and is the headline
    # accuracy claim, so it needs the same column.
    "ALTER TABLE predictions "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "ALTER TABLE electricity_predictions "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    # The bound is derived from the data rather than guessed: the backtest's own
    # aggregate row records score_date = its last evaluated date, so every
    # prediction row it wrote has forecast_date <= that date. A NULL subquery (no
    # backtest seeded) makes the comparison NULL and updates nothing.
    "UPDATE predictions SET source = 'backtest' "
    "WHERE source = 'daily' AND forecast_date <= "
    "(SELECT MAX(score_date) FROM model_performance WHERE source = 'backtest') "
    f"AND created_at < '{MIGRATION_CUTOFF}';",
    "UPDATE electricity_predictions p SET source = 'backtest' "
    "WHERE p.source = 'daily' AND p.forecast_date <= "
    "(SELECT MAX(score_date) FROM electricity_model_performance "
    " WHERE source = 'backtest' AND state = p.state) "
    f"AND p.created_at < '{MIGRATION_CUTOFF}';",
)


def create_tables():
    # Runs unattended in ci.yml; without this a missing DSN reaches psycopg as None
    # and surfaces as a connection-string parse error rather than the real problem.
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for name, ddl in TABLES.items():
                cur.execute(ddl)
                print(f"[OK] {name}")
            conn.commit()

            for sql in MIGRATIONS:
                cur.execute(sql)
                print(f"[OK] {sql[:58]}... ({cur.rowcount} rows)")
            conn.commit()

            for name in TABLES:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (name,))
                cols = cur.fetchall()
                print(f"\n{name} ({len(cols)} columns):")
                for col_name, dtype in cols:
                    print(f"  {col_name}: {dtype}")


if __name__ == "__main__":
    create_tables()
