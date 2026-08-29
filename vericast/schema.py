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
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_pm2_5 FLOAT,
            actual_pm2_5 FLOAT,
            model VARCHAR(50) DEFAULT 'naive_baseline',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, forecast_date, model)
        );
    """,
    # No city column, unlike its electricity counterpart: this table predates
    # the second target and the live PM2.5 record is keyed on (score_date, model).
    # Consequence, since nothing else states it: changing CITY needs a migration,
    # not just an env var - rows from two cities would land on the same
    # (score_date, model) key and overwrite each other.
    #
    # Rows here come from vericast/pm25/score.py (one scored day, sample_size
    # normally 1) and, once at launch, from experiments/save_backtest_results.py
    # (a whole backtest, sample_size in the hundreds). `source` is what tells them
    # apart, because /leaderboard needs it: it reads the latest score_date per
    # model, so a backtest re-run today would become the published leaderboard
    # with a sample_size of several hundred. The DEFAULT is 'daily' so the honest
    # writer (score.py) never has to name the column - only the backtests do.
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
    "electricity_predictions": """
        CREATE TABLE IF NOT EXISTS electricity_predictions (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_demand_mw FLOAT NOT NULL,
            actual_demand_mw FLOAT,
            model VARCHAR(50) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, forecast_date, model)
        );
    """,
    # mape lives here and not on the existing model_performance table: adding a
    # column to a live table the published record reads from is unrequested risk.
    # `source` is the one exception, and it went on both tables together - see the
    # model_performance comment above for why /leaderboard needs it.
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
# this as idempotent as the CREATEs above, so `python -m vericast.schema` is still
# safe to run against production - which is what makes this the migration rather
# than a psql snippet in the README that only one machine ever ran.
#
# The backfill is the half that is easy to miss: ADD COLUMN ... DEFAULT 'daily'
# marks the existing backtest rows 'daily' too, so /leaderboard would still be
# hijackable by the rows that are already there. A daily row is one scored day
# against a UNIQUE(city, forecast_date, model) predictions table, so its
# sample_size is 1 by construction - anything larger came from a backtest.
MIGRATIONS = (
    "ALTER TABLE model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "ALTER TABLE electricity_model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "UPDATE model_performance SET source = 'backtest' "
    "WHERE sample_size > 1 AND source = 'daily';",
    "UPDATE electricity_model_performance SET source = 'backtest' "
    "WHERE sample_size > 1 AND source = 'daily';",
)


def create_tables():
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
