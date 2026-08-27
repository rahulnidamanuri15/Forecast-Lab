"""DDL for every table in both targets. Idempotent: CREATE TABLE IF NOT EXISTS
only, so running it against the live database cannot touch the published record.

    python -m vericast.schema

Replaces the four one-table-per-file create_*_table.py scripts. ponytail: no
migration tool - these tables are created once and columns are added by hand;
add Alembic when a column actually needs to change on a live table.
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
    "model_performance": """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            score_date DATE NOT NULL,
            model VARCHAR(50) NOT NULL,
            mae FLOAT,
            rmse FLOAT,
            sample_size INTEGER,
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
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, score_date, model)
        );
    """,
}


def create_tables():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for name, ddl in TABLES.items():
                cur.execute(ddl)
                print(f"[OK] {name}")
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
