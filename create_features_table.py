import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def create_features_table():
    """Create the features table for storing engineered features"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS features (
        id SERIAL PRIMARY KEY,
        city VARCHAR(100) NOT NULL,
        as_of DATE NOT NULL,
        -- Lagged features (previous day's values)
        pm2_5_lag_1 FLOAT,
        pm10_lag_1 FLOAT,
        temperature_lag_1 FLOAT,
        wind_speed_lag_1 FLOAT,
        precipitation_lag_1 FLOAT,
        -- Rolling averages
        pm2_5_roll_7 FLOAT,
        pm2_5_roll_30 FLOAT,
        pm10_roll_7 FLOAT,
        pm10_roll_30 FLOAT,
        -- Time-based features
        day_of_week INTEGER,  -- 0=Monday, 6=Sunday
        month INTEGER,        -- 1-12
        is_weekend BOOLEAN,   -- Saturday or Sunday
        -- Weather features (same day)
        temperature_2m_mean FLOAT,
        wind_speed_10m_max FLOAT,
        precipitation_sum FLOAT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(city, as_of)
    );
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
                print("Features table created successfully")

                # Verify table exists
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'features'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                print("\nTable schema:")
                for col in columns:
                    print(f"  {col[0]}: {col[1]}")

    except Exception as e:
        print(f"Error creating features table: {e}")
        raise

if __name__ == "__main__":
    create_features_table()