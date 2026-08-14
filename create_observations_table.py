import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def create_observations_table():
    """Create the observations table for storing daily aggregated air quality and weather data"""
    create_table_sql = """
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
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
                print("✅ Observations table created successfully")

                # Verify table exists
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'observations'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                print("\nTable schema:")
                for col in columns:
                    print(f"  {col[0]}: {col[1]}")

    except Exception as e:
        print(f"❌ Error creating observations table: {e}")
        raise

if __name__ == "__main__":
    create_observations_table()