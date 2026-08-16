import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def create_predictions_table():
    """Create the predictions table for storing daily forecasts and actuals"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        city VARCHAR(100) NOT NULL,
        forecast_date DATE NOT NULL,  -- The date being forecasted
        predicted_pm2_5 FLOAT,
        actual_pm2_5 FLOAT,
        model VARCHAR(50) DEFAULT 'naive_baseline',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(city, forecast_date)
    );
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
                print("✅ Predictions table created successfully")

                # Verify table exists
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'predictions'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                print("\nTable schema:")
                for col in columns:
                    print(f"  {col[0]}: {col[1]}")

    except Exception as e:
        print(f"❌ Error creating predictions table: {e}")
        raise

if __name__ == "__main__":
    create_predictions_table()