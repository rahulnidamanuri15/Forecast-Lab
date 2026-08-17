import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def create_model_performance_table():
    """Create the model performance table for storing daily MAE and RMSE"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS model_performance (
        id SERIAL PRIMARY KEY,
        score_date DATE NOT NULL,  -- The date when the performance was scored
        model VARCHAR(50) NOT NULL,
        mae FLOAT,
        rmse FLOAT,
        sample_size INTEGER,  -- Number of predictions used to compute the metrics
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(score_date, model)
    );
    """

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
                print("✅ Model performance table created successfully")

                # Verify table exists
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'model_performance'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                print("\nTable schema:")
                for col in columns:
                    print(f"  {col[0]}: {col[1]}")

    except Exception as e:
        print(f"❌ Error creating model performance table: {e}")
        raise

if __name__ == "__main__":
    create_model_performance_table()