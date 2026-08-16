import psycopg
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(as_of), MAX(as_of) FROM observations WHERE city = 'Nagpur'")
        min_date, max_date = cur.fetchone()
        print(f'Date range in observations: {min_date} to {max_date}')
        cur.execute("SELECT COUNT(*) FROM observations WHERE city = 'Nagpur'")
        count = cur.fetchone()[0]
        print(f'Total observations: {count}')