"""Makes the repo root importable so `from app import app` works in every test.

No fixtures: each test file builds its own TestClient and patches
app.get_db_connection with the entered-context mock shape that matches
`with conn.cursor() as cur`. A shared fixture here had the wrong shape and was
referenced by nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
