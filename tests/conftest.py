"""Makes the repo root importable so `from app import app` works in every test,
and hosts the one mock every endpoint test needs.

wire_cursor() replaces the four-line dance that four test files had each copied:
app.py opens `with get_db_connection() as conn:` then `with conn.cursor() as cur:`,
so a MagicMock has to be wired through both __enter__s or the cursor calls land on
an auto-child mock and the assertions pass against nothing.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wire_cursor(mock_get_db, rows=None):
    """Return the cursor mock reached through a patched app.get_db_connection.

    `rows` sets fetchall for the endpoints that read many rows; fetchone-based
    tests set their own return value on the cursor returned here.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    if rows is not None:
        mock_cursor.fetchall.return_value = rows
    return mock_cursor
