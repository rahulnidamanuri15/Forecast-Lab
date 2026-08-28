"""/history — the empty-table path.

There was no test for this endpoint, and it was the one handler whose 404 was
raised inside its own `try` under a bare `except Exception`, so an empty
observations table came back as a 500 "Internal server error" instead. The
`except HTTPException: raise` re-raise that every other handler already had is
what these two tests hold in place.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

CITY = os.getenv("CITY", "Nagpur")

client = TestClient(app)


def _mock_cursor(mock_get_db):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_cursor


def test_history_empty_table_is_404_not_500():
    """An empty observations table is a 404, not an Internal server error.

    The regression this guards: the 404 is raised inside the try block, so
    without `except HTTPException: raise` the bare handler below it swallowed it
    and returned db_error()'s 500.
    """
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        cur.fetchall.return_value = []

        response = client.get("/history?days=30")

        assert response.status_code == 404
        assert "No historical data found" in response.json()["detail"]


def test_history_returns_oldest_first_and_binds_the_limit():
    """Rows come back chronological for charting, and days= is a bound LIMIT."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        # SQL orders DESC; the handler reverses.
        cur.fetchall.return_value = [
            (date(2026, 8, 27), 31.0, 60.0, 28.1, 12.0, 0.0),
            (date(2026, 8, 26), 29.5, 58.0, 27.4, 11.0, 1.2),
        ]

        response = client.get("/history?days=2")

        assert response.status_code == 200
        data = response.json()
        assert [r["date"] for r in data["historical_data"]] == \
            ["2026-08-26", "2026-08-27"]
        assert data["days_returned"] == 2
        assert data["city"] == CITY

        _, params = cur.execute.call_args[0]
        assert params == (CITY, 2)


def test_history_rejects_zero_days():
    """days is ge=1: 0 is a 422 from the query validator, not an empty window."""
    assert client.get("/history?days=0").status_code == 422
