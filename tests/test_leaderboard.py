import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import date

# Import the app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

def test_leaderboard_endpoint_success():
    """Test that the leaderboard endpoint returns 200 with model performance data."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - model performance data.
        # sample_size 400 rather than a daily 1: these are the backtest-seeded
        # rows experiments/save_backtest_results.py writes, which is what a fresh
        # deployment's leaderboard actually serves. See README, "Not on the
        # production path".
        mock_cursor.fetchall.return_value = [
            ('lightgbm', 2.5, 3.2, 400, date(2026, 8, 17)),
            ('naive_baseline', 3.1, 4.0, 400, date(2026, 8, 17))
        ]

        # Make the request
        response = client.get("/leaderboard")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 2

        # Check that lightgbm comes first (lower MAE)
        assert data["leaderboard"][0]["model"] == "lightgbm"
        assert data["leaderboard"][0]["mae"] == 2.5
        assert data["leaderboard"][0]["rmse"] == 3.2
        assert data["leaderboard"][0]["sample_size"] == 400
        assert data["leaderboard"][0]["as_of"] == date(2026, 8, 17).isoformat()

        # Check that naive_baseline comes second
        assert data["leaderboard"][1]["model"] == "naive_baseline"
        assert data["leaderboard"][1]["mae"] == 3.1
        assert data["leaderboard"][1]["rmse"] == 4.0
        assert data["leaderboard"][1]["sample_size"] == 400
        assert data["leaderboard"][1]["as_of"] == date(2026, 8, 17).isoformat()

def test_leaderboard_endpoint_no_data():
    """Test that the leaderboard endpoint handles missing model performance data."""
    # Mock database connection to return no data
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - no data
        mock_cursor.fetchall.return_value = []

        # Make the request
        response = client.get("/leaderboard")

        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "No model performance data found" in data["detail"]

def test_leaderboard_endpoint_database_error():
    """Test that the leaderboard endpoint handles database errors."""
    # Mock database connection to raise an exception
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        # Make the request
        response = client.get("/leaderboard")

        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data