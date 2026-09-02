import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from conftest import wire_cursor

client = TestClient(app)

def test_health_endpoint_success():
    """Test that the health endpoint returns 200 when database is healthy."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)

        mock_cursor.fetchone.return_value = [datetime.now(timezone.utc).date()]

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "latest_observation" in data
        assert "stale_days" in data
        # Current data must read as fresh, or the dashboard's LIVE pill never lights.
        assert data["source_lag_expected"] is True
        # The cache header is what keeps a looping crawler off the 8-slot pool.
        assert response.headers["Cache-Control"] == "public, max-age=300"


def test_health_endpoint_flags_stalled_source():
    """A stalled upstream is status ok but NOT source_lag_expected.

    The failure this guards: vericast/pm25/predict.py anchors forecast_date to
    the latest observation, so a week-stale Open-Meteo still produces a forecast
    that passes every internal check. status alone cannot distinguish it - this
    flag is what index.html gates the LIVE pill on.
    """
    stale_date = datetime.now(timezone.utc).date() - timedelta(days=9)
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)
        mock_cursor.fetchone.return_value = [stale_date]

        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["stale_days"] >= 9
        assert data["source_lag_expected"] is False

        # The electricity mirror tolerates more, but not 9 days either.
        data = client.get("/electricity/health").json()
        assert data["source_lag_expected"] is False


def test_health_endpoint_database_error():
    """Test that the health endpoint handles database errors gracefully."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data


def test_health_endpoint_empty_table():
    """An empty observations table is 200 no_data, not a 503 outage."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)

        mock_cursor.fetchone.return_value = [None]

        for route in ("/health", "/electricity/health"):
            response = client.get(route)
            assert response.status_code == 200, route
            data = response.json()
            assert data["status"] == "no_data", route
            assert data["latest_observation"] is None, route
            assert data["stale_days"] is None, route
            # Both endpoints carry the key in every branch, so the frontend can
            # read it without checking which target it is looking at.
            assert data["source_lag_expected"] is None, route
