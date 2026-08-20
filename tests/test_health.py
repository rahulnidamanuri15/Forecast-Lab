import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Import the app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

def test_health_endpoint_success():
    """Test that the health endpoint returns 200 when database is healthy."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - latest observation from yesterday
        mock_cursor.fetchone.return_value = [datetime.now(timezone.utc).date()]

        # Make the request
        response = client.get("/health")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "latest_observation" in data
        assert "stale_days" in data

def test_health_endpoint_database_error():
    """Test that the health endpoint handles database errors gracefully."""
    # Mock database connection to raise an exception
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        # Make the request
        response = client.get("/health")

        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data