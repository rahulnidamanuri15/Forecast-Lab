import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import date, datetime, timezone

# Import the app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

def test_forecast_endpoint_lightgbm_success():
    """Test that the forecast endpoint returns 200 for lightgbm model."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - forecast data
        forecast_date = date.today()
        mock_cursor.fetchone.return_value = [
            forecast_date,  # forecast_date
            15.5,           # predicted_pm2_5
            None,           # actual_pm2_5 (pending)
            'lightgbm',     # model
            datetime.now(timezone.utc)  # created_at
        ]

        # Make the request
        response = client.get("/forecast?model=lightgbm")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Nagpur"
        assert data["forecast_date"] == forecast_date.isoformat()
        assert data["forecast_pm2_5"] == 15.5
        assert data["model"] == "lightgbm"
        assert data["actual_pm2_5"] is None
        assert data["status"] == "pending"

def test_forecast_endpoint_naive_baseline_success():
    """Test that the forecast endpoint returns 200 for naive_baseline model."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - forecast data
        forecast_date = date.today()
        mock_cursor.fetchone.return_value = [
            forecast_date,  # forecast_date
            12.3,           # predicted_pm2_5
            12.3,           # actual_pm2_5 (scored)
            'naive_baseline', # model
            datetime.now(timezone.utc)  # created_at
        ]

        # Make the request
        response = client.get("/forecast?model=naive_baseline")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Nagpur"
        assert data["forecast_date"] == forecast_date.isoformat()
        assert data["forecast_pm2_5"] == 12.3
        assert data["model"] == "naive_baseline"
        assert data["actual_pm2_5"] == 12.3
        assert data["status"] == "verified"

def test_forecast_endpoint_invalid_model():
    """Test that the forecast endpoint handles invalid model parameters."""
    # Make the request with invalid model
    response = client.get("/forecast?model=invalid_model")

    # Assertions
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Unsupported model" in data["detail"]

def test_forecast_endpoint_no_data():
    """Test that the forecast endpoint handles missing forecast data."""
    # Mock database connection to return no data
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - no data
        mock_cursor.fetchone.return_value = None

        # Make the request
        response = client.get("/forecast?model=lightgbm")

        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "No lightgbm forecast found" in data["detail"]

def test_forecast_endpoint_database_error():
    """Test that the forecast endpoint handles database errors."""
    # Mock database connection to raise an exception
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        # Make the request
        response = client.get("/forecast?model=lightgbm")

        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data