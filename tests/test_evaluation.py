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

def test_evaluation_endpoint_success():
    """Test that the evaluation endpoint returns 200 with evaluation data."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - prediction data for evaluation
        # Format: (model, predicted_pm2_5, actual_pm2_5)
        mock_cursor.fetchall.return_value = [
            ('lightgbm', 15.0, 14.5),   # scored
            ('lightgbm', 16.0, 15.8),   # scored
            ('lightgbm', 15.5, None),   # pending
            ('naive_baseline', 12.0, 12.2), # scored
            ('naive_baseline', None, 12.0), # pending (predicted is None)
        ]

        # Make the request
        response = client.get("/evaluation?days=7")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert len(data["evaluation"]) == 2

        # Find lightgbm evaluation
        lightgbm_eval = next((e for e in data["evaluation"] if e["model"] == "lightgbm"), None)
        assert lightgbm_eval is not None
        assert lightgbm_eval["window_days"] == 7
        assert lightgbm_eval["scored_count"] == 2  # Only 2 scored predictions
        assert lightgbm_eval["pending_count"] == 1  # 1 pending prediction
        assert abs(lightgbm_eval["mae"] - 0.35) < 0.001  # (|14.5-15.0| + |15.8-16.0|)/2 = (0.5 + 0.2)/2 = 0.35
        # RMSE = sqrt(((14.5-15.0)^2 + (15.8-16.0)^2)/2) = sqrt((0.25 + 0.04)/2) = sqrt(0.145) ≈ 0.381
        assert abs(lightgbm_eval["rmse"] - 0.381) < 0.01

        # Find naive_baseline evaluation
        naive_eval = next((e for e in data["evaluation"] if e["model"] == "naive_baseline"), None)
        assert naive_eval is not None
        assert naive_eval["window_days"] == 7
        assert naive_eval["scored_count"] == 1  # Only 1 scored prediction (the one with both predicted and actual)
        assert naive_eval["pending_count"] == 1  # 1 pending prediction (predicted is None)
        assert abs(naive_eval["mae"] - 0.2) < 0.001  # |12.2-12.0| = 0.2
        assert abs(naive_eval["rmse"] - 0.2) < 0.001  # sqrt((12.2-12.0)^2) = sqrt(0.04) = 0.2

def test_evaluation_endpoint_no_data():
    """Test that the evaluation endpoint handles missing prediction data."""
    # Mock database connection to return no data
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - no data
        mock_cursor.fetchall.return_value = []

        # Make the request
        response = client.get("/evaluation?days=7")

        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "No predictions found in that window" in data["detail"]

def test_evaluation_endpoint_database_error():
    """Test that the evaluation endpoint handles database errors."""
    # Mock database connection to raise an exception
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        # Make the request
        response = client.get("/evaluation?days=7")

        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


def test_evaluation_full_record_omits_window():
    """No days= means every prediction ever published, and window_days is null."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            ('lightgbm', 15.0, 14.5),
            ('lightgbm', 16.0, 15.8),
        ]

        response = client.get("/evaluation")

        assert response.status_code == 200
        entry = response.json()["evaluation"][0]
        assert entry["window_days"] is None
        assert entry["scored_count"] == 2

        # The full-record query must not filter on forecast_date at all.
        sql = mock_cursor.execute.call_args[0][0]
        assert "forecast_date" not in sql


def test_evaluation_rejects_negative_days():
    """days=-1 is a client error, not a silently-empty window."""
    response = client.get("/evaluation?days=-1")
    assert response.status_code == 400


def test_evaluation_sorts_unscored_models_last():
    """A model with no scored rows must not blow up the sort."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Both models are entirely pending -> two None maes.
        mock_cursor.fetchall.return_value = [
            ('lightgbm', 15.0, None),
            ('naive_baseline', 12.0, None),
        ]

        response = client.get("/evaluation?days=7")

        assert response.status_code == 200
        assert all(e["mae"] is None for e in response.json()["evaluation"])