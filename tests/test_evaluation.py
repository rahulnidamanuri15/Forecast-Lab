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

# /evaluation aggregates in SQL, so a mocked row is one per model:
# (model, scored_count, pending_count, mae, rmse). The arithmetic itself is
# Postgres's now and is checked against real rows in test_feature_alignment.py;
# what these tests cover is the payload shape, the window/full-record branch,
# and the sort.

def test_evaluation_endpoint_success():
    """Test that the evaluation endpoint returns 200 with evaluation data."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            ('lightgbm', 2, 1, 0.35, 0.3807886553),
            ('naive_baseline', 1, 1, 0.2, 0.2),
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
        assert lightgbm_eval["scored_count"] == 2
        assert lightgbm_eval["pending_count"] == 1
        assert abs(lightgbm_eval["mae"] - 0.35) < 0.001
        assert abs(lightgbm_eval["rmse"] - 0.381) < 0.01
        # Same payload shape as /electricity/evaluation, description included.
        assert lightgbm_eval["description"]

        # Find naive_baseline evaluation
        naive_eval = next((e for e in data["evaluation"] if e["model"] == "naive_baseline"), None)
        assert naive_eval is not None
        assert naive_eval["window_days"] == 7
        assert naive_eval["scored_count"] == 1
        assert naive_eval["pending_count"] == 1
        assert abs(naive_eval["mae"] - 0.2) < 0.001
        assert abs(naive_eval["rmse"] - 0.2) < 0.001

        # The windowed query must be bounded by forecast_date, and must not
        # fetch the rows themselves - one aggregate row per model only.
        sql = mock_cursor.execute.call_args[0][0]
        assert "forecast_date" in sql
        assert "GROUP BY model" in sql

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
            ('lightgbm', 2, 0, 0.35, 0.3807886553),
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
            ('lightgbm', 0, 1, None, None),
            ('naive_baseline', 0, 1, None, None),
        ]

        response = client.get("/evaluation?days=7")

        assert response.status_code == 200
        assert all(e["mae"] is None for e in response.json()["evaluation"])