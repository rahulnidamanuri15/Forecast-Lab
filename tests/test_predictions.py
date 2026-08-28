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

def test_predictions_endpoint_success():
    """Test that the predictions endpoint returns 200 with prediction data."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - prediction data
        forecast_date = date.today()
        mock_cursor.fetchall.return_value = [
            (forecast_date, 'lightgbm', 15.5, None, datetime.now(timezone.utc)),
            (forecast_date, 'naive_baseline', 12.3, 12.3, datetime.now(timezone.utc))
        ]

        # Make the request. Deliberately unfiltered: the mock returns rows for
        # both models and ignores the WHERE clause, so asking for
        # ?model=lightgbm here and then asserting a naive_baseline row came back
        # encoded the opposite of the endpoint's contract. The model filter is
        # asserted on the query itself in test_predictions_model_filter_reaches_sql.
        response = client.get("/predictions?limit=5&scored_only=false")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

        # Check first prediction (lightgbm)
        pred1 = data["predictions"][0]
        assert pred1["forecast_date"] == forecast_date.isoformat()
        assert pred1["model"] == "lightgbm"
        assert pred1["predicted_pm2_5"] == 15.5
        assert pred1["actual_pm2_5"] is None
        assert pred1["error"] is None
        assert "created_at" in pred1

        # Check second prediction (naive_baseline)
        pred2 = data["predictions"][1]
        assert pred2["forecast_date"] == forecast_date.isoformat()
        assert pred2["model"] == "naive_baseline"
        assert pred2["predicted_pm2_5"] == 12.3
        assert pred2["actual_pm2_5"] == 12.3
        assert pred2["error"] == 0.0
        assert "created_at" in pred2

        # Check count
        assert data["count"] == 2

def test_predictions_endpoint_scored_only():
    """Test that the predictions endpoint filters scored predictions correctly."""
    # Mock database connection and cursor
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - only scored predictions
        forecast_date = date.today()
        mock_cursor.fetchall.return_value = [
            (forecast_date, 'naive_baseline', 12.3, 12.3, datetime.now(timezone.utc))
        ]

        # Make the request with scored_only=true
        response = client.get("/predictions?scored_only=true")

        # The mock ignores the WHERE clause, so the filter can only be verified
        # on the SQL that was actually sent.
        sql, params = mock_cursor.execute.call_args[0]
        assert "actual_pm2_5 IS NOT NULL" in sql

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

        # Check the prediction is scored
        pred = data["predictions"][0]
        assert pred["actual_pm2_5"] is not None
        assert pred["error"] == 0.0

def test_predictions_model_filter_reaches_sql():
    """?model=lightgbm must become a WHERE clause with the model as a bound
    parameter - not a value interpolated into the SQL, and not silently dropped."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        assert client.get("/predictions?model=lightgbm").status_code == 200

        sql, params = mock_cursor.execute.call_args[0]
        assert "model = %s" in sql
        assert "lightgbm" in params
        assert "lightgbm" not in sql   # bound, never interpolated


def test_predictions_rejects_unknown_model():
    """An unallowlisted model is a 400 before any query runs - that allowlist is
    what makes the f-string WHERE clause in app.py safe."""
    response = client.get("/predictions?model=DROP TABLE predictions")
    assert response.status_code == 400


def test_predictions_endpoint_no_data():
    """Test that the predictions endpoint handles missing prediction data."""
    # Mock database connection to return no data
    with patch('app.get_db_connection') as mock_get_db:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock the database response - no data
        mock_cursor.fetchall.return_value = []

        # Make the request
        response = client.get("/predictions?model=lightgbm")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 0
        assert data["count"] == 0

def test_predictions_endpoint_database_error():
    """Test that the predictions endpoint handles database errors."""
    # Mock database connection to raise an exception
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        # Make the request
        response = client.get("/predictions?model=lightgbm")

        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data