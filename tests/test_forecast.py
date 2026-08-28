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


# --------------------------------------------------------------------------
# /electricity/* routes. Same mocking shape; the allowlist and the MAPE
# arithmetic are the parts that can silently go wrong.
# --------------------------------------------------------------------------

def _mock_cursor(mock_get_db):
    """Wire a MagicMock cursor into the `with get_db_connection()` pattern."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_cursor


def test_electricity_forecast_success():
    """A pending electricity forecast comes back with MW units and status."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        forecast_date = date(2026, 8, 23)
        cur.fetchone.return_value = [
            forecast_date, 25923.9, None, 'lightgbm', datetime.now(timezone.utc)
        ]

        response = client.get("/electricity/forecast?model=lightgbm")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "Maharashtra"
        assert data["forecast_demand_mw"] == 25923.9
        assert data["status"] == "pending"


def test_electricity_forecast_seasonal_naive_allowed():
    """seasonal_naive is a published electricity model, so it must not 400."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        cur.fetchone.return_value = [
            date(2026, 8, 23), 24907.0, 24907.0, 'seasonal_naive',
            datetime.now(timezone.utc)
        ]

        response = client.get("/electricity/forecast?model=seasonal_naive")

        assert response.status_code == 200
        assert response.json()["status"] == "verified"


def test_seasonal_naive_rejected_on_pm25_forecast():
    """The two allowlists stay separate: there is no PM2.5 seasonal_naive model,
    so /forecast must keep rejecting it even though /electricity/forecast takes it."""
    response = client.get("/forecast?model=seasonal_naive")
    assert response.status_code == 400


def test_electricity_forecast_invalid_model():
    response = client.get("/electricity/forecast?model=invalid_model")
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


def test_electricity_evaluation_metrics():
    """Payload shape, MAPE passthrough, and the MAE sort putting the best first.

    /electricity/evaluation aggregates in SQL, so one mocked row per model:
    (model, scored, pending, mae, rmse, mape).
    """
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        cur.fetchall.return_value = [
            ('lightgbm', 2, 0, 100.0, 100.0, 10.0),
            ('naive_baseline', 2, 0, 100.0, 141.4213562, 10.0),
            ('seasonal_naive', 0, 1, None, None, None),
        ]

        response = client.get("/electricity/evaluation")

        assert response.status_code == 200
        models = {m["model"]: m for m in response.json()["evaluation"]}

        lgb = models["lightgbm"]
        assert lgb["scored_count"] == 2
        assert lgb["mae"] == pytest.approx(100.0)
        assert lgb["rmse"] == pytest.approx(100.0)
        assert lgb["mape"] == pytest.approx(10.0)

        naive = models["naive_baseline"]
        assert naive["mae"] == pytest.approx(100.0)
        # RMSE punishes the single large error: sqrt((200^2 + 0)/2) > MAE
        assert naive["rmse"] == pytest.approx(141.4213562, rel=1e-6)

        # Unscored model sorts last rather than raising on a None comparison.
        assert models["seasonal_naive"]["mae"] is None
        assert models["seasonal_naive"]["pending_count"] == 1
        assert response.json()["evaluation"][-1]["model"] == "seasonal_naive"

        # Full record must stay unfiltered by date, and must not fetch raw rows.
        sql = cur.execute.call_args[0][0]
        assert "forecast_date" not in sql
        assert "GROUP BY model" in sql


def test_electricity_evaluation_no_data():
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        cur.fetchall.return_value = []

        response = client.get("/electricity/evaluation")

        assert response.status_code == 404


def test_electricity_predictions_error_pct():
    """error_pct is what the dashboard bands its badges on."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = _mock_cursor(mock_get_db)
        cur.fetchall.return_value = [
            (date(2026, 8, 22), 'lightgbm', 27000.0, 28000.0, datetime.now(timezone.utc)),
            (date(2026, 8, 23), 'lightgbm', 25000.0, None, datetime.now(timezone.utc)),
        ]

        response = client.get("/electricity/predictions?limit=2")

        assert response.status_code == 200
        rows = response.json()["predictions"]
        assert rows[0]["error"] == pytest.approx(1000.0)
        assert rows[0]["error_pct"] == pytest.approx(1000 / 28000 * 100)
        assert rows[1]["error"] is None
        assert rows[1]["error_pct"] is None