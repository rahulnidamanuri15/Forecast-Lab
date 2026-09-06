import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import date, datetime, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from conftest import wire_cursor

client = TestClient(app)

def test_predictions_endpoint_success():
    """Test that the predictions endpoint returns 200 with prediction data."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)

        # Six columns: `source` is last, so a row can be told apart from a
        # launch-backtest row.
        forecast_date = date.today()
        mock_cursor.fetchall.return_value = [
            (forecast_date, 'lightgbm', 15.5, None, datetime.now(timezone.utc), 'daily'),
            (forecast_date, 'naive_baseline', 12.3, 12.3, datetime.now(timezone.utc), 'backtest')
        ]

        # Deliberately unfiltered: the mock returns rows for both models and
        # ignores the WHERE clause, so asking for ?model=lightgbm here and then
        # asserting a naive_baseline row came back encoded the opposite of the
        # endpoint's contract. The model filter is asserted on the query itself
        # in test_predictions_model_filter_reaches_sql.
        response = client.get("/predictions?limit=5&scored_only=false")

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

        pred1 = data["predictions"][0]
        assert pred1["forecast_date"] == forecast_date.isoformat()
        assert pred1["model"] == "lightgbm"
        assert pred1["predicted_pm2_5"] == 15.5
        assert pred1["actual_pm2_5"] is None
        assert pred1["error"] is None
        assert "created_at" in pred1
        # 'daily' is renamed `verified` on the way out, same word /evaluation uses.
        assert pred1["source"] == "verified"

        pred2 = data["predictions"][1]
        assert pred2["forecast_date"] == forecast_date.isoformat()
        assert pred2["model"] == "naive_baseline"
        assert pred2["predicted_pm2_5"] == 12.3
        assert pred2["actual_pm2_5"] == 12.3
        assert pred2["error"] == 0.0
        assert "created_at" in pred2
        # The row the split exists for: a launch-backtest row is still returned,
        # but a caller can now see it is not part of the published record.
        assert pred2["source"] == "backtest"

        assert data["count"] == 2


def test_predictions_selects_the_source_column():
    """`source` has to be in the SELECT list, not just in the response dict.

    Asserted on the SQL as well as on the payload because the mock returns
    whatever tuple it is given: a handler that dropped the column from the query
    would still pass the payload assertions above once the mock was updated.
    """
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)
        mock_cursor.fetchall.return_value = []

        assert client.get("/predictions").status_code == 200
        sql, _ = mock_cursor.execute.call_args[0]
        assert "source" in sql

def test_predictions_endpoint_scored_only():
    """Test that the predictions endpoint filters scored predictions correctly."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)

        forecast_date = date.today()
        mock_cursor.fetchall.return_value = [
            (forecast_date, 'naive_baseline', 12.3, 12.3, datetime.now(timezone.utc), 'daily')
        ]

        response = client.get("/predictions?scored_only=true")

        # The mock ignores the WHERE clause, so the filter can only be verified
        # on the SQL that was actually sent.
        sql, params = mock_cursor.execute.call_args[0]
        assert "actual_pm2_5 IS NOT NULL" in sql

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

        pred = data["predictions"][0]
        assert pred["actual_pm2_5"] is not None
        assert pred["error"] == 0.0

def test_predictions_model_filter_reaches_sql():
    """?model=lightgbm must become a WHERE clause with the model as a bound
    parameter - not a value interpolated into the SQL, and not silently dropped."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)
        mock_cursor.fetchall.return_value = []

        assert client.get("/predictions?model=lightgbm").status_code == 200

        sql, params = mock_cursor.execute.call_args[0]
        assert "model = %s" in sql
        assert "lightgbm" in params
        assert "lightgbm" not in sql   # bound, never interpolated


@pytest.mark.parametrize("path", ["/predictions", "/electricity/predictions"])
def test_predictions_date_and_offset_filters(path):
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db, rows=[])
        res = client.get(f"{path}?start_date=2026-08-01&end_date=2026-08-15&offset=20&limit=10")
        assert res.status_code == 200
        sql, params = mock_cursor.execute.call_args[0]
        assert "forecast_date >= %s" in sql
        assert "forecast_date <= %s" in sql
        assert "2026-08-01" in params
        assert "2026-08-15" in params
        assert params[-2:] == [10, 20]  # LIMIT 10 OFFSET 20


@pytest.mark.parametrize("path", ["/predictions", "/electricity/predictions"])
def test_predictions_invalid_date_format(path):
    assert client.get(f"{path}?start_date=not-a-date").status_code == 400
    assert client.get(f"{path}?end_date=2026-13-45").status_code == 400
    res = client.get(f"{path}?start_date=2026-09-10&end_date=2026-09-01")
    assert res.status_code == 400
    assert "start_date cannot be after end_date" in res.json()["detail"]



def test_predictions_rejects_unknown_model():
    """An unallowlisted model is a 400 before any query runs - that allowlist is
    what makes the f-string WHERE clause in app.py safe."""
    response = client.get("/predictions?model=DROP TABLE predictions")
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/predictions", "/electricity/predictions"])
def test_source_filter_is_applied_before_the_limit(path):
    """?source=daily has to reach the WHERE clause, not just the payload.

    The dashboard asks for 15 published rows. Without this parameter the server
    spent the LIMIT on `ORDER BY forecast_date DESC` across both provenances and
    the page dropped the backtest rows client-side, so the panel headed "logged
    before actual outcomes were known" ran on whatever fraction of 15 the
    interleave happened to leave - roughly half, on the electricity side, whose
    backtest tail is ~2 weeks behind the daily record rather than a year.

    Asserted on the position in the SQL as well as on the clause: a filter that
    landed after the LIMIT would be the bug this fixes, spelled differently.
    """
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db, rows=[])

        assert client.get(f"{path}?limit=15&source=daily").status_code == 200

        sql, params = mock_cursor.execute.call_args[0]
        assert "source = %s" in sql
        assert "daily" in params
        assert "daily" not in sql          # bound, never interpolated
        assert sql.index("source = %s") < sql.index("LIMIT")


@pytest.mark.parametrize("path", ["/predictions", "/electricity/predictions"])
def test_unknown_source_is_a_400_not_an_empty_list(path):
    """?source=verified is the renamed value, not the stored one.

    A pass-through filter would answer 200 with zero rows, which a caller cannot
    tell apart from "nothing published yet" - so the allowlist that already
    guards `model` guards `source` too.
    """
    assert client.get(f"{path}?source=verified").status_code == 400
    assert client.get(f"{path}?source=daily' OR '1'='1").status_code == 400


def test_predictions_endpoint_no_data():
    """Test that the predictions endpoint handles missing prediction data."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db)

        mock_cursor.fetchall.return_value = []

        response = client.get("/predictions?model=lightgbm")

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 0
        assert data["count"] == 0


# The empty-result contract, both halves in one place. It was per-route and
# undocumented: these two answer 200 with an empty list while the six below 404,
# the same inconsistency class the project already closed for `days` validation
# (400 -> 422). The split is deliberate, so it is pinned rather than left to the
# next reader to guess at - a "consistency" fix in either direction fails here.
#
# The rule: a *filtered log* answers about the filter the caller composed, so no
# matching rows is 200 and `count: 0`. A *record* route returns the one current
# answer, so its absence is the pipeline not having produced one - a 404.
@pytest.mark.parametrize("path", ["/predictions", "/electricity/predictions"])
def test_a_filtered_log_answers_200_with_an_empty_list(path):
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, rows=[])

        response = client.get(f"{path}?model=lightgbm&limit=5&source=daily")

        assert response.status_code == 200
        assert response.json() == {"predictions": [], "count": 0}


@pytest.mark.parametrize("path", [
    "/history", "/leaderboard", "/evaluation",
    "/electricity/history", "/electricity/leaderboard", "/electricity/evaluation",
])
def test_a_record_route_404s_when_there_is_no_record(path):
    """Every route here reads its rows with fetchall, so one empty list covers all
    six. /forecast and /electricity/forecast are the same contract on fetchone and
    are covered in tests/test_forecast.py."""
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, rows=[])

        assert client.get(path).status_code == 404


def test_predictions_endpoint_database_error():
    """Test that the predictions endpoint handles database errors."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        response = client.get("/predictions?model=lightgbm")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data