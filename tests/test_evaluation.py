import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from conftest import wire_cursor
from vericast import local_time

CITY = os.getenv("CITY", "Nagpur")

client = TestClient(app)

# /evaluation aggregates in SQL and groups by (model, source), so a mocked row is
# one per model *per provenance*: (model, source, scored, pending, mae, rmse).
# The arithmetic is Postgres's and is checked against real rows in
# test_feature_alignment.py; what these tests cover is the payload shape, the
# provenance split, the window/full-record branch, and the sort.
#
# 'daily' surfaces as `verified` and 'backtest' as `backtest` - see app.PROVENANCE.


def test_evaluation_endpoint_success():
    """200, one entry per model, metrics nested under their provenance."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db, [
            ('lightgbm', 'daily', 2, 1, 0.35, 0.3807886553),
            ('naive_baseline', 'daily', 1, 1, 0.2, 0.2),
        ])

        response = client.get("/evaluation?days=7")

        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert len(data["evaluation"]) == 2
        # The note is the reason the split exists; a client quoting one number
        # needs to know which one it is quoting.
        assert "verified" in data["note"]

        lightgbm_eval = next(e for e in data["evaluation"] if e["model"] == "lightgbm")
        assert lightgbm_eval["window_days"] == 7
        assert lightgbm_eval["verified"]["scored_count"] == 2
        assert lightgbm_eval["verified"]["pending_count"] == 1
        assert abs(lightgbm_eval["verified"]["mae"] - 0.35) < 0.001
        assert abs(lightgbm_eval["verified"]["rmse"] - 0.381) < 0.01
        # Same payload shape as /electricity/evaluation, description included.
        assert lightgbm_eval["description"]
        # Nothing seeded a backtest row for this model, so there is no block at
        # all - not a zero-filled one that reads as "measured, and it was 0".
        assert "backtest" not in lightgbm_eval

        naive_eval = next(e for e in data["evaluation"] if e["model"] == "naive_baseline")
        assert naive_eval["window_days"] == 7
        assert naive_eval["verified"]["scored_count"] == 1
        assert abs(naive_eval["verified"]["mae"] - 0.2) < 0.001
        assert abs(naive_eval["verified"]["rmse"] - 0.2) < 0.001

        # The windowed branch must bind the app-timezone "today" and the day
        # count. Asserted on the bound parameters rather than by grepping the SQL
        # for "forecast_date": a substring check passes on any query that merely
        # mentions the column, including one that selects it and never filters.
        _, params = mock_cursor.execute.call_args[0]
        assert params == (CITY, local_time.today(), 7)


def test_evaluation_splits_provenance_without_averaging():
    """The whole point of `source`: a backtest row must not move the verified MAE.

    Both blocks appear under one model entry, each with its own metrics, and
    there is deliberately no combined figure anywhere in the payload for a
    reader to quote by mistake.
    """
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, [
            ('lightgbm', 'daily', 4, 1, 8.0, 9.0),
            ('lightgbm', 'backtest', 300, 0, 2.0, 3.0),
        ])

        response = client.get("/evaluation")

        assert response.status_code == 200
        entries = response.json()["evaluation"]
        assert len(entries) == 1
        entry = entries[0]

        assert entry["verified"] == {
            "scored_count": 4, "pending_count": 1, "mae": 8.0, "rmse": 9.0}
        assert entry["backtest"] == {
            "scored_count": 300, "pending_count": 0, "mae": 2.0, "rmse": 3.0}
        # No top-level metric key: the flat shape this replaced averaged the two
        # into one headline number, which is the bug.
        assert "mae" not in entry and "scored_count" not in entry


def test_evaluation_reports_unknown_source_under_its_own_name():
    """A source nobody planned for still has to appear, or the counts stop adding up."""
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, [('lightgbm', 'shadow', 3, 0, 1.5, 2.0)])

        response = client.get("/evaluation")

        assert response.status_code == 200
        entry = response.json()["evaluation"][0]
        assert entry["shadow"]["scored_count"] == 3


def test_evaluation_endpoint_no_data():
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, [])

        response = client.get("/evaluation?days=7")

        assert response.status_code == 404
        assert "No predictions found in that window" in response.json()["detail"]


def test_evaluation_endpoint_database_error():
    with patch('app.get_db_connection') as mock_get_db:
        mock_get_db.side_effect = Exception("Database connection failed")

        response = client.get("/evaluation?days=7")

        assert response.status_code == 500
        assert "detail" in response.json()


def test_evaluation_full_record_omits_window():
    """No days= means every prediction ever published, and window_days is null."""
    with patch('app.get_db_connection') as mock_get_db:
        mock_cursor = wire_cursor(mock_get_db, [
            ('lightgbm', 'daily', 2, 0, 0.35, 0.3807886553),
        ])

        response = client.get("/evaluation")

        assert response.status_code == 200
        entry = response.json()["evaluation"][0]
        assert entry["window_days"] is None
        assert entry["verified"]["scored_count"] == 2

        # The full-record branch takes no window, so CITY is the only bound
        # parameter. Same reason as above for asserting on params, not on the SQL
        # text: `"forecast_date" not in sql` would break the moment the SELECT
        # list mentions the column, while still passing a query that filtered on
        # a different date expression.
        _, params = mock_cursor.execute.call_args[0]
        assert params == (CITY,)


def test_evaluation_rejects_negative_days():
    """days=-1 is a client error, not a silently-empty window.

    422, not 400: `days` is validated by Query(ge=0) so that every endpoint
    taking a `days` rejects a bad one the same way - /history?days=0 already
    answered 422 while this answered 400.
    """
    assert client.get("/evaluation?days=-1").status_code == 422


def test_evaluation_sorts_unscored_models_last():
    """A model with no scored rows must not blow up the sort."""
    with patch('app.get_db_connection') as mock_get_db:
        # Both models are entirely pending -> two None maes.
        wire_cursor(mock_get_db, [
            ('lightgbm', 'daily', 0, 1, None, None),
            ('naive_baseline', 'daily', 0, 1, None, None),
        ])

        response = client.get("/evaluation?days=7")

        assert response.status_code == 200
        entries = response.json()["evaluation"]
        assert all(e["verified"]["mae"] is None for e in entries)


def test_evaluation_sorts_on_verified_mae():
    """Ordering follows the verified figure, and a backtest-only model still sorts.

    The unscored model has to land last rather than raising on None < None, and a
    model whose only metric is a backtest one falls back to it instead of being
    treated as unscored.
    """
    with patch('app.get_db_connection') as mock_get_db:
        wire_cursor(mock_get_db, [
            ('unscored', 'daily', 0, 2, None, None),
            ('worse', 'daily', 5, 0, 9.0, 10.0),
            ('backtest_only', 'backtest', 40, 0, 4.0, 5.0),
            ('best', 'daily', 5, 0, 1.0, 2.0),
        ])

        response = client.get("/evaluation")

        assert response.status_code == 200
        assert [e["model"] for e in response.json()["evaluation"]] == [
            "best", "backtest_only", "worse", "unscored"]
