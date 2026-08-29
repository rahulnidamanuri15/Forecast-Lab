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
        # sample_size 1 because /leaderboard now filters on source = 'daily', and a
        # daily row is one scored day against a UNIQUE(city, forecast_date, model)
        # table. The backtest-seeded rows (sample_size in the hundreds) are exactly
        # what that filter exists to exclude: they carry the newest score_date on a
        # re-run and would otherwise become the published leaderboard.
        mock_cursor.fetchall.return_value = [
            ('lightgbm', 2.5, 3.2, 1, date(2026, 8, 17)),
            ('naive_baseline', 3.1, 4.0, 1, date(2026, 8, 17))
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
        assert data["leaderboard"][0]["sample_size"] == 1
        assert data["leaderboard"][0]["as_of"] == date(2026, 8, 17).isoformat()

        # Check that naive_baseline comes second
        assert data["leaderboard"][1]["model"] == "naive_baseline"
        assert data["leaderboard"][1]["mae"] == 3.1
        assert data["leaderboard"][1]["rmse"] == 4.0
        assert data["leaderboard"][1]["sample_size"] == 1
        assert data["leaderboard"][1]["as_of"] == date(2026, 8, 17).isoformat()

        # The filter itself, asserted on the SQL: the mock returns whatever it is
        # given, so nothing above would fail if the WHERE clause were dropped.
        sql = mock_cursor.execute.call_args[0][0]
        assert "source = 'daily'" in sql

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


# The other half of the provenance contract. The tests above assert the *reader*
# filters on source = 'daily'; these assert the *writers* label. A DEFAULT applies
# to inserts only, so without this line in the DO UPDATE branch a daily score
# landing on a score_date a backtest already wrote keeps source = 'backtest' and is
# filtered out of the leaderboard permanently. String assertions rather than a DB
# round-trip: the omission is in the SQL text, so that is where it is caught, and
# these run in CI with or without a database.
@pytest.mark.parametrize("module_path,attr", [
    ("vericast.pm25.score", "UPSERT_PERF_SQL"),
    ("vericast.elec.score", "UPSERT_PERF_SQL"),
    # The prediction writers need the same line for the same reason, one table
    # down: /evaluation splits `predictions` on source, so a real forecast landing
    # on a date the launch backtest seeded would keep source = 'backtest' and never
    # count towards the published-then-verified record.
    ("vericast.elec.predict", "UPSERT_SQL"),
])
def test_upsert_relabels_source_on_conflict(module_path, attr):
    import importlib

    sql = getattr(importlib.import_module(module_path), attr)
    _assert_forces_daily(sql, f"{module_path}.{attr}")


def test_pm25_predict_upsert_relabels_source_on_conflict():
    """vericast/pm25/predict.py builds its upsert inside make_daily_prediction(),
    so there is no module-level constant to import - read the source instead."""
    import inspect

    from vericast.pm25 import predict

    source = inspect.getsource(predict.make_daily_prediction)
    assert "INSERT INTO predictions" in source, (
        "pm25 predict no longer inserts into predictions here; this test needs "
        "repointing at wherever the upsert moved to")
    _assert_forces_daily(source, "vericast.pm25.predict.make_daily_prediction")


def _assert_forces_daily(sql, label):
    on_conflict = sql.split("DO UPDATE SET", 1)
    assert len(on_conflict) == 2, f"{label} has no DO UPDATE branch"
    assert "source = 'daily'" in on_conflict[1], (
        f"{label} does not reset source on conflict; a daily write overwriting a "
        f"backtest row would stay labelled 'backtest'"
    )


# The mirror of the above: the backtest seeders must NOT be able to overwrite a
# verified actual. Their DO UPDATE is guarded by a WHERE on the existing row's
# source, so a re-run refreshes its own rows and skips every daily one. Without
# it, a second run recomputes actual_pm2_5 from the backtest's own data on every
# overlapping date - silently replacing a genuinely verified observation.
@pytest.mark.parametrize("script,table,actual_col", [
    ("experiments/save_backtest_results.py", "predictions", "actual_pm2_5"),
    ("experiments/save_elec_backtest_results.py", "electricity_predictions",
     "actual_demand_mw"),
])
def test_backtest_seeder_cannot_overwrite_a_daily_row(script, table, actual_col):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), script)
    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    assert "VALUES (%s, %s, %s, %s, %s, 'backtest')" in source, (
        f"{script} no longer writes the 'backtest' literal; its rows would default "
        f"to 'daily' and be counted as published-then-verified")
    assert f"WHERE {table}.source = 'backtest'" in source, (
        f"{script}'s DO UPDATE has no source guard; a re-run would overwrite a "
        f"verified actual with a backtest-computed one")
    assert f"{actual_col} = EXCLUDED" not in source, (
        f"{script} updates the actual on conflict; that is the data loss the "
        f"guard above exists to prevent")
