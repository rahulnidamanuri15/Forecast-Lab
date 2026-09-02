"""A Decimal reaching a value handler must not 500.

Every value column in vericast/schema.py is FLOAT today, so psycopg hands back
Python floats and the float() calls in app.py are no-ops. This file is the check
on the day that stops being true: a NUMERIC migration - or one applied to one
column and not its pair - makes psycopg return Decimal, and `abs(actual -
predicted)` across a Decimal and a float is a TypeError inside the row loop,
which the blanket handler turns into a 500 "Internal server error".

Mixed on purpose: all-Decimal arithmetic works fine, so a test that migrated
both columns in the mock would pass without the coercion and prove nothing. The
half-migrated pair is the case that breaks.

Serialization itself is not at risk - FastAPI's encoder handles Decimal - so the
two /history endpoints have no test here: they do no arithmetic, and their
float() calls exist only so all four handlers read the same way.
"""
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from conftest import wire_cursor

client = TestClient(app)


def test_predictions_survives_a_half_migrated_numeric_column():
    """predicted_pm2_5 as NUMERIC, actual_pm2_5 still FLOAT: `error` still computes."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = wire_cursor(mock_get_db)
        cur.fetchall.return_value = [
            (date(2026, 8, 28), 'lightgbm', Decimal("15.5"), 18.0,
             datetime.now(timezone.utc), 'daily'),
        ]

        response = client.get("/predictions?limit=1")

        assert response.status_code == 200
        row = response.json()["predictions"][0]
        assert row["predicted_pm2_5"] == pytest.approx(15.5)
        assert row["error"] == pytest.approx(2.5)


def test_electricity_predictions_survives_a_half_migrated_numeric_column():
    """Same, plus error_pct - which divides by the coerced actual."""
    with patch('app.get_db_connection') as mock_get_db:
        cur = wire_cursor(mock_get_db)
        cur.fetchall.return_value = [
            (date(2026, 8, 28), 'lightgbm', 27000.0, Decimal("28000.0"),
             datetime.now(timezone.utc), 'daily'),
        ]

        response = client.get("/electricity/predictions?limit=1")

        assert response.status_code == 200
        row = response.json()["predictions"][0]
        assert row["actual_demand_mw"] == pytest.approx(28000.0)
        assert row["error"] == pytest.approx(1000.0)
        assert row["error_pct"] == pytest.approx(1000 / 28000 * 100)
