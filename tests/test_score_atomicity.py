"""Exercise scoring rollback/retry using session-local PostgreSQL tables."""
from contextlib import contextmanager
from datetime import date
import importlib
import os

import psycopg
from psycopg.conninfo import conninfo_to_dict
import pytest

from vericast.schema import TABLES


@pytest.mark.parametrize("domain,prefix,key,predicted,actual,observed,value", [
    ("pm25", "", "city", "predicted_pm2_5", "actual_pm2_5", "pm2_5", 50.0),
    ("elec", "electricity_", "state", "predicted_demand_mw", "actual_demand_mw", "peak_demand_mw", 25000.0),
])
def test_metrics_failure_rolls_back_actual_and_retry_succeeds(
        monkeypatch, domain, prefix, key, predicted, actual, observed, value):
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL is required for PostgreSQL integration coverage")
    if conninfo_to_dict(dsn).get("host", "") not in ("", "localhost", "127.0.0.1", "::1"):
        pytest.skip("Use a local throwaway PostgreSQL database")
    module = importlib.import_module(f"vericast.{domain}.score")
    tables = [prefix + name for name in ("observations", "predictions", "model_performance")]
    location = module.CITY if domain == "pm25" else module.STATE
    day = date(2019, 1, 1)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Temporary tables shadow persistent names without changing any data
            # outside this session, including on failure or parallel test runs.
            for table in tables:
                cur.execute(TABLES[table].replace("CREATE TABLE IF NOT EXISTS", "CREATE TEMP TABLE"))
            cur.execute(f"INSERT INTO {tables[0]} ({key}, as_of, {observed}) VALUES (%s, %s, %s)",
                        (location, day, value))
            cur.execute(f"INSERT INTO {tables[1]} ({key}, forecast_date, model, {predicted}, source) "
                        "VALUES (%s, %s, 'atomicity_test', %s, 'daily')",
                        (location, day, value + 1))
        conn.commit()

        @contextmanager
        def connection_context(*args, **kwargs):
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        monkeypatch.setattr(module.psycopg, "connect", connection_context)
        monkeypatch.setattr(module, "DATABASE_URL", dsn)
        original_sql = module.UPSERT_PERF_SQL
        # A deterministic statement failure after SCORE_SQL attaches the actual.
        monkeypatch.setattr(module, "UPSERT_PERF_SQL", "SELECT 1 / 0")
        with pytest.raises(RuntimeError, match="rolling back"):
            module.score_pending_predictions()
        with conn.cursor() as cur:
            cur.execute(f"SELECT {actual} FROM {tables[1]} WHERE model = 'atomicity_test'")
            assert cur.fetchone()[0] is None
            cur.execute(f"SELECT COUNT(*) FROM {tables[2]}")
            assert cur.fetchone()[0] == 0
        conn.commit()
        monkeypatch.setattr(module, "UPSERT_PERF_SQL", original_sql)
        assert module.score_pending_predictions() == 1
        with conn.cursor() as cur:
            cur.execute(f"SELECT {actual} FROM {tables[1]} WHERE model = 'atomicity_test'")
            assert cur.fetchone()[0] == value
            cur.execute(f"SELECT mae FROM {tables[2]} WHERE model = 'atomicity_test'")
            assert cur.fetchone()[0] == pytest.approx(1.0)
