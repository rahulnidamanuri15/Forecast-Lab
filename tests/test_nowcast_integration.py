"""Real PostgreSQL upgrades and predictors -> scoring -> API provenance checks.

All writes are restricted to a random schema in a local disposable database.
No upstream ingestion, remote databases, or production rows are touched.
"""
from datetime import date, datetime, timedelta, timezone
import importlib
import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest
from fastapi.testclient import TestClient

import app as api
from vericast import publication, schema


@pytest.fixture
def isolated_db():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL required for PostgreSQL integration checks")
    if conninfo_to_dict(dsn).get("host") not in ("localhost", "127.0.0.1", "::1"):
        pytest.skip("Only a local disposable PostgreSQL instance is allowed")
    name = "nowcast_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(dsn, options=f"-c search_path={name}")
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


@pytest.mark.parametrize("configured_timeout", [None, 60000])
def test_api_pool_preserves_schema_and_enforces_timeout(isolated_db, monkeypatch, configured_timeout):
    options = conninfo_to_dict(isolated_db)["options"]
    if configured_timeout is not None:
        options += f" -c statement_timeout={configured_timeout}"
    dsn = make_conninfo(isolated_db, options=options)
    monkeypatch.setattr(api, "DATABASE_URL", dsn)
    with psycopg.connect(dsn) as conn:
        expected_schema = conn.execute("SELECT current_schema()").fetchone()[0]
    # Enter the real application lifespan so this tests pooled connections,
    # not get_db_connection's direct-connection fallback used by unit tests.
    with TestClient(api.app):
        assert api._pool is not None
        with api.get_db_connection() as conn:
            actual = conn.execute("SELECT current_schema(), "
                                  "current_setting('statement_timeout')").fetchone()
            assert actual == (expected_schema, "10s")
    assert api._pool is None
    assert api._under_lifespan is False


@pytest.mark.parametrize("legacy", [True, False])
def test_schema_upgrade_preserves_values_and_is_idempotent(isolated_db, monkeypatch, legacy):
    monkeypatch.setattr(schema, "DATABASE_URL", isolated_db)
    if legacy:
        with psycopg.connect(isolated_db) as conn:
            for name, ddl in schema.TABLES.items():
                ddl = ddl.replace(", model, source)", ", model)")
                if name == "model_performance":
                    ddl = ddl.replace("UNIQUE(city, score_date, model)", "UNIQUE(score_date, model)")
                conn.execute(ddl)
            conn.execute("CREATE UNIQUE INDEX model_performance_city_score_model_uidx "
                         "ON model_performance (city, score_date, model)")
    else:
        # New DDL with no migration marker yet; also exercise coexistence with
        # explicitly-labelled nowcasts before running the correction.
        with psycopg.connect(isolated_db) as conn:
            for ddl in schema.TABLES.values():
                conn.execute(ddl)
    for prefix, key, value_col, zone in (("", "city", "predicted_pm2_5", "UTC"),
            ("electricity_", "state", "predicted_demand_mw", "Asia/Kolkata")):
        target = date(2030, 1, 2)
        cutoff = publication.publication_timing(target, zone)["cutoff"]
        with psycopg.connect(isolated_db) as conn:
            for model, issued, source in (("early", cutoff - timedelta(microseconds=1), "daily"),
                    ("at_cutoff", cutoff, "daily"), ("late", cutoff + timedelta(days=3), "daily"),
                    ("unknown", None, "daily"), ("backtest", cutoff, "backtest")):
                conn.execute(f"INSERT INTO {prefix}predictions "
                             f"({key}, forecast_date, model, {value_col}, source, created_at) "
                             "VALUES ('test', %s, %s, 25000, %s, %s)",
                             (target, model, source, issued))
                conn.execute(f"INSERT INTO {prefix}model_performance "
                             f"({key}, score_date, model, mae, rmse, sample_size, source) "
                             "VALUES ('test', %s, %s, 3, 3, 1, %s)", (target, model, source))
            if not legacy:
                conn.execute(f"INSERT INTO {prefix}predictions "
                             f"({key}, forecast_date, model, {value_col}, source, created_at) "
                             "VALUES ('test', %s, 'early', 25001, 'nowcast', %s)", (target, cutoff))
    schema.create_tables()
    schema.create_tables()
    for prefix, key in (("", "city"), ("electricity_", "state")):
        with psycopg.connect(isolated_db) as conn:
            rows = conn.execute(f"SELECT model, source FROM {prefix}model_performance").fetchall()
            assert dict(rows) == {"early": "daily", "at_cutoff": "nowcast", "late": "nowcast",
                                  "unknown": "nowcast", "backtest": "backtest"}
            values = conn.execute(f"SELECT source, created_at FROM {prefix}predictions "
                                  "WHERE model = 'unknown'").fetchone()
            assert values == ("nowcast", None)
            # Former source-blind keys must no longer prevent overlapping records.
            conn.execute(f"INSERT INTO {prefix}model_performance "
                         f"({key}, score_date, model, mae, rmse, sample_size, source) "
                         "VALUES ('test', '2030-01-02', 'early', 4, 4, 1, 'nowcast')")
            conn.execute(f"INSERT INTO {prefix}model_performance "
                         f"({key}, score_date, model, mae, rmse, sample_size, source) "
                         "VALUES ('test', '2030-01-02', 'early', 5, 5, 10, 'backtest')")
            column = "predicted_pm2_5" if not prefix else "predicted_demand_mw"
            assert conn.execute(f"SELECT {column} FROM {prefix}predictions "
                                "WHERE model = 'at_cutoff'").fetchone()[0] == 25000


@pytest.mark.parametrize("domain,lag", [("pm25", 1), ("pm25", 2), ("elec", 2), ("elec", 4)])
def test_delayed_pipeline_scoring_and_api_are_isolated(isolated_db, monkeypatch, domain, lag):
    monkeypatch.setattr(schema, "DATABASE_URL", isolated_db)
    schema.create_tables()
    predict = importlib.import_module(f"vericast.{domain}.predict")
    features = importlib.import_module(f"vericast.{domain}.features")
    score = importlib.import_module(f"vericast.{domain}.score")
    for module in (predict, features, score):
        monkeypatch.setattr(module, "DATABASE_URL", isolated_db)
    monkeypatch.setattr(api, "DATABASE_URL", isolated_db)
    today = date(2030, 1, 10)
    as_of = today - timedelta(days=lag)
    target = as_of + timedelta(days=1)
    issued_at = datetime(2030, 1, 10, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(predict.local_time, "today", lambda *a, **kw: today)
    clock = type("Clock", (), {"now": staticmethod(lambda tz: issued_at),
                               "combine": staticmethod(datetime.combine)})
    monkeypatch.setattr(publication, "datetime", clock)
    prefix, key, location, observed, predicted, actual, zone, base = (
        ("", "city", predict.CITY, "pm2_5", "predicted_pm2_5", "actual_pm2_5", "UTC", "")
        if domain == "pm25" else
        ("electricity_", "state", predict.STATE, "peak_demand_mw", "predicted_demand_mw",
         "actual_demand_mw", "Asia/Kolkata", "/electricity"))
    value = 50 if domain == "pm25" else 25000
    with psycopg.connect(isolated_db) as conn:
        for n in range(40):
            day = as_of - timedelta(days=39 - n)
            if domain == "pm25":
                conn.execute("INSERT INTO observations (city, as_of, pm2_5, pm10, "
                             "temperature_2m_mean, wind_speed_10m_max, precipitation_sum) "
                             "VALUES (%s, %s, %s, 80, 30, 10, 0)", (location, day, value))
            else:
                conn.execute("INSERT INTO electricity_observations (state, as_of, peak_demand_mw, "
                             "energy_met_mu, temperature_2m_mean, temperature_2m_max) "
                             "VALUES (%s, %s, %s, 480, 30, 36)", (location, day, value))
    features.engineer_features()
    assert predict.make_daily_prediction() is True
    with psycopg.connect(isolated_db) as conn:
        rows = conn.execute(f"SELECT forecast_date, model, source, created_at, {predicted} "
                            f"FROM {prefix}predictions ORDER BY model").fetchall()
        assert len(rows) == (2 if domain == "pm25" else 3)
        assert all(row[0] == target and row[2:4] == ("nowcast", issued_at) for row in rows)
    assert predict.make_daily_prediction() is True
    with psycopg.connect(isolated_db) as conn:
        assert conn.execute(f"SELECT forecast_date, model, source, created_at, {predicted} "
                            f"FROM {prefix}predictions ORDER BY model").fetchall() == rows
        # Same model/target in all three provenances, deliberately different errors.
        cutoff = publication.publication_timing(target, zone)["cutoff"]
        for source, estimate, timestamp in (("daily", value + 10, cutoff - timedelta(seconds=1)),
                                             ("backtest", value + 20, issued_at)):
            conn.execute(f"INSERT INTO {prefix}predictions "
                         f"({key}, forecast_date, model, {predicted}, source, created_at) "
                         "VALUES (%s, %s, 'naive_baseline', %s, %s, %s)",
                         (location, target, estimate, source, timestamp))
        conn.execute(f"UPDATE {prefix}predictions SET {actual} = %s WHERE source = 'backtest'", (value,))
        conn.execute(f"INSERT INTO {prefix}observations ({key}, as_of, {observed}) VALUES (%s, %s, %s)",
                     (location, target, value + 1))
    assert score.score_pending_predictions() == len(rows) + 1
    assert score.score_pending_predictions() == 0
    client = TestClient(api.app)
    latest = client.get(f"{base}/forecast?model=naive_baseline&source=latest")
    assert latest.status_code == 200
    payload = latest.json()
    assert payload["source"] == "nowcast"
    assert payload["publication_source"] == "nowcast"
    assert payload["timing_status"] == "delayed_estimate"
    assert payload["status"] == "verified"  # scoring status is independent
    assert payload["provenance_consistent"] is True
    assert payload["feature_as_of"] == as_of.isoformat()
    assert payload["horizon_days"] == 1
    assert client.get(f"{base}/forecast?model=naive_baseline&source=daily").json()["source"] == "verified"
    for source, alias, mae in (("daily", "verified", 9), ("nowcast", "nowcast", 1)):
        response = client.get(f"{base}/leaderboard?source={source}")
        assert response.status_code == 200
        assert response.json()["source"] == alias
        entry = next(r for r in response.json()["leaderboard"] if r["model"] == "naive_baseline")
        assert entry["mae"] == mae
        page = client.get(f"{base}/predictions?source={source}&limit=1&scored_only=true").json()
        assert page["count"] == 1
        assert page["predictions"][0]["source"] == alias
    evaluation = client.get(f"{base}/evaluation").json()["evaluation"]
    entry = next(r for r in evaluation if r["model"] == "naive_baseline")
    assert "mae" not in entry
    assert [entry[s]["mae"] for s in ("verified", "nowcast", "backtest")] == [9, 1, 20]
    for route in ("forecast", "leaderboard", "predictions"):
        assert client.get(f"{base}/{route}?source=invalid").status_code == 400
    with psycopg.connect(isolated_db) as conn:
        conn.execute(f"UPDATE {prefix}observations SET {observed} = %s WHERE as_of = %s", (value + 2, target))
    assert score.score_pending_predictions() == len(rows) + 1
    evaluation = client.get(f"{base}/evaluation").json()["evaluation"]
    entry = next(r for r in evaluation if r["model"] == "naive_baseline")
    assert [entry[s]["mae"] for s in ("verified", "nowcast", "backtest")] == [8, 2, 20]


def test_cutoff_crossing_rolls_back_real_batch(isolated_db, monkeypatch):
    monkeypatch.setattr(schema, "DATABASE_URL", isolated_db)
    schema.create_tables()
    times = iter([datetime(2030, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
                  datetime(2030, 1, 2, tzinfo=timezone.utc)])
    clock = type("Clock", (), {"now": staticmethod(lambda tz: next(times)),
                               "combine": staticmethod(datetime.combine)})
    monkeypatch.setattr(publication, "datetime", clock)
    insert = ("INSERT INTO predictions (city, forecast_date, predicted_pm2_5, model, source, created_at) "
              "VALUES (%s, %s, %s, %s, %s, %s)")
    with pytest.raises(RuntimeError, match="cutoff passed"):
        with psycopg.connect(isolated_db) as conn:
            publication.publish_predictions(conn.cursor(), insert,
                [("Nagpur", date(2030, 1, 2), 50, "lightgbm"),
                 ("Nagpur", date(2030, 1, 2), 51, "naive_baseline")], date(2030, 1, 2), "UTC")
    with psycopg.connect(isolated_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
