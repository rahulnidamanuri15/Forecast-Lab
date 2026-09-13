"""Timing boundaries and artifact diagnostics, independent of upstream services."""
from datetime import date, datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as api
from vericast import artifacts, publication


@pytest.mark.parametrize("zone,cutoff", [
    ("UTC", datetime(2030, 1, 2, tzinfo=timezone.utc)),
    ("Asia/Kolkata", datetime(2030, 1, 1, 18, 30, tzinfo=timezone.utc)),
])
def test_publication_boundary(zone, cutoff):
    target = date(2030, 1, 2)
    for delta, source in ((timedelta(microseconds=-1), "daily"),
                          (timedelta(0), "nowcast"), (timedelta(days=4), "nowcast")):
        timing = publication.publication_timing(target, zone, cutoff + delta)
        assert timing["source"] == source
        assert timing["cutoff"] == cutoff
        assert timing["feature_as_of"] == target - timedelta(days=1)
        assert timing["horizon_days"] == 1
    with pytest.raises(ValueError, match="timezone-aware"):
        publication.publication_timing(target, zone, datetime(2030, 1, 1))


def test_batch_crossing_cutoff_is_refused(monkeypatch):
    before = datetime(2030, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
    after = before + timedelta(seconds=1)
    clock = MagicMock()
    clock.now.side_effect = [before, after]
    clock.combine = datetime.combine
    monkeypatch.setattr(publication, "datetime", clock)
    with pytest.raises(RuntimeError, match="cutoff passed"):
        publication.publish_predictions(MagicMock(), "insert", [("city",)],
                                        date(2030, 1, 2), "UTC")


@pytest.mark.parametrize("source", ["backtest", "nowcast"])
def test_unknown_issuance_does_not_invent_historical_timing(source):
    result = api.timing_payload(date(2030, 1, 2), None, source, "UTC")
    assert result["issued_at"] is None
    assert result["timing_status"] == ("backtest" if source == "backtest" else "unknown")
    assert result["provenance_consistent"] is None
    assert result["target_timezone"] == "UTC"
    assert result["horizon_days"] == 1


@pytest.fixture
def real_bundle(tmp_path):
    # Actual LightGBM serialization, not a mocked loader that accepts broken text.
    data = np.arange(40, dtype=float).reshape(20, 2)
    model = lgb.train({"objective": "regression", "verbosity": -1, "num_threads": 1,
                       "min_data_in_leaf": 2}, lgb.Dataset(data, label=data[:, 0]),
                      num_boost_round=2)
    path = tmp_path / "model.txt"
    artifacts.save_bundle(model, path, [date(2030, 1, 1), date(2030, 1, 2)], 2)
    return path


def test_bundle_only_artifact_is_authoritative(real_bundle):
    assert not real_bundle.exists()
    metadata = artifacts.model_metadata(real_bundle, 2)
    assert metadata["artifact_format"] == "bundle"
    assert metadata["bundle_version"] == 1
    assert metadata["training_window"]["rows"] == 2
    assert metadata["scope"] == "current_deployment"
    assert "model" not in metadata
    with pytest.raises(ValueError, match="feature count"):
        artifacts.model_metadata(real_bundle, 3)
    real_bundle.write_text("invalid legacy file", encoding="utf-8")
    assert artifacts.model_metadata(real_bundle, 2) == metadata


@pytest.mark.parametrize("window", [
    {"first": "nonsense", "last": "2030-01-02", "rows": 2},
    {"first": "2030-01-03", "last": "2030-01-02", "rows": 2},
    {"first": "2030-01-01", "last": "2030-01-02", "rows": True},
    {"first": "2030-01-01", "last": "2030-01-02", "rows": 3},
])
def test_bad_bundle_metadata_cannot_fall_back(real_bundle, window):
    path = Path(artifacts.bundle_path(real_bundle))
    payload = json.loads(path.read_text())
    artifacts.load_model(real_bundle).save_model(str(real_bundle))
    payload["window"] = window
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training window"):
        artifacts.model_metadata(real_bundle)
    with pytest.raises(ValueError, match="training window"):
        artifacts.load_model(real_bundle)


@pytest.mark.parametrize("domain,route,attribute", [
    ("pm25", "/diagnostics", "MODEL_PM25"),
    ("elec", "/electricity/diagnostics", "MODEL_ELEC"),
])
def test_diagnostics_use_bundle_and_fail_on_corruption(monkeypatch, real_bundle,
                                                       domain, route, attribute):
    module = importlib.import_module(f"vericast.{domain}.diagnose")
    monkeypatch.setattr(module, "MODEL_PATH", str(real_bundle))
    monkeypatch.setattr(module, "FEATURE_COLUMNS", ["a", "b"])
    cur = MagicMock()
    today = date(2030, 1, 1)
    monkeypatch.setattr(module.local_time, "today", lambda: today)
    if domain == "pm25":
        cur.fetchone.side_effect = [(today, 50), (today,), (50, 50),
                                    (today + timedelta(days=1), 50, None), (50, None)]
        models, value = ("lightgbm", "naive_baseline"), 50
    else:
        cur.fetchone.side_effect = [(today,), (today,), (today, 25000, 25000), (25000, None)]
        models, value = ("lightgbm", "naive_baseline", "seasonal_naive"), 25000
    cur.fetchall.return_value = [(model, value, "nowcast",
                                  datetime(2030, 1, 2, tzinfo=timezone.utc)) for model in models]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    connect = MagicMock()
    connect.return_value.__enter__.return_value = conn
    monkeypatch.setattr(module.psycopg, "connect", connect)
    monkeypatch.setattr(module, "send_alert", MagicMock())
    assert module.main() is True
    monkeypatch.setattr(api, attribute, str(real_bundle))
    monkeypatch.setattr(api, "PM25_FEATURE_COLUMNS" if domain == "pm25" else
                        "ELEC_FEATURE_COLUMNS", ["a", "b"])
    client = TestClient(api.app)
    assert client.get(route).json()["model_bundle"]["status"] == "ok"
    Path(artifacts.bundle_path(real_bundle)).write_text("{broken")
    response = client.get(route)
    assert response.status_code == 200
    assert response.json()["model_bundle"] == {
        "status": "error", "detail": "Model artifact missing or invalid"}
    cur.fetchone.side_effect = [(None, None)] if domain == "pm25" else [(None,)]
    assert module.main() is False
