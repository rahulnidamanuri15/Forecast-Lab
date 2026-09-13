"""Regression coverage for the repository integrity review."""
import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vericast import artifacts, gate
from vericast.publication import require_advance_forecast

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("zone,cutoff", [
    ("UTC", datetime(2030, 1, 2, tzinfo=timezone.utc)),
    ("Asia/Kolkata", datetime(2030, 1, 1, 18, 30, tzinfo=timezone.utc)),
])
def test_issuance_cutoff(zone, cutoff):
    day = date(2030, 1, 2)
    require_advance_forecast(day, zone, cutoff - timedelta(microseconds=1))
    for now in (cutoff, cutoff + timedelta(days=1)):
        with pytest.raises(RuntimeError, match="cutoff passed"):
            require_advance_forecast(day, zone, now)


def test_naive_issuance_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        require_advance_forecast(date(2030, 1, 2), "UTC", datetime(2030, 1, 1))


def test_database_handlers_are_synchronous():
    tree = ast.parse((ROOT / "app.py").read_text())
    names = {"get_forecast", "get_leaderboard", "get_predictions", "get_evaluation",
             "get_history", "health", "electricity_health", "get_electricity_forecast",
             "get_electricity_predictions", "get_electricity_evaluation",
             "get_electricity_leaderboard", "get_electricity_history"}
    handlers = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(handlers) == len(names)
    assert all(isinstance(node, ast.FunctionDef) for node in handlers)


@pytest.mark.parametrize("domain", ["pm25", "elec"])
def test_publish_sql_does_not_replace_existing_forecasts(domain):
    tree = ast.parse((ROOT / "vericast" / domain / "predict.py").read_text())
    statements = [node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and "INSERT INTO" in node.value]
    assert statements
    for sql in statements:
        assert "DO NOTHING" in sql
        assert "DO UPDATE" not in sql


@pytest.mark.parametrize("rows", [0, 24, 50, 89])
def test_small_dataset_cannot_promote(monkeypatch, rows):
    train = MagicMock()
    monkeypatch.setattr(gate.lgb, "train", train)
    assert not gate.challenger_ships(np.ones((rows, 2)), np.ones(rows), {}, 1, 0)
    train.assert_not_called()


def test_promotion_requires_improvement():
    assert gate.MAX_BASELINE_RATIO < 1


def test_bundle_replacement_failure_keeps_old_generation(tmp_path, monkeypatch):
    path = str(tmp_path / "model.txt")
    model = MagicMock()
    model.model_to_string.return_value = "old-model"
    monkeypatch.setattr(artifacts.lgb, "Booster", MagicMock())
    artifacts.save_bundle(model, path, [date(2030, 1, 1)], 1)
    before = artifacts.read_bundle(path)
    model.model_to_string.return_value = "new-model"

    def fail_replace(*args):
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError):
        artifacts.save_bundle(model, path, [date(2030, 1, 2)], 1)
    assert artifacts.read_bundle(path) == before
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_bundle_does_not_fall_back(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("legacy-model")
    Path(artifacts.bundle_path(path)).write_text('{"version":')
    with pytest.raises(ValueError):
        artifacts.load_model(path)
