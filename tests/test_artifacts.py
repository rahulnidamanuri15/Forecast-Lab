"""read_bundle validation branches — the corrupt-production-bundle failure mode.

gate.read_training_window degrades an invalid bundle to "unknown window", and
artifacts.model_metadata/load_model refuse to fall back silently. These branches
had no dedicated coverage: the only bundle exercised in tests was a valid one.
"""
import json
import os

import pytest

from vericast import artifacts


def _write(path, payload):
    with open(artifacts.bundle_path(str(path)), "w", encoding="utf-8") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)


def test_missing_bundle_reads_as_none(tmp_path):
    assert artifacts.read_bundle(str(tmp_path / "absent.txt")) is None


def test_non_json_bundle_is_invalid(tmp_path):
    p = str(tmp_path / "m.txt")
    _write(p, "{not json")
    with pytest.raises(ValueError):
        artifacts.read_bundle(p)


@pytest.mark.parametrize("payload", [
    {},
    {"version": 2, "model": "x", "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 2}},
    {"version": 1, "model": "", "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 2}},
    {"version": 1, "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 2}},
    {"version": 1, "model": "x"},
    {"version": 1, "model": "x", "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 0}},
    {"version": 1, "model": "x", "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": "2"}},
    {"version": 1, "model": "x", "window": {"first": "2026-01-03", "last": "2026-01-02", "rows": 2}},
    {"version": 1, "model": "x", "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 5}},
    {"version": 1, "model": "x", "window": {"first": "not-a-date", "last": "2026-01-02", "rows": 1}},
])
def test_invalid_bundles_raise(tmp_path, payload):
    p = str(tmp_path / "m.txt")
    _write(p, payload)
    with pytest.raises(ValueError):
        artifacts.read_bundle(p)


def test_corrupt_bundle_window_degrades_to_none_not_raise(tmp_path):
    """gate.read_training_window promises the retrain never crashes on metadata."""
    from vericast import gate
    p = str(tmp_path / "m.txt")
    _write(p, {"version": 1, "model": "", "window": {}})
    assert gate.read_training_window(p) is None


def test_model_exists_covers_bundle_and_legacy(tmp_path):
    p = tmp_path / "m.txt"
    assert artifacts.model_exists(str(p)) is False
    p.write_text("legacy", encoding="utf-8")
    assert artifacts.model_exists(str(p)) is True
    os.remove(str(p))
    _write(str(p), {"version": 1, "model": "x",
                    "window": {"first": "2026-01-01", "last": "2026-01-01", "rows": 1}})
    assert artifacts.model_exists(str(p)) is True
