"""The manual backfill escape hatch must be as guarded as the mirror path.

When the Grid-Sentinel CSV stalls, vericast/elec/backfill.py lets an operator
supply genuine NLDC PSP figures for the missing days. Operator input is the
least trustworthy data this repo ever accepts, so every entry is validated
before anything is written: one implausible value aborts the whole run, and a
future or duplicated date never reaches the upsert. These tests assert that
all-or-nothing property without a database.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from vericast.elec import backfill as bf


YESTERDAY = date(2026, 10, 7)


def test_parse_entry_accepts_both_shapes():
    assert bf.parse_entry("2026-09-30:24500") == (date(2026, 9, 30), 24500.0, None)
    assert bf.parse_entry("2026-09-30:24500.5:312.4") == (
        date(2026, 9, 30), 24500.5, 312.4)


def test_parse_entry_rejects_malformed_input():
    for bad in ["2026-09-30", "2026/09/30:24500", "2026-09-30:abc",
                "2026-09-30:24500:312.4:extra", "not-a-date:24500",
                "2026-09-30:24500:-3"]:
        with pytest.raises(ValueError):
            bf.parse_entry(bad)


def test_validate_sorts_and_accepts_a_clean_block():
    entries = bf.validate_entries(
        ["2026-10-01:25100", "2026-09-30:24500"], YESTERDAY)
    assert [day for day, _, _ in entries] == [date(2026, 9, 30), date(2026, 10, 1)]


def test_validate_aborts_the_whole_run_on_any_bad_entry():
    """One typo must not land while another aborts: nothing is returned at all."""
    with pytest.raises(RuntimeError, match="Refusing backfill"):
        bf.validate_entries(
            ["2026-09-30:24500", "2026-10-01:32419000"], YESTERDAY)  # kW slip


def test_validate_rejects_duplicates_future_and_pre_series_dates():
    with pytest.raises(RuntimeError, match="duplicate"):
        bf.validate_entries(
            ["2026-09-30:24500", "2026-09-30:24600"], YESTERDAY)
    with pytest.raises(RuntimeError, match="after yesterday"):
        bf.validate_entries(["2026-10-08:24500"], YESTERDAY)
    with pytest.raises(RuntimeError, match="predates"):
        bf.validate_entries(["2022-01-01:24500"], YESTERDAY)


def test_validate_warns_on_gaps_but_proceeds(capsys):
    entries = bf.validate_entries(
        ["2026-09-30:24500", "2026-10-02:25100"], YESTERDAY)
    assert len(entries) == 2
    assert "gap" in capsys.readouterr().out


def wire_main(monkeypatch):
    """Mock the network/DB boundary; return (insert_mock, temps)."""
    monkeypatch.setattr(bf, "DATABASE_URL", "postgresql://test")
    temps = {"2026-09-30": (30.0, 35.0), "2026-10-01": (31.0, 36.0)}
    monkeypatch.setattr(bf, "fetch_temperature", MagicMock(return_value=temps))
    insert = MagicMock()
    monkeypatch.setattr(bf, "insert_observations", insert)
    monkeypatch.setattr(bf.local_time, "yesterday", lambda tz=None: YESTERDAY)
    return insert


def test_main_writes_ingest_shaped_records(monkeypatch):
    insert = wire_main(monkeypatch)
    bf.main(["2026-09-30:24500", "2026-10-01:25100"])
    (records,), _ = insert.call_args
    assert records == [
        ("Maharashtra", "2026-09-30", 24500.0, None, 30.0, 35.0),
        ("Maharashtra", "2026-10-01", 25100.0, None, 31.0, 36.0),
    ]


def test_main_degrades_to_null_temps_when_the_archive_fails(monkeypatch, capsys):
    insert = wire_main(monkeypatch)
    monkeypatch.setattr(bf, "fetch_temperature",
                        MagicMock(side_effect=RuntimeError("archive down")))
    bf.main(["2026-09-30:24500"])
    (records,), _ = insert.call_args
    assert records == [("Maharashtra", "2026-09-30", 24500.0, None, None, None)]
    assert "NULL temperatures" in capsys.readouterr().out


def test_main_writes_nothing_when_validation_fails(monkeypatch):
    insert = wire_main(monkeypatch)
    with patch.object(bf, "fetch_temperature") as fetch, \
         pytest.raises(RuntimeError, match="Refusing backfill"):
        bf.main(["2026-09-30:24500", "2026-10-01:not-a-number"])
    assert not insert.called
    assert not fetch.called
