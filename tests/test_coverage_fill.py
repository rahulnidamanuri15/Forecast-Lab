"""Fill coverage gaps that CI enforces (fail-under=70) without needing a live DB.

Targets the modules that sit at 0-60% in CI: both leakage_test modules,
gate.demo/sidecar helpers, local_time, vericast/__init__ helpers,
publication and artifacts edge cases. All DB/network access is mocked.
"""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pytest


# ---------------------------------------------------------------- leakage mismatch
def test_elec_mismatch_branches():
    from vericast.elec import leakage_test as lt

    assert lt.mismatch(None, None) is False
    assert lt.mismatch(None, 1.0) is True
    assert lt.mismatch(1.0, None) is True
    assert lt.mismatch(1.0, 1.0 + 1e-10) is False
    assert lt.mismatch(1.0, 2.0) is True


def test_pm25_mismatch_branches():
    from vericast.pm25 import leakage_test as lt

    assert lt.mismatch(None, None) is False
    assert lt.mismatch(None, 1.0) is True
    assert lt.mismatch(1.0, None) is True
    assert lt.mismatch(1.0, 1.0) is False
    assert lt.mismatch(1.0, 1.5) is True


def _mock_conn(mock_connect):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    return mock_conn, mock_cur


def _elec_pass_rows(n=35):
    from vericast.elec.leakage_test import COOLING_BASE, FEAT_COLS, OBS_COLS

    base = date(2026, 1, 1)
    obs = {}
    for i in range(n):
        d = base + timedelta(days=i)
        obs[d] = {
            "peak_demand_mw": 25000.0 + (i % 5) * 100.0,
            "energy_met_mu": 500.0,
            "temperature_2m_mean": 30.0,
            "temperature_2m_max": 35.0,
        }
    obs_rows = [(d, *(obs[d][c] for c in OBS_COLS)) for d in sorted(obs)]
    feat_rows = []
    for d in sorted(obs):
        today = obs[d]
        feat = {}
        lag_map = [("demand_lag_1", "peak_demand_mw", 1),
                   ("demand_lag_2", "peak_demand_mw", 2),
                   ("demand_lag_6", "peak_demand_mw", 6),
                   ("temp_lag_1", "temperature_2m_mean", 1)]
        for fcol, ocol, back in lag_map:
            prev = obs.get(d - timedelta(days=back))
            feat[fcol] = None if prev is None else prev[ocol]
        rolls = [("demand_roll_7_mean", "peak_demand_mw", 7, "mean"),
                 ("demand_roll_7_max", "peak_demand_mw", 7, "max"),
                 ("demand_roll_30_mean", "peak_demand_mw", 30, "mean"),
                 ("temp_roll_7", "temperature_2m_mean", 7, "mean")]
        for fcol, ocol, days, agg in rolls:
            window = [obs.get(d - timedelta(days=x)) for x in range(days)]
            values = [w[ocol] for w in window if w is not None and w[ocol] is not None]
            if len(values) != days:
                feat[fcol] = None
            else:
                feat[fcol] = sum(values) / days if agg == "mean" else max(values)
        feat["temperature_2m_mean"] = today["temperature_2m_mean"]
        feat["temperature_2m_max"] = today["temperature_2m_max"]
        feat["cooling_degree_days"] = max(0.0, today["temperature_2m_mean"] - COOLING_BASE)
        feat["day_of_week"] = d.weekday()
        feat["month"] = d.month
        feat["is_weekend"] = int(d.weekday() >= 5)
        feat_rows.append((d, *(feat[c] for c in FEAT_COLS)))
    return obs_rows, feat_rows


def _pm25_pass_rows(n=35):
    from vericast.pm25.leakage_test import FEAT_COLS, OBS_COLS

    base = date(2026, 1, 1)
    obs = {}
    for i in range(n):
        d = base + timedelta(days=i)
        obs[d] = {"pm2_5": 50.0, "pm10": 80.0, "temperature_2m_mean": 25.0,
                  "wind_speed_10m_max": 10.0, "precipitation_sum": 2.0}
    obs_rows = [(d, *(obs[d][c] for c in OBS_COLS)) for d in sorted(obs)]
    feat_rows = []
    for d in sorted(obs):
        today = obs[d]
        feat = {}
        for fcol, ocol in [("pm2_5_lag_1", "pm2_5"), ("pm10_lag_1", "pm10"),
                           ("temperature_lag_1", "temperature_2m_mean"),
                           ("wind_speed_lag_1", "wind_speed_10m_max"),
                           ("precipitation_lag_1", "precipitation_sum")]:
            prev = obs.get(d - timedelta(days=1))
            feat[fcol] = None if prev is None else prev[ocol]
        for fcol, ocol, days in [("pm2_5_roll_7", "pm2_5", 7),
                                 ("pm2_5_roll_30", "pm2_5", 30),
                                 ("pm10_roll_7", "pm10", 7),
                                 ("pm10_roll_30", "pm10", 30)]:
            window = [obs.get(d - timedelta(days=x)) for x in range(days)]
            values = [w[ocol] for w in window if w is not None and w[ocol] is not None]
            feat[fcol] = sum(values) / days if len(values) == days else None
        for c in ["temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"]:
            feat[c] = today[c]
        feat["day_of_week"] = d.weekday()
        feat["month"] = d.month
        feat["is_weekend"] = d.weekday() >= 5
        feat_rows.append((d, *(feat[c] for c in FEAT_COLS)))
    return obs_rows, feat_rows


def test_elec_leakage_pass_and_fail(capsys):
    from vericast.elec import leakage_test as lt

    obs_rows, feat_rows = _elec_pass_rows()
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows, feat_rows]
        assert lt.run_leakage_test() is True
    # corrupt the last row across every invariant family so all error
    # branches (same-day, cdd, calendar, lags, rolls) execute.
    from vericast.elec.leakage_test import FEAT_COLS
    bad_rows = []
    for r in feat_rows:
        d, vals = r[0], list(r[1:])
        if d == feat_rows[-1][0]:
            m = dict(zip(FEAT_COLS, vals))
            for k in list(m):
                if isinstance(m[k], (int, float)) and m[k] is not None:
                    m[k] = m[k] + 9999.0
            # calendar columns need explicit wrong values (numeric shift
            # already breaks them, but flip weekend flag deterministically)
            m["day_of_week"] = (m["day_of_week"] + 3) % 7 if m["day_of_week"] is not None else 0
            m["month"] = 12 if m["month"] != 12 else 1
            # opposite of the truth so bool() mismatches even on weekdays
            m["is_weekend"] = 1 if d.weekday() < 5 else 0
            vals = [m[c] for c in FEAT_COLS]
        bad_rows.append((d, *vals))
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows, bad_rows]
        assert lt.run_leakage_test() is False
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_elec_leakage_missing_obs_and_gap(capsys):
    from vericast.elec import leakage_test as lt

    obs_rows, feat_rows = _elec_pass_rows(n=10)
    # drop one obs day so a feat row is orphaned; also truncate obs to force
    # NULL-lag and incomplete-window branches on early rows
    orphan_day = feat_rows[5][0]
    obs_rows_gap = [r for r in obs_rows if r[0] != orphan_day]
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows_gap, feat_rows]
        assert lt.run_leakage_test() is False
    assert "missing in electricity_observations" in capsys.readouterr().out


def test_elec_leakage_requires_db():
    from vericast.elec import leakage_test as lt

    import os
    with patch.object(lt, "DATABASE_URL", None), \
         patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            lt.run_leakage_test()


def test_pm25_leakage_pass_and_fail(capsys):
    from vericast.pm25 import leakage_test as lt
    from vericast.pm25.leakage_test import FEAT_COLS

    obs_rows, feat_rows = _pm25_pass_rows()
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows, feat_rows]
        assert lt.run_leakage_test() is True
    bad_rows = []
    for r in feat_rows:
        d, vals = r[0], list(r[1:])
        if d == feat_rows[-1][0]:
            m = dict(zip(FEAT_COLS, vals))
            for k in list(m):
                if isinstance(m[k], (int, float)) and m[k] is not None:
                    m[k] = m[k] + 999.0
            m["day_of_week"] = (m["day_of_week"] + 3) % 7
            m["month"] = 12 if m["month"] != 12 else 1
            m["is_weekend"] = not (d.weekday() >= 5)
            vals = [m[c] for c in FEAT_COLS]
        bad_rows.append((d, *vals))
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows, bad_rows]
        assert lt.run_leakage_test() is False
    assert "FAILED" in capsys.readouterr().out


def test_pm25_leakage_missing_obs(capsys):
    from vericast.pm25 import leakage_test as lt

    obs_rows, feat_rows = _pm25_pass_rows(n=10)
    obs_rows_gap = [r for r in obs_rows if r[0] != feat_rows[3][0]]
    with patch.object(lt, "DATABASE_URL", "postgresql://x"), \
         patch.object(lt.psycopg, "connect") as mc:
        _, cur = _mock_conn(mc)
        cur.fetchall.side_effect = [obs_rows_gap, feat_rows]
        assert lt.run_leakage_test() is False
    assert "missing in observations" in capsys.readouterr().out


def test_pm25_leakage_requires_db():
    from vericast.pm25 import leakage_test as lt

    import os
    with patch.object(lt, "DATABASE_URL", None), \
         patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            lt.run_leakage_test()


# ---------------------------------------------------------------- local_time
def test_local_time_helpers():
    from vericast import local_time

    assert (local_time.today() - local_time.yesterday()).days == 1
    assert local_time.today_in("UTC") == local_time.today("UTC")
    assert local_time.today("UTC") == datetime.now(ZoneInfo("UTC")).date()
    assert local_time.yesterday("UTC") == local_time.today("UTC") - timedelta(days=1)
    # __main__ assertions
    now = datetime.now(local_time.TZ)
    assert now.utcoffset() == timedelta(hours=5, minutes=30)


# ---------------------------------------------------------------- gate extras
def test_gate_mae_window_and_sidecar(tmp_path, capsys):
    from vericast import gate

    assert gate._mae([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert gate.window_path("/tmp/m.txt") == "/tmp/m.txt.window.json"
    # empty dates -> None
    assert gate.record_training_window(str(tmp_path / "m.txt"), [], 0) is None
    # round-trip via legacy sidecar
    artifact = str(tmp_path / "model.txt")
    assert gate.read_training_window(artifact) is None
    dates = ["2026-08-01", "2026-08-02"]
    payload = gate.record_training_window(artifact, dates, 2)
    assert payload == {"first": "2026-08-01", "last": "2026-08-02", "rows": 2}
    assert gate.read_training_window(artifact) == payload
    # corrupt sidecar -> None
    with open(gate.window_path(artifact), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert gate.read_training_window(artifact) is None
    capsys.readouterr()


def test_gate_save_atomic_artifact(tmp_path):
    from vericast import gate
    from vericast.artifacts import read_bundle

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = X[:, 0] * 2 + rng.normal(size=20)
    model = lgb.train({"objective": "regression", "verbose": -1},
                      lgb.Dataset(X, label=y), num_boost_round=5)
    path = str(tmp_path / "m.txt")
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    payload = gate.save_atomic_artifact(model, path, dates, 20)
    assert payload["rows"] == 20
    assert read_bundle(path) is not None
    assert gate.read_training_window(path) == payload


def test_gate_validation_errors():
    from vericast import gate

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    params = {"objective": "regression", "verbose": -1}
    with pytest.raises(ValueError):
        gate.challenger_ships(X, y, params, 5, 0, holdout_days=10)
    with pytest.raises(ValueError):
        gate.challenger_ships(X, y[:-1], params, 5, 0)
    with pytest.raises(ValueError):
        gate.challenger_ships(X, y, params, 5, 99)
    with pytest.raises(ValueError):
        gate.challenger_ships(np.full_like(X, np.nan), y, params, 5, 0)


def test_gate_insufficient_rows(capsys):
    from vericast import gate

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    assert gate.challenger_ships(X, y, {"objective": "regression", "verbose": -1},
                                 5, 0) is False
    assert "REJECT" in capsys.readouterr().out


def _series(n=120):
    rng = np.random.default_rng(0)
    driver = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 20.0
    for t in range(1, n):
        y[t] = 20.0 + 0.5 * (y[t - 1] - 20.0) + 5.0 * driver[t] + rng.normal()
    X = np.column_stack([np.roll(y, 1), driver, rng.normal(size=n)])
    X[0, 0] = y[0]
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    return X, y, dates


_PARAMS = {"objective": "regression", "verbose": -1, "num_leaves": 7,
           "min_data_in_leaf": 5, "random_state": 0}


def test_gate_incumbent_dates_edge_cases(tmp_path, capsys):
    from vericast import gate
    from vericast.artifacts import save_bundle

    X, y, dates = _series()
    model = lgb.train(_PARAMS, lgb.Dataset(X, label=y), num_boost_round=5)
    # incumbent with a valid bundle window
    ipath = str(tmp_path / "inc.txt")
    save_bundle(model, ipath, dates, len(dates))
    # dates=None -> not comparable line
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=ipath, dates=None)
    assert "not comparable" in capsys.readouterr().out or "assume it saw" in capsys.readouterr().out
    # misaligned dates (wrong length) -> not comparable
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=ipath,
                           dates=dates[:-5])
    assert "not comparable" in capsys.readouterr().out
    # non-monotonic dates
    bad = list(dates)
    bad[10], bad[11] = bad[11], bad[10]
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=ipath, dates=bad)
    assert "not comparable" in capsys.readouterr().out
    # duplicated dates
    dup = list(dates)
    dup[5] = dup[6]
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=ipath, dates=dup)
    assert "not comparable" in capsys.readouterr().out
    # non-list dates (TypeError path)
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=ipath, dates=123)
    assert "not comparable" in capsys.readouterr().out


def test_gate_incumbent_corrupt_window(tmp_path, capsys):
    from vericast import gate

    X, y, dates = _series()
    # legacy artifact (no bundle) + corrupt sidecar content still reads None
    path = str(tmp_path / "legacy.txt")
    lgb.train(_PARAMS, lgb.Dataset(X, label=y),
              num_boost_round=5).save_model(path)
    with open(gate.window_path(path), "w", encoding="utf-8") as fh:
        fh.write('{"first": "not-a-date", "last": "also-bad", "rows": 5}')
    gate.challenger_ships(X, y, _PARAMS, 5, 0, incumbent_path=path, dates=dates)
    out = capsys.readouterr().out
    assert "assume it saw them" in out or "not comparable" in out


def test_gate_demo():
    from vericast import gate

    gate.demo()


# ---------------------------------------------------------------- vericast __init__
def test_require_database_url():
    from vericast import require_database_url

    assert require_database_url("postgresql://x") == "postgresql://x"
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(RuntimeError):
            require_database_url(None)
        os.environ["DATABASE_URL"] = "postgresql://env"
        try:
            assert require_database_url() == "postgresql://env"
        finally:
            del os.environ["DATABASE_URL"]


def test_require_city_state():
    from vericast import require_city_of_record, require_state_of_record

    assert require_city_of_record("Nagpur") == "Nagpur"
    assert require_state_of_record("Maharashtra") == "Maharashtra"
    with pytest.raises(RuntimeError):
        require_city_of_record("Mumbai")
    with pytest.raises(RuntimeError):
        require_state_of_record("Goa")


def test_refuse_stale_and_implausible():
    from vericast import refuse_implausible, refuse_stale

    assert refuse_stale(date(2026, 8, 30), date(2026, 8, 31), 2, "PM2.5") == 1
    with pytest.raises(RuntimeError):
        refuse_stale(date(2026, 8, 20), date(2026, 8, 31), 2, "PM2.5")
    assert refuse_implausible(50.0, 1.0, 500.0, "m", "ug/m3") == 50.0
    with pytest.raises(RuntimeError):
        refuse_implausible(None, 1.0, 500.0, "m", "ug/m3")
    with pytest.raises(RuntimeError):
        refuse_implausible(9999.0, 1.0, 500.0, "m", "ug/m3")


def test_acquire_pipeline_lock():
    from vericast import acquire_pipeline_lock

    cur = MagicMock()
    acquire_pipeline_lock(cur, "pm25_ingest")
    cur.execute.assert_called_once()
    with pytest.raises(KeyError):
        acquire_pipeline_lock(MagicMock(), "nope")
    # failing cursor without allow -> raise; with allow -> swallow
    bad = MagicMock()
    bad.execute.side_effect = RuntimeError("no pg")
    with pytest.raises(RuntimeError):
        acquire_pipeline_lock(bad, "pm25_ingest")
    acquire_pipeline_lock(bad, "pm25_ingest", allow_missing_lock=True)


def test_revision_helpers(capsys):
    from vericast import reopen_revised_actuals, revision_sql

    div, reopen = revision_sql("p", "o", "city", "actual", "pred", "obs")
    assert "p" in div and "o" in reopen
    cur = MagicMock()
    cur.fetchall.return_value = []
    assert reopen_revised_actuals(cur, div, reopen, "Nagpur", "ug/m3") == 0
    # one frozen (NULL now) -> 0 reopened
    cur.fetchall.return_value = [(date(2026, 8, 1), "m", 50.0, None)]
    cur.rowcount = 0
    assert reopen_revised_actuals(cur, div, reopen, "Nagpur", "ug/m3") == 0
    # one revised -> executes reopen
    cur.fetchall.return_value = [(date(2026, 8, 1), "m", 50.0, 55.0)]
    cur.rowcount = 1
    assert reopen_revised_actuals(cur, div, reopen, "Nagpur", "ug/m3") == 1
    # all frozen -> early 0 without reopen execute count change
    cur2 = MagicMock()
    cur2.fetchall.return_value = [(date(2026, 8, 1), "m", 50.0, None)]
    assert reopen_revised_actuals(cur2, div, reopen, "Nagpur", "ug/m3") == 0
    cur2.execute.assert_called_once()
    capsys.readouterr()


def test_alignment_helpers(capsys):
    from vericast import alignment_sql, verify_alignment

    gap_sql, orphan_sql = alignment_sql("features", "observations", "city")
    assert "features" in orphan_sql and "observations" in gap_sql
    cur = MagicMock()
    cur.fetchone.side_effect = [(2,), (2,)]
    verify_alignment(cur, gap_sql, orphan_sql, "Nagpur")
    assert "Alignment OK" in capsys.readouterr().out
    cur.fetchone.side_effect = [(1,), (2,)]
    with pytest.raises(AssertionError):
        verify_alignment(cur, gap_sql, orphan_sql, "Nagpur")


def test_send_alert(tmp_path, monkeypatch, capsys):
    from vericast import send_alert

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    send_alert("subj", "msg")
    assert "subj" in summary.read_text(encoding="utf-8")
    # bad summary path -> prints failure but does not raise
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "nope" / "s.md"))
    send_alert("s2", "m2")
    # webhook success / failure / exception
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.test/hook")
    ok = MagicMock()
    ok.is_success = True
    with patch("httpx.post", return_value=ok):
        send_alert("s", "m")
    bad = MagicMock()
    bad.is_success = False
    bad.status_code = 500
    bad.text = "err"
    with patch("httpx.post", return_value=bad):
        send_alert("s", "m")
    with patch("httpx.post", side_effect=RuntimeError("down")):
        send_alert("s", "m")
    capsys.readouterr()


# ---------------------------------------------------------------- publication
def test_publication_timing_and_guards():
    from vericast.publication import (publication_timing,
                                      publish_predictions,
                                      require_advance_forecast)

    fc = date(2026, 8, 20)
    before = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    t = publication_timing(fc, "UTC", now=before)
    assert t["source"] == "daily" and t["feature_as_of"] == date(2026, 8, 19)
    after = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    t2 = publication_timing(fc, "UTC", now=after)
    assert t2["source"] == "nowcast"
    with pytest.raises(ValueError):
        publication_timing(fc, "UTC", now=datetime(2026, 8, 19, 12, 0))
    with pytest.raises(ValueError):
        require_advance_forecast(fc, "UTC", now=datetime(2026, 8, 19, 12, 0))
    assert require_advance_forecast(fc, "UTC", now=before) == before
    with pytest.raises(RuntimeError):
        require_advance_forecast(fc, "UTC", now=after)
    # empty records -> RuntimeError (covers send_alert path)
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("GITHUB_STEP_SUMMARY", None)
        os.environ.pop("ALERT_WEBHOOK_URL", None)
        with pytest.raises(RuntimeError):
            publish_predictions(MagicMock(), "SQL", [], fc, "UTC")


def test_publish_predictions_daily_and_nowcast():
    from vericast.publication import publish_predictions

    fc = date(2026, 8, 20)
    before = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    after = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    with patch("vericast.publication.publication_timing",
               return_value={"source": "daily", "issued_at": before,
                             "timing_status": "advance_forecast"}), \
         patch("vericast.publication.require_advance_forecast",
               return_value=before):
        cur = MagicMock()
        timing = publish_predictions(cur, "SQL", [("a",)], fc, "UTC")
        assert timing["source"] == "daily"
        cur.execute.assert_called_once()
    with patch("vericast.publication.publication_timing",
               return_value={"source": "nowcast", "issued_at": after,
                             "timing_status": "delayed_estimate"}):
        cur = MagicMock()
        timing = publish_predictions(cur, "SQL", [("a",), ("b",)], fc, "UTC")
        assert timing["source"] == "nowcast"
        assert cur.execute.call_count == 2


# ---------------------------------------------------------------- artifacts
def test_artifacts_bundle_and_legacy(tmp_path, capsys):
    from vericast.artifacts import (bundle_path, load_model, model_exists,
                                    model_metadata, read_bundle, save_bundle)

    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    y = X[:, 0] + rng.normal(size=30)
    model = lgb.train({"objective": "regression", "verbose": -1},
                      lgb.Dataset(X, label=y), num_boost_round=5)
    base = str(tmp_path / "m.txt")
    assert bundle_path(base) == base + ".bundle.json"
    assert model_exists(base) is False
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    save_bundle(model, base, dates, 30)
    assert model_exists(base) is True
    payload = read_bundle(base)
    assert payload["version"] == 1
    meta = model_metadata(base)
    assert meta["artifact_format"] == "bundle"
    assert meta["feature_count"] == 3
    # stale legacy beside bundle -> warn path
    with open(base, "w", encoding="utf-8") as fh:
        fh.write("legacy")
    meta2 = model_metadata(base, expected_features=3)
    assert meta2["artifact_format"] == "bundle"
    with pytest.raises(ValueError):
        model_metadata(base, expected_features=99)
    m = load_model(base)
    assert m.num_feature() == 3
    capsys.readouterr()
    # invalid bundles
    import json
    with open(bundle_path(base), "w", encoding="utf-8") as fh:
        json.dump({"version": 2}, fh)
    with pytest.raises(ValueError):
        read_bundle(base)
    with open(bundle_path(base), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "model": "x",
                   "window": {"first": "2026-01-01", "last": "2026-01-02", "rows": 0}}, fh)
    with pytest.raises(ValueError):
        read_bundle(base)
    with open(bundle_path(base), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "model": "x",
                   "window": {"first": "not-a-date", "last": "2026-01-02", "rows": 1}}, fh)
    with pytest.raises(ValueError):
        read_bundle(base)
    with open(bundle_path(base), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "model": "x",
                   "window": {"first": "2026-01-05", "last": "2026-01-01", "rows": 2}}, fh)
    with pytest.raises(ValueError):
        read_bundle(base)


def test_artifacts_legacy_model(tmp_path, capsys):
    from vericast.artifacts import load_model, model_exists, model_metadata

    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    path = str(tmp_path / "legacy.txt")
    lgb.train({"objective": "regression", "verbose": -1},
              lgb.Dataset(X, label=y), num_boost_round=3).save_model(path)
    assert model_exists(path) is True
    meta = model_metadata(path)
    assert meta["artifact_format"] == "legacy"
    m = load_model(path)
    assert m.num_feature() == 2
    capsys.readouterr()


def test_save_bundle_validation(tmp_path):
    from vericast.artifacts import save_bundle

    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    model = lgb.train({"objective": "regression", "verbose": -1},
                      lgb.Dataset(X, label=y), num_boost_round=3)
    with pytest.raises(ValueError):
        save_bundle(model, str(tmp_path / "a.txt"), [], 0)
    with pytest.raises(ValueError):
        save_bundle(model, str(tmp_path / "a.txt"),
                    [date(2026, 1, 1)], 2)
