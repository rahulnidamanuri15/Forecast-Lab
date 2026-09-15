"""DB/network-free coverage for both ingest modules and both train modules.

Mocks psycopg.connect and httpx.get so every branch runs without Postgres
or upstream HTTP. Complements test_coverage_fill.py (leakage/gate/init).
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


def _mock_pg(monkey_module, fetchall=None, fetchone=None, rowcount=1):
    """Patch <module>.psycopg.connect to yield a mock conn/cursor."""
    mock_connect = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    if fetchall is not None:
        mock_cur.fetchall.return_value = fetchall
    if fetchone is not None:
        mock_cur.fetchone.return_value = fetchone
    mock_cur.rowcount = rowcount
    patcher = patch.object(monkey_module.psycopg, "connect",
                           return_value=mock_connect.return_value)
    patcher.start()
    return mock_connect, mock_conn, mock_cur, patcher


# ------------------------------------------------------- elec ingest basics
def test_elec_get_last_and_earliest_hole():
    from vericast.elec import ingest as ei

    cur = MagicMock()
    cur.fetchone.return_value = [date(2026, 8, 10)]
    assert ei.get_last_observed_date(cur) == date(2026, 8, 10)
    cur.fetchone.return_value = None
    assert ei.get_last_observed_date(cur) is None
    cur.fetchone.return_value = [date(2026, 8, 5)]
    assert ei.get_earliest_hole(cur, date(2026, 8, 1),
                                end=date(2026, 8, 10)) == date(2026, 8, 5)
    # default end goes through local_time.yesterday
    with patch("vericast.elec.ingest.local_time.yesterday",
               return_value=date(2026, 8, 10)):
        cur.fetchone.return_value = [None]
        assert ei.get_earliest_hole(cur, date(2026, 8, 1)) is None


def test_elec_resolve_date_range_first_run():
    from vericast.elec import ingest as ei

    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch("vericast.elec.ingest.local_time.yesterday",
               return_value=date(2026, 8, 10)), \
         patch.object(ei.psycopg, "connect") as mc, \
         patch.object(ei, "get_last_observed_date", return_value=None) as gl, \
         patch.object(ei, "get_earliest_hole") as gh:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mc.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        start, end = ei.resolve_date_range()
        assert start == date(2023, 1, 1) and end == date(2026, 8, 10)
        gl.assert_called_once()
        gh.assert_not_called()


def test_elec_resolve_date_range_hole_and_no_hole():
    from vericast.elec import ingest as ei

    yesterday = date(2026, 8, 10)
    # hole inside window -> re-scan from hole
    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch("vericast.elec.ingest.local_time.yesterday",
               return_value=yesterday), \
         patch.object(ei.psycopg, "connect") as mc, \
         patch.object(ei, "get_last_observed_date",
                      return_value=date(2026, 8, 9)), \
         patch.object(ei, "get_earliest_hole",
                      return_value=date(2026, 8, 5)):
        mc.return_value.__enter__.return_value = MagicMock()
        mc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
        start, end = ei.resolve_date_range()
        assert start == date(2026, 8, 5) and end == yesterday
    # hole at edge of RESCAN window -> warn branch
    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch("vericast.elec.ingest.local_time.yesterday",
               return_value=yesterday), \
         patch.object(ei.psycopg, "connect") as mc, \
         patch.object(ei, "get_last_observed_date",
                      return_value=yesterday - timedelta(days=1)), \
         patch.object(ei, "get_earliest_hole",
                      return_value=yesterday - timedelta(days=30)):
        mc.return_value.__enter__.return_value = MagicMock()
        mc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
        start, _ = ei.resolve_date_range()
        assert start == yesterday - timedelta(days=30)
    # no hole -> monotonic resume
    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch("vericast.elec.ingest.local_time.yesterday",
               return_value=yesterday), \
         patch.object(ei.psycopg, "connect") as mc, \
         patch.object(ei, "get_last_observed_date",
                      return_value=date(2026, 8, 9)), \
         patch.object(ei, "get_earliest_hole", return_value=None):
        mc.return_value.__enter__.return_value = MagicMock()
        mc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
        start, _ = ei.resolve_date_range()
        assert start == date(2026, 8, 10)


def test_elec_fetch_demand_missing_columns():
    from vericast.elec import ingest as ei

    resp = MagicMock()
    resp.text = "state,date\nMaharashtra,2026-08-01\n"
    resp.content = resp.text.encode()
    with patch.object(ei.httpx, "get", return_value=resp):
        with pytest.raises(RuntimeError, match="missing column"):
            ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 31))


def test_elec_fetch_demand_oversize_and_retries():
    from vericast.elec import ingest as ei

    # oversize body hits the cap check (raised + retried, then falls through
    # to parsing the small .text in this synthetic response)
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.content = b"x" * (25 * 1024 * 1024 + 1)
    resp.text = "state,date,max_demand_met_mw,energy_met_mu\n"
    with patch.object(ei.httpx, "get", return_value=resp), \
         patch("vericast.elec.ingest.time.sleep"):
        rows = ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 2))
        assert rows == {}
    # retry then success
    good = MagicMock()
    good.text = ("state,date,max_demand_met_mw,energy_met_mu\n"
                 f"{ei.STATE},2026-08-02,25000.0,500.0\n")
    good.content = good.text.encode()
    good.raise_for_status.return_value = None
    with patch.object(ei.httpx, "get",
                      side_effect=[RuntimeError("down"), good]), \
         patch("vericast.elec.ingest.time.sleep"):
        rows = ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 31))
        assert rows == {"2026-08-02": (25000.0, 500.0)}
    # all attempts fail -> raise
    with patch.object(ei.httpx, "get", side_effect=RuntimeError("down")), \
         patch("vericast.elec.ingest.time.sleep"):
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 2))


def test_elec_fetch_demand_bytes_and_skips():
    from vericast.elec import ingest as ei

    # bytes-only response (no .text str) exercises the raw-decode branch
    raw = ("state,date,max_demand_met_mw,energy_met_mu\n"
           f"{ei.STATE},2026-08-03,26000.0,\n").encode()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    type(resp).text = property(lambda self: None)
    resp.content = raw
    with patch.object(ei.httpx, "get", return_value=resp):
        rows = ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 31))
        assert rows == {"2026-08-03": (26000.0, None)}
    # invalid UTF-8
    resp2 = MagicMock()
    resp2.raise_for_status.return_value = None
    type(resp2).text = property(lambda self: None)
    resp2.content = b"\xff\xfe bad"
    with patch.object(ei.httpx, "get", return_value=resp2):
        with pytest.raises(RuntimeError, match="UTF-8"):
            ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 2))
    # blank peak, unparseable peak, implausible peak all skip the day
    resp3 = MagicMock()
    resp3.text = ("state,date,max_demand_met_mw,energy_met_mu\n"
                  f"{ei.STATE},2026-08-04,,500.0\n"
                  f"{ei.STATE},2026-08-05,N/A,500.0\n"
                  f"{ei.STATE},2026-08-06,99999999,500.0\n"
                  "Other,2026-08-07,25000.0,500.0\n"
                  f"{ei.STATE},2026-09-05,25000.0,500.0\n")
    resp3.content = resp3.text.encode()
    with patch.object(ei.httpx, "get", return_value=resp3):
        rows = ei.fetch_demand(date(2026, 8, 1), date(2026, 8, 31))
        assert rows == {}


def test_elec_fetch_temperature_branches():
    from vericast.elec import ingest as ei

    def _resp(payload):
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status.return_value = None
        return r

    start, end = date(2026, 8, 1), date(2026, 8, 2)
    # success: 3 locations averaged
    payload = [{"daily": {"time": ["2026-08-01"],
                          "temperature_2m_mean": [30.0],
                          "temperature_2m_max": [35.0]}} for _ in range(3)]
    with patch.object(ei.httpx, "get", return_value=_resp(payload)):
        temps = ei.fetch_temperature(start, end)
        assert temps["2026-08-01"] == (30.0, 35.0)
    # single object (not list) + partial cities warn + implausible nulled
    payload2 = {"daily": {"time": ["2026-08-01"],
                          "temperature_2m_mean": [999.0],
                          "temperature_2m_max": [999.0]}}
    with patch.object(ei.httpx, "get", return_value=_resp(payload2)):
        temps = ei.fetch_temperature(start, end)
        assert temps["2026-08-01"] == (None, None)
    # retry then success
    with patch.object(ei.httpx, "get",
                      side_effect=[RuntimeError("x"), _resp(payload)]), \
         patch("vericast.elec.ingest.time.sleep"):
        assert ei.fetch_temperature(start, end)["2026-08-01"] == (30.0, 35.0)
    # all fail
    with patch.object(ei.httpx, "get", side_effect=RuntimeError("down")), \
         patch("vericast.elec.ingest.time.sleep"):
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            ei.fetch_temperature(start, end)
    # non-JSON
    bad = MagicMock()
    bad.raise_for_status.return_value = None
    bad.json.side_effect = ValueError("nope")
    with patch.object(ei.httpx, "get", return_value=bad):
        with pytest.raises(RuntimeError, match="not JSON"):
            ei.fetch_temperature(start, end)
    # missing daily block
    with patch.object(ei.httpx, "get", return_value=_resp([{"nope": 1}])):
        with pytest.raises(RuntimeError, match="missing daily"):
            ei.fetch_temperature(start, end)
    # malformed per-date arrays
    with patch.object(ei.httpx, "get", return_value=_resp(
            [{"daily": {"time": ["2026-08-01"]}}])):
        with pytest.raises(RuntimeError, match="malformed"):
            ei.fetch_temperature(start, end)
    # None temps stay None; partial mean warn path
    payload3 = [{"daily": {"time": ["2026-08-01"],
                           "temperature_2m_mean": [None],
                           "temperature_2m_max": [None]}},
                {"daily": {"time": ["2026-08-01"],
                           "temperature_2m_mean": [30.0],
                           "temperature_2m_max": [35.0]}}]
    with patch.object(ei.httpx, "get", return_value=_resp(payload3)):
        temps = ei.fetch_temperature(start, end)
        assert temps["2026-08-01"] == (30.0, 35.0)


def test_elec_report_revisions_and_insert(capsys):
    from vericast.elec import ingest as ei

    # no revision -> quiet
    ei.report_revisions({}, [("Maharashtra", "2026-08-01", 25000.0)])
    # revised + NaN (non-finite) + None; all values numeric so the
    # print formatting holds (unparseable strings never reach here in prod)
    before = {"2026-08-01": 25000.0, "2026-08-02": 25000.0,
              "2026-08-03": 25000.0}
    records = [("Maharashtra", "2026-08-01", 25100.0),
               ("Maharashtra", "2026-08-02", float("nan")),
               ("Maharashtra", "2026-08-03", None)]
    ei.report_revisions(before, records)
    assert "REVISED" in capsys.readouterr().out
    # insert success with mocked pg
    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch.object(ei.psycopg, "connect") as mc, \
         patch("vericast.elec.ingest.acquire_pipeline_lock"):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mc.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_cur.fetchone.return_value = [5, date(2026, 8, 1), date(2026, 8, 5)]
        mock_cur.rowcount = 2
        ei.insert_observations([("Maharashtra", date(2026, 8, 5), 25000.0, 500.0,
                                 30.0, 35.0)])
        mock_cur.executemany.assert_called_once()
    # insert exception path re-raises
    with patch.object(ei, "DATABASE_URL", "postgresql://x"), \
         patch.object(ei.psycopg, "connect", side_effect=RuntimeError("db down")), \
         patch("vericast.elec.ingest.acquire_pipeline_lock"):
        with pytest.raises(RuntimeError):
            ei.insert_observations([("Maharashtra", date(2026, 8, 5), 25000.0,
                                     500.0, 30.0, 35.0)])
    capsys.readouterr()


def test_elec_main_branches():
    from vericast.elec import ingest as ei

    with patch.object(ei, "resolve_date_range",
                      return_value=(date(2026, 8, 11), date(2026, 8, 10))):
        ei.main()  # nothing to fetch
    with patch.object(ei, "resolve_date_range",
                      return_value=(date(2026, 8, 1), date(2026, 8, 5))), \
         patch.object(ei, "fetch_demand", return_value={}):
        ei.main()  # mirror lag, no demand
    with patch.object(ei, "resolve_date_range",
                      return_value=(date(2026, 8, 1), date(2026, 8, 2))), \
         patch.object(ei, "fetch_demand",
                      return_value={"2026-08-01": (25000.0, 500.0)}), \
         patch.object(ei, "fetch_temperature",
                      return_value={"2026-08-01": (30.0, 35.0)}), \
         patch.object(ei, "insert_observations") as ins:
        ei.main()
        ins.assert_called_once()


# ------------------------------------------------------- pm25 ingest
def test_pm25_get_last_and_hole_and_range():
    from vericast.pm25 import ingest as pi

    cur = MagicMock()
    cur.fetchone.return_value = [date(2026, 8, 9)]
    assert pi.get_last_observed_date(cur) == date(2026, 8, 9)
    cur.fetchone.return_value = None
    assert pi.get_last_observed_date(cur) is None
    cur.fetchone.return_value = [date(2026, 8, 4)]
    assert pi.get_earliest_hole(cur, date(2026, 8, 1),
                                end=date(2026, 8, 10)) == date(2026, 8, 4)
    with patch.object(pi, "DATABASE_URL", "postgresql://x"), \
         patch("vericast.pm25.ingest.local_time.yesterday",
               return_value=date(2026, 8, 10)), \
         patch.object(pi.psycopg, "connect") as mc, \
         patch.object(pi, "get_last_observed_date", return_value=None):
        mc.return_value.__enter__.return_value = MagicMock()
        mc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
        start, end = pi.resolve_date_range()
        assert start == date(2023, 8, 1) and end == date(2026, 8, 10)
    # hole + edge warn + no-hole
    y = date(2026, 8, 10)
    for hole, last, want in [(date(2026, 8, 5), date(2026, 8, 9), date(2026, 8, 5)),
                             (y - timedelta(days=30), y - timedelta(days=1),
                              y - timedelta(days=30)),
                             (None, date(2026, 8, 9), date(2026, 8, 10))]:
        with patch.object(pi, "DATABASE_URL", "postgresql://x"), \
             patch("vericast.pm25.ingest.local_time.yesterday", return_value=y), \
             patch.object(pi.psycopg, "connect") as mc, \
             patch.object(pi, "get_last_observed_date", return_value=last), \
             patch.object(pi, "get_earliest_hole", return_value=hole):
            mc.return_value.__enter__.return_value = MagicMock()
            mc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
            assert pi.resolve_date_range()[0] == want


def _aq_wx(aq_times, pm25, pm10, wx_days):
    aq = MagicMock()
    aq.json.return_value = {"hourly": {"time": aq_times, "pm2_5": pm25,
                                       "pm10": pm10}}
    aq.raise_for_status.return_value = None
    aq.content = b"{}"
    wx = MagicMock()
    wx.json.return_value = {"daily": {
        "time": [d for d, _ in wx_days],
        "temperature_2m_mean": [t for _, t in wx_days],
        "wind_speed_10m_max": [10.0] * len(wx_days),
        "precipitation_sum": [0.0] * len(wx_days)}}
    wx.raise_for_status.return_value = None
    wx.content = b"{}"
    return aq, wx


def test_pm25_fetch_and_aggregate_success_and_thin_day():
    from vericast.pm25 import ingest as pi

    start, end = date(2026, 8, 1), date(2026, 8, 1)
    hours = [f"2026-08-01T{h:02d}:00" for h in range(20)]
    aq, wx = _aq_wx(hours, [50.0] * 20, [80.0] * 20, [("2026-08-01", 25.0)])
    with patch.object(pi.httpx, "get", side_effect=[aq, wx]):
        rows = pi.fetch_and_aggregate_data(start, end)
        assert len(rows) == 1 and rows[0][2] == pytest.approx(50.0)
    # thin day (<18h) -> NULLs
    hours2 = ["2026-08-01T00:00", "2026-08-01T01:00"]
    aq2, wx2 = _aq_wx(hours2, [50.0, 60.0], [80.0, 90.0],
                       [("2026-08-01", 25.0)])
    with patch.object(pi.httpx, "get", side_effect=[aq2, wx2]):
        rows = pi.fetch_and_aggregate_data(start, end)
        assert rows[0][2] is None and rows[0][3] is None
    # implausible pm2_5 nulled; bad aux nulled; unparseable timestamp skipped
    hours3 = [f"2026-08-01T{h:02d}:00" for h in range(20)] + ["not-a-time"]
    aq3, wx3 = _aq_wx(hours3, [1000.0] * 20 + [50.0], [5000.0] * 20 + [80.0],
                       [("2026-08-01", 999.0)])
    wx3.json.return_value["daily"]["wind_speed_10m_max"] = [999.0]
    wx3.json.return_value["daily"]["precipitation_sum"] = [999.0]
    with patch.object(pi.httpx, "get", side_effect=[aq3, wx3]):
        rows = pi.fetch_and_aggregate_data(start, end)
        assert rows[0][2] is None and rows[0][4] is None


def test_pm25_fetch_errors():
    from vericast.pm25 import ingest as pi

    start, end = date(2026, 8, 1), date(2026, 8, 2)
    # retry then fail
    with patch.object(pi.httpx, "get", side_effect=RuntimeError("down")), \
         patch("vericast.pm25.ingest.time.sleep"):
        with pytest.raises(RuntimeError, match="failed after"):
            pi.fetch_and_aggregate_data(start, end)
    # oversize cap
    big = MagicMock()
    big.raise_for_status.return_value = None
    big.content = b"x" * (25 * 1024 * 1024 + 1)
    with patch.object(pi.httpx, "get", return_value=big):
        with pytest.raises(RuntimeError, match="exceeds"):
            pi.fetch_and_aggregate_data(start, end)
    # AQ missing fields
    aq = MagicMock()
    aq.json.return_value = {"hourly": {}}
    aq.raise_for_status.return_value = None
    aq.content = b"{}"
    wx = MagicMock()
    wx.json.return_value = {"daily": {"time": [], "temperature_2m_mean": [],
                                      "wind_speed_10m_max": [],
                                      "precipitation_sum": []}}
    wx.raise_for_status.return_value = None
    wx.content = b"{}"
    with patch.object(pi.httpx, "get", side_effect=[aq, wx]):
        with pytest.raises(RuntimeError, match="missing hourly"):
            pi.fetch_and_aggregate_data(start, end)
    # length mismatch AQ
    aq2 = MagicMock()
    aq2.json.return_value = {"hourly": {"time": ["a", "b"], "pm2_5": [1.0],
                                        "pm10": [1.0, 2.0]}}
    aq2.raise_for_status.return_value = None
    aq2.content = b"{}"
    with patch.object(pi.httpx, "get", side_effect=[aq2, wx]):
        with pytest.raises(RuntimeError, match="length mismatch"):
            pi.fetch_and_aggregate_data(start, end)
    # weather not JSON / missing daily / length mismatch
    good_aq = MagicMock()
    good_aq.json.return_value = {"hourly": {"time": [], "pm2_5": [],
                                            "pm10": []}}
    good_aq.raise_for_status.return_value = None
    good_aq.content = b"{}"
    bad_wx = MagicMock()
    bad_wx.raise_for_status.return_value = None
    bad_wx.content = b"{}"
    bad_wx.json.side_effect = ValueError("bad")
    with patch.object(pi.httpx, "get", side_effect=[good_aq, bad_wx]):
        with pytest.raises(RuntimeError, match="not JSON"):
            pi.fetch_and_aggregate_data(start, end)
    bad_wx2 = MagicMock()
    bad_wx2.raise_for_status.return_value = None
    bad_wx2.content = b"{}"
    bad_wx2.json.return_value = {"daily": {}}
    with patch.object(pi.httpx, "get", side_effect=[good_aq, bad_wx2]):
        with pytest.raises(RuntimeError, match="missing daily"):
            pi.fetch_and_aggregate_data(start, end)
    bad_wx3 = MagicMock()
    bad_wx3.raise_for_status.return_value = None
    bad_wx3.content = b"{}"
    bad_wx3.json.return_value = {"daily": {"time": ["x"],
                                           "temperature_2m_mean": [],
                                           "wind_speed_10m_max": [],
                                           "precipitation_sum": []}}
    with patch.object(pi.httpx, "get", side_effect=[good_aq, bad_wx3]):
        with pytest.raises(RuntimeError, match="length mismatch"):
            pi.fetch_and_aggregate_data(start, end)


def test_pm25_report_insert_main(capsys):
    from vericast.pm25 import ingest as pi

    pi.report_revisions({}, [("Nagpur", "2026-08-01", 50.0)])
    before = {"2026-08-01": 50.0, "2026-08-02": 50.0}
    pi.report_revisions(before, [("Nagpur", "2026-08-01", 55.0),
                                 ("Nagpur", "2026-08-02", None)])
    assert "REVISED" in capsys.readouterr().out
    with patch.object(pi, "DATABASE_URL", "postgresql://x"), \
         patch.object(pi.psycopg, "connect") as mc, \
         patch("vericast.pm25.ingest.acquire_pipeline_lock"):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mc.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = []
        mock_cur.fetchone.return_value = [3]
        mock_cur.rowcount = 1
        pi.insert_observations([("Nagpur", date(2026, 8, 1), 50.0, 80.0,
                                 25.0, 10.0, 0.0)])
    with patch.object(pi, "DATABASE_URL", "postgresql://x"), \
         patch.object(pi.psycopg, "connect", side_effect=RuntimeError("db")), \
         patch("vericast.pm25.ingest.acquire_pipeline_lock"):
        with pytest.raises(RuntimeError):
            pi.insert_observations([("Nagpur", date(2026, 8, 1), 50.0, 80.0,
                                     25.0, 10.0, 0.0)])
    with patch.object(pi, "resolve_date_range",
                      return_value=(date(2026, 8, 11), date(2026, 8, 10))):
        pi.main()
    with patch.object(pi, "resolve_date_range",
                      return_value=(date(2026, 8, 1), date(2026, 8, 2))), \
         patch.object(pi, "fetch_and_aggregate_data", return_value=[]):
        pi.main()
    with patch.object(pi, "resolve_date_range",
                      return_value=(date(2026, 8, 1), date(2026, 8, 2))), \
         patch.object(pi, "fetch_and_aggregate_data",
                      return_value=[("Nagpur", "2026-08-01", 1.0, 1.0, 1.0,
                                     1.0, 1.0)]), \
         patch.object(pi, "insert_observations") as ins:
        pi.main()
        ins.assert_called_once()
    capsys.readouterr()


# ------------------------------------------------------- train
def test_train_load_full_dataset():
    from vericast.elec import train as et
    from vericast.pm25 import train as pt

    for mod in (et, pt):
        with patch.object(mod, "DATABASE_URL", "postgresql://x"), \
             patch.object(mod.psycopg, "connect") as mc:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mc.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cur
            mock_cur.fetchall.return_value = []
            with pytest.raises(RuntimeError, match="No .*training data"):
                mod.load_full_dataset()
            mock_cur.fetchall.return_value = [("2026-08-01", "2026-08-02",
                                               1.0, 2.0)]
            assert len(mod.load_full_dataset()) == 1
    import os
    for mod in (et, pt):
        with patch.object(mod, "DATABASE_URL", None), \
             patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError):
                mod.load_full_dataset()


def test_train_and_save_branches():
    from vericast.elec import train as et
    from vericast.pm25 import train as pt

    # bad alignment
    for mod in (et, pt):
        rows = [(date(2026, 8, 1), date(2026, 8, 3),
                 *([1.0] * len(mod.FEATURE_COLUMNS)), 10.0)]
        with patch.object(mod, "load_full_dataset", return_value=rows):
            with pytest.raises(ValueError, match="Invalid target alignment"):
                mod.train_and_save()
    # feature-count mismatch (row too short vs header) -> AssertionError
    for mod in (et, pt):
        n = len(mod.FEATURE_COLUMNS)
        X = [[1.0] * (n - 1)]
        y = [10.0]
        dates = [date(2026, 8, 2)]
        feat_dates = [date(2026, 8, 1)]
        import numpy as np
        with patch.object(mod, "load_full_dataset",
                          return_value=[(feat_dates[0], dates[0],
                                         *X[0], y[0])]):
            # X built from row[2:-1] will have n-1 cols -> mismatch raises
            # before the gate is even consulted
            with pytest.raises(AssertionError, match="Feature count mismatch"):
                mod.train_and_save()
    # gate reject -> None (covers send_alert paths with/without window)
    for mod in (et, pt):
        n = len(mod.FEATURE_COLUMNS)
        rows = [(date(2026, 8, 1) + timedelta(days=i),
                 date(2026, 8, 2) + timedelta(days=i),
                 *([float(i + 1)] * n), float(10 + i)) for i in range(5)]
        with patch.object(mod, "load_full_dataset", return_value=rows), \
             patch.object(mod, "challenger_ships", return_value=False), \
             patch.object(mod, "read_training_window",
                          return_value={"first": "2026-08-01",
                                        "last": "2026-08-02", "rows": 2}), \
             patch.object(mod, "send_alert"):
            assert mod.train_and_save() is None
        with patch.object(mod, "load_full_dataset", return_value=rows), \
             patch.object(mod, "challenger_ships", return_value=False), \
             patch.object(mod, "read_training_window", return_value=None), \
             patch.object(mod, "send_alert"):
            assert mod.train_and_save() is None


def test_train_and_save_success():
    from vericast.elec import train as et
    from vericast.pm25 import train as pt

    import numpy as np

    for mod in (et, pt):
        n = len(mod.FEATURE_COLUMNS)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, n)) + 10.0
        y = X[:, 0] * 0.9 + rng.normal(size=100)
        feat_base = date(2026, 5, 1)
        rows = [(feat_base + timedelta(days=i),
                 feat_base + timedelta(days=i + 1),
                 *X[i].tolist(), float(y[i])) for i in range(100)]
        with patch.object(mod, "load_full_dataset", return_value=rows), \
             patch.object(mod, "challenger_ships", return_value=True), \
             patch.object(mod, "save_atomic_artifact",
                          return_value={"first": "x", "last": "y",
                                        "rows": 100}) as save, \
             patch.object(mod, "load_model") as lm:
            m = MagicMock()
            m.predict.return_value = np.array([10.0])
            lm.return_value = m
            assert mod.train_and_save() == mod.MODEL_PATH
            save.assert_called_once()
