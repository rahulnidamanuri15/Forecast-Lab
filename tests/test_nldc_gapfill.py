"""Official NLDC gap-fill: same guards as the CSV mirror path, tested without network.

vericast/elec/ingest.py now retries dates the community mirror lacks against
Grid-India's own daily PSP reports (API index + .xls on webcdn). These tests
pin the safety properties with stubbed workbooks and mocked transports:
a misdated, unit-slipped or reshaped report must skip its day, never poison
the scored record. A dead API is recorded in the run diagnostics; the
freshness gate at the end of ingest decides whether the stored series is
publishable, so a stale source fails at 1/6 instead of 5/6.
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from vericast import ELEC_STALE_LIMIT_DAYS, local_time
from vericast.elec import ingest as ei


class _Cell:
    def __init__(self, value):
        self.value = value


class _Sheet:
    def __init__(self, grid):
        self._grid = grid
        self.nrows = len(grid)
        self.ncols = max((len(r) for r in grid), default=0)

    def cell(self, rx, cx):
        try:
            return _Cell(self._grid[rx][cx])
        except IndexError:
            raise IndexError(f"cell ({rx}, {cx}) out of range")


def _book(sheet=None, sheets=("MOP_E",)):
    book = MagicMock()
    if sheet is None:
        import xlrd
        book.sheet_by_name.side_effect = xlrd.XLRDError("No sheet named 'MOP_E'")
    else:
        book.sheet_by_name.return_value = sheet
    return book


def _maha_grid(mw=33480.0, mu=680.3, reporting="14-Oct-2026", state="Maharashtra"):
    return [
        ["", "", "", "", "", "", "", ""],
        ["Report for previous day", "", "", "", "", "", "Date of Reporting:", reporting],
        ["Region", "States", "Max.Demand Met during the day (MW)", "x", "Energy", "", "", ""],
        ["WR", state, mw, 0.0, mu, 255.9, 8.0, 0.58],
    ]


def parse(grid, day="2026-10-13"):
    with patch("xlrd.open_workbook", return_value=_book(_Sheet(grid))):
        return ei.parse_nldc_workbook(b"fake-bytes", day)


def test_happy_path_reads_the_same_two_columns_as_the_mirror():
    assert parse(_maha_grid()) == (33480.0, 680.3)


def test_a_dash_energy_value_degrades_to_null_like_the_mirror():
    assert parse(_maha_grid(mu="-")) == (33480.0, None)


def test_a_unit_slip_skips_the_day_instead_of_entering_the_record():
    assert parse(_maha_grid(mw=33480000.0)) is None


def test_an_unparseable_peak_skips_the_day():
    assert parse(_maha_grid(mw="N/A")) is None


def test_a_reshaped_workbook_skips_instead_of_raising():
    with patch("xlrd.open_workbook", return_value=_book(None)):
        assert ei.parse_nldc_workbook(b"fake-bytes", "2026-10-13") is None


def test_a_missing_maharashtra_row_skips():
    grid = _maha_grid(state="Gujarat")
    assert parse(grid) is None


def test_a_misdated_report_is_never_accepted():
    # Reporting date must be data date + 1; a drifted Title->date mapping
    # would otherwise file a genuine row under the wrong day.
    assert parse(_maha_grid(reporting="01-Jan-2020")) is None


def test_oversize_body_skips_before_parsing():
    with patch("xlrd.open_workbook") as mock_open:
        assert ei.parse_nldc_workbook(b"x" * (ei.NLDC_MAX_FILE_BYTES + 1),
                                      "2026-10-13") is None
        assert not mock_open.called


def _index_payload():
    return {"retData": [
        {"Title_": "13.10.26_NLDC_PSP", "MimeType": "application/vnd.ms-excel",
         "FilePath": "files/grdw/2026/10/13.10.26_NLDC_PSP_216.xls"},
        {"Title_": "13.10.26_NLDC_PSP", "MimeType": "application/pdf",
         "FilePath": "files/grdw/2026/10/13.10.26_NLDC_PSP_915.pdf"},
        {"Title_": "not-a-report", "MimeType": "application/vnd.ms-excel",
         "FilePath": "files/grdw/2026/10/junk.xls"},
        {"Title_": "99.99.99_NLDC_PSP", "MimeType": "application/vnd.ms-excel",
         "FilePath": "files/grdw/2026/10/bad.xls"},
    ]}


def test_index_keeps_xls_only_with_parseable_titles():
    resp = MagicMock()
    resp.json.return_value = _index_payload()
    with patch.object(ei.httpx, "post", return_value=resp):
        assert ei.fetch_nldc_index() == {
            "2026-10-13": "files/grdw/2026/10/13.10.26_NLDC_PSP_216.xls"}


def test_index_accepts_an_xls_path_when_the_mime_type_is_generic():
    payload = _index_payload()
    payload["retData"][0]["MimeType"] = "application/octet-stream"
    resp = MagicMock()
    resp.json.return_value = payload
    with patch.object(ei.httpx, "post", return_value=resp):
        assert ei.fetch_nldc_index() == {
            "2026-10-13": "files/grdw/2026/10/13.10.26_NLDC_PSP_216.xls"}


def test_index_uses_the_webapps_uppercase_GRDW_source():
    """The grid-india.in bundle posts _source:"GRDW"; pin that exact body so a
    case-only regression cannot silently empty the gap-fill index."""
    resp = MagicMock()
    resp.json.return_value = _index_payload()
    with patch.object(ei.httpx, "post", return_value=resp) as mock_post:
        ei.fetch_nldc_index()
    assert mock_post.call_args.kwargs["json"] == {
        "_source": "GRDW", "_type": ei.NLDC_FILE_TYPE}


def test_index_falls_back_to_the_page_filetype():
    """The webapp reads `_type` from page metadata; a renamed FileType should
    be discovered rather than leaving gap-fill permanently empty."""
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        resp = MagicMock()
        if url == ei.NLDC_PAGE_URL:
            resp.json.return_value = {"retData": [{"FileType": "F_PG00999"}]}
        elif kwargs["json"]["_type"] == "F_PG00999":
            resp.json.return_value = _index_payload()
        else:
            resp.json.return_value = {
                "flagType": 1, "retMessage": "unknown type", "retData": None}
        return resp

    with patch.object(ei.httpx, "post", side_effect=fake_post):
        index = ei.fetch_nldc_index()
    assert index == {
        "2026-10-13": "files/grdw/2026/10/13.10.26_NLDC_PSP_216.xls"}
    assert [body for url, body in calls] == [
        {"_source": "GRDW", "_type": "DAILY_PSP_REPORT"},
        {"_source": "GRDW", "_relPath": ei.NLDC_PAGE_PATH},
        {"_source": "GRDW", "_type": "F_PG00999"},
    ]


def test_index_surfaces_api_flagtype_and_message():
    resp = MagicMock()
    resp.json.return_value = {
        "flagType": 1, "retMessage": "invalid source", "retData": None}
    with patch.object(ei.httpx, "post", return_value=resp):
        with pytest.raises(RuntimeError, match="invalid source") as excinfo:
            ei.fetch_nldc_index()
    assert "flagType" in str(excinfo.value)


def test_index_unreachable_raises_for_the_caller_to_record():
    with patch.object(ei.httpx, "post", side_effect=RuntimeError("down")), \
         patch("vericast.elec.ingest.time.sleep"):
        with pytest.raises(RuntimeError, match="unreachable"):
            ei.fetch_nldc_index()


def test_index_failure_names_the_titles_and_mime_types():
    """A silent upstream switch to .xlsx/PDF must say so, not just 'empty'."""
    resp = MagicMock()
    resp.json.return_value = {"retData": [
        {"Title_": "14.10.26_NLDC_PSP",
         "MimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         "FilePath": "files/grdw/2026/10/14.10.26_NLDC_PSP_216.xlsx"},
    ]}
    with patch.object(ei.httpx, "post", return_value=resp):
        with pytest.raises(RuntimeError, match="no parseable .xls entries") as excinfo:
            ei.fetch_nldc_index()
    assert "14.10.26_NLDC_PSP" in str(excinfo.value)
    assert "spreadsheetml" in str(excinfo.value)


def test_gap_fill_merges_parsed_days_and_skips_unknown_ones(capsys):
    with patch.object(ei, "fetch_nldc_index",
                      return_value={"2026-10-06": "files/x.xls"}), \
         patch.object(ei.httpx, "get") as mock_get, \
         patch.object(ei, "parse_nldc_workbook",
                      return_value=(31330.0, 662.3)) as mock_parse:
        mock_get.return_value.content = b"xls-bytes"
        mock_get.return_value.raise_for_status.return_value = None
        rows = ei.fetch_nldc_demand(["2026-10-06", "2026-10-07"])
    assert rows == {"2026-10-06": (31330.0, 662.3)}
    assert "2026-10-07" not in rows  # no index entry -> left for backfill
    assert mock_parse.call_count == 1


def test_gap_fill_download_failure_skips_the_day():
    with patch.object(ei, "fetch_nldc_index",
                      return_value={"2026-10-06": "files/x.xls"}), \
         patch.object(ei.httpx, "get", side_effect=RuntimeError("cdn down")), \
         patch("vericast.elec.ingest.time.sleep"):
        assert ei.fetch_nldc_demand(["2026-10-06"]) == {}


def test_gap_fill_dead_api_records_reason_instead_of_raising():
    diagnostics = {}
    with patch.object(ei, "fetch_nldc_index",
                      side_effect=RuntimeError("unreachable")), \
         patch.object(ei, "_download_nldc_report",
                      side_effect=RuntimeError("raw mirror down")):
        assert ei.fetch_nldc_demand(["2026-10-06"], diagnostics=diagnostics) == {}
    assert "unreachable" in diagnostics["nldc_index_error"]
    assert diagnostics["nldc_raw_filled"] == 0
    assert diagnostics["nldc_unfilled"] == ["2026-10-06"]


def test_gap_fill_uses_raw_mirror_when_the_api_is_blocked():
    """The official API resets CI datacenter connections; the same official
    .xls files are available raw in the mirror repo, so use them rather than
    leaving the series to go stale."""
    diagnostics = {}
    with patch.object(ei, "fetch_nldc_index",
                      side_effect=RuntimeError("connection reset")), \
         patch.object(ei, "_download_nldc_report",
                      return_value=b"xls-bytes") as mock_download, \
         patch.object(ei, "parse_nldc_workbook",
                      return_value=(32118.0, 677.9)):
        rows = ei.fetch_nldc_demand(["2026-10-14", "2026-10-15"],
                                    diagnostics=diagnostics)
    assert rows == {"2026-10-14": (32118.0, 677.9),
                    "2026-10-15": (32118.0, 677.9)}
    assert [call.args[0] for call in mock_download.call_args_list] == [
        ei.nldc_raw_mirror_url("2026-10-14"),
        ei.nldc_raw_mirror_url("2026-10-15"),
    ]
    assert mock_download.call_args_list[0].args[0].endswith(
        "/14.10.26_NLDC_PSP.xls")
    assert diagnostics["nldc_index_error"] == "connection reset"
    assert diagnostics["nldc_raw_filled"] == 2
    assert diagnostics["nldc_newest_available"] == "2026-10-15"
    assert diagnostics["nldc_unfilled"] == []


def test_gap_fill_records_unfilled_days_and_newest_index_date():
    diagnostics = {}
    with patch.object(ei, "fetch_nldc_index",
                      return_value={"2026-10-13": "files/x.xls"}):
        assert ei.fetch_nldc_demand(["2026-10-14"], diagnostics=diagnostics) == {}
    assert diagnostics["nldc_index_size"] == 1
    assert diagnostics["nldc_newest_available"] == "2026-10-13"
    assert diagnostics["nldc_unfilled"] == ["2026-10-14"]
    assert diagnostics["nldc_filled"] == 0


def test_freshness_gate_fails_at_ingest_with_the_source_status():
    stale = local_time.today("Asia/Kolkata") - timedelta(
        days=ELEC_STALE_LIMIT_DAYS + 1)
    diagnostics = {
        "mirror_head": "2026-09-29",
        "nldc_newest_available": "2026-10-13",
        "nldc_index_error": "API unreachable",
        "nldc_unfilled": ["2026-10-14"],
    }
    with patch.object(ei, "latest_observation_date", return_value=stale):
        with pytest.raises(RuntimeError, match="has stalled") as excinfo:
            ei.assert_fresh_enough(diagnostics)
    message = str(excinfo.value)
    assert "No forecast was published" in message
    assert "mirror_head=2026-09-29" in message
    assert "nldc_newest=2026-10-13" in message
    assert "nldc_unfilled=2026-10-14" in message


def test_freshness_gate_accepts_a_series_within_the_limit():
    fresh = local_time.today("Asia/Kolkata") - timedelta(days=1)
    with patch.object(ei, "latest_observation_date", return_value=fresh):
        assert ei.assert_fresh_enough({}) == 1


def test_freshness_gate_rejects_an_empty_observations_table():
    with patch.object(ei, "latest_observation_date", return_value=None):
        with pytest.raises(RuntimeError, match="No electricity observations"):
            ei.assert_fresh_enough()


def test_gap_fill_is_capped_to_protect_the_ingest_step_budget():
    days = [f"2026-09-{d:02d}" for d in range(1, 21)]  # 20 missing days
    index = {d: f"files/{d}.xls" for d in days}
    with patch.object(ei, "fetch_nldc_index", return_value=index), \
         patch.object(ei.httpx, "get") as mock_get, \
         patch.object(ei, "parse_nldc_workbook", return_value=(30000.0, 600.0)):
        mock_get.return_value.content = b"xls-bytes"
        mock_get.return_value.raise_for_status.return_value = None
        rows = ei.fetch_nldc_demand(days)
    assert len(rows) == ei.NLDC_MAX_DAYS_PER_RUN
    assert mock_get.call_count == ei.NLDC_MAX_DAYS_PER_RUN
    # most recent days win the cap
    assert max(rows) == "2026-09-20"


def test_gap_fill_empty_input_does_no_io():
    with patch.object(ei, "fetch_nldc_index") as mock_index:
        assert ei.fetch_nldc_demand([]) == {}
        assert not mock_index.called
