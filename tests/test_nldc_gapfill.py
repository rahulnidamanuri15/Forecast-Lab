"""Official NLDC gap-fill: same guards as the CSV mirror path, tested without network.

vericast/elec/ingest.py now retries dates the community mirror lacks against
Grid-India's own daily PSP reports (API index + .xls on webcdn). These tests
pin the safety properties with stubbed workbooks and mocked transports:
a misdated, unit-slipped or reshaped report must skip its day, never poison
the scored record - and a dead API must degrade to mirror-only, never fail
ingest (step 1/6).
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

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


def test_index_unreachable_raises_for_the_caller_to_downgrade():
    with patch.object(ei.httpx, "post", side_effect=RuntimeError("down")), \
         patch("vericast.elec.ingest.time.sleep"):
        with pytest.raises(RuntimeError, match="unreachable"):
            ei.fetch_nldc_index()


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


def test_gap_fill_dead_api_returns_empty_instead_of_raising():
    with patch.object(ei, "fetch_nldc_index",
                      side_effect=RuntimeError("unreachable")):
        assert ei.fetch_nldc_demand(["2026-10-06"]) == {}


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
