"""Dashboard reads must match the API payload shape.

index.html is the one consumer of /evaluation that no other test covers: it is a
static file with no build step and no test runner, so a shape change in the API
surfaces as em-dashes in the browser and a green suite. That is exactly what
happened when the provenance split landed - `mae` moved from the top level into
`verified`, every `m.mae` read went undefined, and four KPI cards rendered "—".

These are text assertions on purpose, same as tests/test_leaderboard.py: the
defect class is a missing line in a template, and a text assertion runs in CI with
no browser and no database.
"""
import os
import re

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "index.html")

with open(INDEX, encoding="utf-8") as fh:
    SOURCE = fh.read()

# The spread that flattens `verified` up to the top level so the reads below it
# keep working. One per panel: PM2.5 and electricity each load their own
# /evaluation.
SPREAD = "{ ...m, ...(m.verified || {}) }"


def test_both_panels_flatten_the_verified_block():
    assert SOURCE.count(SPREAD) == 2, (
        f"expected the verified spread in both panels, found "
        f"{SOURCE.count(SPREAD)} - a panel reading m.mae straight off "
        f"/evaluation gets undefined and renders an em-dash")


def test_no_panel_reads_an_evaluation_entry_without_flattening_it():
    """`evaluation.evaluation.find(...)` must go through the mapped array.

    This is the regression itself: `.find(m => m.model === 'lightgbm')` called on
    the raw payload returns an entry whose metrics are one level down. Searching
    for the raw access is what catches a future panel added without the spread.
    """
    raw = re.findall(r"evaluation\.evaluation\.find\(", SOURCE)
    assert not raw, (
        f"{len(raw)} read(s) find a model on the raw /evaluation array; map it "
        f"through {SPREAD} first")


def test_backtest_stays_out_of_the_headline_metrics():
    """The backtest may be quoted in a tooltip, never assigned to a KPI value.

    Averaging the launch backtest into the published record is the retro-fitting
    the split exists to prevent, so a `.textContent = ...backtest...` assignment
    is the thing to fail on - a `.title =` is the intended surface.

    Scoped to assignments rather than to the word: the provenance filter below
    names 'backtest' in its own comment doing the opposite job, and a test that
    forbade the string outright would fail on the fix.
    """
    for m in re.finditer(r"(textContent|innerHTML)\s*=\s*([^;]{0,200})", SOURCE):
        assert "backtest" not in m.group(2), (
            f"backtest figure assigned to a rendered value: {m.group(0)[:120]}")
    assert SOURCE.count("Launch backtest: ") == 2, (
        "the backtest tooltip disappeared from one of the two panels; it is how "
        "the launch record stays discoverable without becoming the headline")


def test_backtest_rows_are_filtered_out_of_the_history_tables():
    """/predictions labels provenance without filtering it, so the page must.

    Both tables are headed "logged before actual outcomes were known" and both
    derive their within-tolerance and average-accuracy figures from the same rows.
    A backtest row under that heading is a fitted-after-the-fact number presented
    as a published one - the retro-fitting claim the whole record rests on.

    The filter must be an allowlist. The API renames source 'daily' to 'verified'
    on the way out, so `!== 'backtest'` admits every value that is not that one
    literal - including a third provenance added upstream under a name this page
    has never seen.
    """
    assert "r.source === 'verified'" in SOURCE, (
        "publishedOnly() no longer allowlists verified rows; the history tables "
        "would quote the launch record, or any future provenance, as published "
        "forecasts")

    used = SOURCE.count("publishedOnly(predictions)")
    assert used == 2, (
        f"expected both panels to filter their /predictions payload, found "
        f"{used} call(s) to publishedOnly")

    raw = re.findall(r"(?:const|let)\s+rows\s*=\s*predictions\.predictions", SOURCE)
    assert not raw, (
        f"{len(raw)} history render(s) read predictions.predictions directly; "
        f"route them through publishedOnly() so backtest rows are dropped")


def test_the_improvement_callout_never_renders_a_double_sign():
    """`+${x.toFixed(1)}%` prints "+-4.2%" the first week the model loses.

    Both callouts go through signedPct(), which adds the '+' only when the number
    is not already carrying a '-'.
    """
    assert SOURCE.count("signedPct(improvement)") == 2, (
        "an improvement callout is formatting its own sign again")
    assert "`+${improvement" not in SOURCE, (
        "hardcoded '+' back in front of an improvement figure that can be negative")


def test_the_tablist_reports_which_tab_is_selected():
    """Two tab buttons with no aria-selected announce as neither or both."""
    assert 'role="tablist"' in SOURCE and SOURCE.count('role="tab"') == 2
    assert SOURCE.count('aria-selected') >= 3, (
        "aria-selected must be on both buttons at rest and updated in switchTab; "
        "the class alone is invisible to a screen reader")
    assert "setAttribute('aria-selected'" in SOURCE
