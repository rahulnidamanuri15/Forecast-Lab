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
    """
    for m in re.finditer(r"(textContent|innerHTML)\s*=\s*([^;]{0,200})", SOURCE):
        assert "backtest" not in m.group(2), (
            f"backtest figure assigned to a rendered value: {m.group(0)[:120]}")
    assert SOURCE.count("Launch backtest: ") == 2, (
        "the backtest tooltip disappeared from one of the two panels; it is how "
        "the launch record stays discoverable without becoming the headline")
