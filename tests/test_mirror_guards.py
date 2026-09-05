"""Upstream value guards: a unit change must fail, a real day must not.

vericast/elec/ingest.py deliberately tracks the demand mirror's `main` so its late
backfills can reach the RESCAN_DAYS hole-filling path. Two guards stand in for the
pin: REQUIRED_CSV_COLUMNS catches a rename before any row is accepted, and
plausible_mw() catches a changed *value* - a unit switch to kW or GW - which no
column-name check can see, and which would otherwise parse fine and enter the
scored record as an ordinary-looking number. plausible_pm25() is the same guard on
the Open-Meteo side, where the risk is mg/m3 or a -999 sentinel served in place of
a null.

Both directions matter here. Too tight and a real hot summer day gets skipped and
never scored; too loose and a x1000 unit change is accepted. So the bounds are
checked against the actual observed ranges, not just against a slogan.

The diagnose modules gate the outgoing *forecast* on the same two pairs, so those
are asserted to be the same objects: a publish gate looser than its ingest guard
would let a value into the record that the record's own ingest would have refused.
"""
from datetime import date
from unittest.mock import MagicMock, patch

from vericast import ELEC_MAX_MW, ELEC_MIN_MW, PM25_MAX, PM25_MIN
from vericast.elec import ingest as elec_ingest
from vericast.elec.ingest import REQUIRED_CSV_COLUMNS, plausible_mw
from vericast.pm25.ingest import plausible_pm25

# The observed 2023-2026 Maharashtra range the bounds were chosen around.
OBSERVED_MIN, OBSERVED_MAX = 20_147.0, 32_419.0

# Nagpur's observed 2023-2026 daily CAMS range, same role.
PM25_OBSERVED_MIN, PM25_OBSERVED_MAX = 4.0, 160.0


def test_the_observed_range_is_accepted():
    """A guard that skips real days would silently thin the record instead."""
    assert plausible_mw(OBSERVED_MIN)
    assert plausible_mw(OBSERVED_MAX)


def test_bounds_leave_headroom_but_not_a_factor_of_ten():
    assert ELEC_MIN_MW < OBSERVED_MIN
    assert ELEC_MAX_MW > OBSERVED_MAX
    assert ELEC_MAX_MW < OBSERVED_MAX * 2


def test_a_unit_change_is_rejected():
    """The failure this guard exists for: kW or GW parses fine as a float."""
    assert not plausible_mw(OBSERVED_MAX * 1000)   # MW -> kW
    assert not plausible_mw(OBSERVED_MAX / 1000)   # MW -> GW
    assert not plausible_mw(0.0)


def test_column_guard_names_every_column_the_loop_reads():
    """A column dropped from this set turns back into a mid-loop KeyError."""
    assert REQUIRED_CSV_COLUMNS == {
        "state", "date", "max_demand_met_mw", "energy_met_mu"}


def test_the_observed_pm25_range_is_accepted():
    assert plausible_pm25(PM25_OBSERVED_MIN)
    assert plausible_pm25(PM25_OBSERVED_MAX)


def test_pm25_bounds_leave_headroom_for_a_bad_diwali_week():
    assert PM25_MIN < PM25_OBSERVED_MIN
    assert PM25_MAX > PM25_OBSERVED_MAX * 2


def test_a_pm25_unit_change_or_sentinel_is_rejected():
    """The failures this guard exists for: all three parse fine as floats."""
    assert not plausible_pm25(PM25_OBSERVED_MAX / 1000)   # ug/m3 -> mg/m3
    assert not plausible_pm25(-999.0)                     # null sentinel
    assert not plausible_pm25(0.0)                        # empty field, not clean air


def test_the_publish_gates_share_the_ingest_bounds():
    """A diagnose.py copy that drifted looser would publish what ingest refused."""
    from vericast.elec import diagnose as elec_diagnose
    from vericast.pm25 import diagnose as pm25_diagnose

    assert (elec_diagnose.MIN_MW, elec_diagnose.MAX_MW) == (ELEC_MIN_MW, ELEC_MAX_MW)
    assert (pm25_diagnose.MIN_PM25, pm25_diagnose.MAX_PM25) == (PM25_MIN, PM25_MAX)


def test_the_model_allowlists_agree():
    """Three hand-maintained model sets per target, with nothing tying them.

    diagnose.EXPECTED_MODELS decides which missing forecast is a failure, app's
    *_MODELS decides which `model=` query is accepted, and the description dicts
    label them. Add a model to one and not the others and the failures are all
    quiet: no diagnose complaint for a model that never published, a 400 on a model
    that does, and an empty description string instead of an error.
    """
    import app
    from vericast.elec import diagnose as elec_diagnose
    from vericast.pm25 import diagnose as pm25_diagnose

    assert (pm25_diagnose.EXPECTED_MODELS == app.PM25_MODELS
            == set(app.MODEL_DESCRIPTIONS))
    assert (elec_diagnose.EXPECTED_MODELS == app.ELEC_MODELS
            == set(app.ELEC_MODEL_DESCRIPTIONS))


# --- Which bad cell loses the day ------------------------------------------------
#
# peak_demand_mw is the scored target and NOT NULL, so a bad one has to lose its
# day. energy_met_mu is nullable and nothing scores against it, so a bad one must
# NOT - discarding the row would throw away a peak that is scored, and the
# RESCAN_DAYS re-read would fail identically every run until the upstream cell
# changed, making the loss permanent.

CSV_HEADER = "state,date,max_demand_met_mw,energy_met_mu\n"


def fetch_rows(csv_body):
    """fetch_demand() over a synthetic mirror covering 2026-08-01 -> 08-31."""
    response = MagicMock()
    response.text = CSV_HEADER + csv_body
    with patch.object(elec_ingest.httpx, "get", return_value=response):
        return elec_ingest.fetch_demand(date(2026, 8, 1), date(2026, 8, 31))


def test_an_unparseable_energy_value_keeps_the_scored_peak():
    """The failure this guards: one stray 'N/A' in a column nothing scores against
    discarding the NOT NULL peak the whole electricity target is measured on."""
    rows = fetch_rows(f"{elec_ingest.STATE},2026-08-02,25000.0,N/A\n")

    assert rows == {"2026-08-02": (25_000.0, None)}


def test_a_blank_energy_value_is_already_null_not_a_skip():
    rows = fetch_rows(f"{elec_ingest.STATE},2026-08-03,26000.0,\n")

    assert rows == {"2026-08-03": (26_000.0, None)}


def test_an_unparseable_peak_still_loses_its_day():
    """The other direction: a demand row without demand has nothing to score."""
    assert fetch_rows(f"{elec_ingest.STATE},2026-08-04,N/A,512.0\n") == {}


def test_one_bad_cell_does_not_abandon_the_rows_around_it():
    rows = fetch_rows(
        f"{elec_ingest.STATE},2026-08-05,25000.0,500.0\n"
        f"{elec_ingest.STATE},2026-08-06,N/A,501.0\n"
        f"{elec_ingest.STATE},2026-08-07,27000.0,oops\n"
    )

    assert rows == {"2026-08-05": (25_000.0, 500.0),
                    "2026-08-07": (27_000.0, None)}
