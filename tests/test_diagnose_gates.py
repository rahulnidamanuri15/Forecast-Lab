"""The completeness gates must expect exactly what predict.py publishes.

Both diagnose modules FAIL the daily pipeline on a model that did not publish.
Both predict modules deliberately *skip* a model when one input is absent - a
thin-hours day for PM2.5's naive_baseline, a date gap for electricity's
seasonal_naive - and warn instead of raising. When the two disagree the pipeline
goes red on a run that behaved exactly as designed, and the real failure it
exists to catch gets ignored as noise.

`expected_models()` is a plain function in each module for this reason: the rule
is reachable without a database, which is the only way CI can check it.
"""
from vericast.elec.diagnose import expected_models as elec_expected
from vericast.pm25.diagnose import expected_models as pm25_expected


def test_pm25_excuses_naive_baseline_only_on_a_null_observation():
    assert pm25_expected(42.0) == {"naive_baseline", "lightgbm"}
    # Thin-hours day: predict.py skipped naive_baseline, so the gate must not
    # demand it - but lightgbm is still expected, because check 4 owns that.
    assert pm25_expected(None) == {"lightgbm"}


def test_elec_excuses_seasonal_naive_only_on_a_null_lag():
    assert elec_expected(True, 25_000.0) == {
        "naive_baseline", "seasonal_naive", "lightgbm"}
    assert elec_expected(True, None) == {"naive_baseline", "lightgbm"}


def test_elec_excuses_nothing_when_the_features_row_is_stale():
    """A NULL lag read off a stale row says nothing about today's run, and the
    as_of mismatch is already a FAIL - so nothing is excused on that path."""
    assert elec_expected(False, None) == {
        "naive_baseline", "seasonal_naive", "lightgbm"}
