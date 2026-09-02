"""The publish gates: a refused forecast must leave the record untouched.

`refuse_stale` and `refuse_implausible` were moved ahead of `conn.commit()`
precisely because diagnose.py, running as step 6/6, could only ever find a bad row
that was already public and already being served by /forecast. That fix is only
worth anything if the raise really does happen before the INSERT, so that is what
these tests assert - not that the guards reject bad numbers (they plainly do), but
that nothing reaches `predictions` / `electricity_predictions` when they fire.

The interesting case is a partial run: the naive baseline publishes from a healthy
observation, and the model's own forecast is the absurd one. One row must land and
the other must not - a gate that aborted the whole run, or committed anyway, would
both pass a test that only looked at the exception.

`psycopg.connect` is patched directly rather than through conftest's wire_cursor,
which exists for app.get_db_connection and does not reach here.
"""
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from vericast import (
    ELEC_MAX_MW,
    ELEC_MIN_MW,
    PM25_MAX,
    PM25_MIN,
    local_time,
    refuse_implausible,
)
from vericast.elec import predict as elec_predict
from vericast.elec.train import FEATURE_COLUMNS as ELEC_FEATURES
from vericast.pm25 import predict as pm25_predict
from vericast.pm25.train import FEATURE_COLUMNS as PM25_FEATURES


def wire_connect(mock_connect, fetches):
    """The cursor reached through a patched psycopg.connect, plus its connection.

    Both predict modules open `with psycopg.connect(...) as conn, conn.cursor() as
    cur:`, so the mock has to be wired through two __enter__s or every execute
    lands on an auto-child and the assertions below pass against nothing.
    `fetches` is the fetchone sequence: the latest observation, then the latest
    features row for the callers that get that far.
    """
    conn = MagicMock()
    cur = MagicMock()
    mock_connect.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cur
    cur.fetchone.side_effect = fetches
    return conn, cur


def upserts(cur, table):
    """Every execute that would write a forecast row to `table`."""
    return [call.args for call in cur.execute.call_args_list
            if f"INSERT INTO {table}" in call.args[0]]


def pm25_feature_row(as_of):
    """A features row that passes every check but the plausibility gate."""
    return (as_of, *[10.0] * len(PM25_FEATURES))


def elec_feature_row(as_of, **overrides):
    """Same, keyed by name so a FEATURE_COLUMNS reorder cannot mislabel it."""
    values = {name: 25_000.0 for name in ELEC_FEATURES}
    values.update(overrides)
    return (as_of, *[values[name] for name in ELEC_FEATURES])


def test_an_out_of_range_pm25_model_forecast_is_not_published():
    """The naive row lands, the model's -/+ absurdity does not.

    A booster predicting 900 ug/m3 means a broken artifact or corrupt features, and
    before this gate moved ahead of the commit it would have been public until
    diagnose.py failed the job two steps later.
    """
    as_of = local_time.today()
    broken_model = MagicMock()
    broken_model.predict.return_value = [PM25_MAX * 2]

    with patch("psycopg.connect") as mock_connect, \
         patch.object(pm25_predict, "load_lightgbm_model", return_value=broken_model):
        conn, cur = wire_connect(mock_connect, [(as_of, 42.0),
                                                pm25_feature_row(as_of)])

        with pytest.raises(RuntimeError, match="outside the plausible range"):
            pm25_predict.make_daily_prediction()

    written = upserts(cur, "predictions")
    assert len(written) == 1, "only the naive baseline should have been published"
    assert written[0][1][-1] == "naive_baseline"
    assert conn.commit.call_count == 1


def test_a_stalled_pm25_source_publishes_nothing_at_all():
    """The anchor slides with the stall, so every downstream check still passes.

    forecast_date is latest_obs + 1 day: a source three days behind produces a
    forecast that is internally consistent and quietly three days out of date.
    """
    stalled = local_time.today() - timedelta(days=3)

    with patch("psycopg.connect") as mock_connect, \
         patch.object(pm25_predict, "load_lightgbm_model", return_value=None):
        conn, cur = wire_connect(mock_connect, [(stalled, 42.0)])

        with pytest.raises(RuntimeError, match="has stalled"):
            pm25_predict.make_daily_prediction()

    assert not upserts(cur, "predictions")
    assert not conn.commit.called


def test_an_out_of_range_elec_feature_is_not_published():
    """demand_lag_6 is a stored observation carried forward, so an absurd value
    means ingest's own guard let one through - and seasonal_naive would have
    published it verbatim."""
    as_of = local_time.today()

    with patch("psycopg.connect") as mock_connect, \
         patch.object(elec_predict, "load_lightgbm_model", return_value=None):
        conn, cur = wire_connect(
            mock_connect,
            [(as_of, 25_000.0),
             elec_feature_row(as_of, demand_lag_6=ELEC_MAX_MW * 3)])

        with pytest.raises(RuntimeError, match="outside the plausible range"):
            elec_predict.make_daily_prediction()

    written = upserts(cur, "electricity_predictions")
    assert len(written) == 1, "only the naive baseline should have been published"
    assert written[0][1][-1] == "naive_baseline"
    assert conn.commit.call_count == 1


def test_a_stalled_elec_source_publishes_nothing_at_all():
    """2-4 days behind is this mirror's normal state; past 5 it has stalled."""
    stalled = local_time.today() - timedelta(days=6)

    with patch("psycopg.connect") as mock_connect, \
         patch.object(elec_predict, "load_lightgbm_model", return_value=None):
        conn, cur = wire_connect(mock_connect, [(stalled, 25_000.0)])

        with pytest.raises(RuntimeError, match="has stalled"):
            elec_predict.make_daily_prediction()

    assert not upserts(cur, "electricity_predictions")
    assert not conn.commit.called


def test_a_null_forecast_is_refused_rather_than_stored():
    """A NULL row 500s /forecast and scores as a NaN MAE that nothing can attribute."""
    with pytest.raises(RuntimeError, match="produced no value"):
        refuse_implausible(None, PM25_MIN, PM25_MAX, "lightgbm", "ug/m3")


def test_the_plausible_range_includes_its_own_endpoints():
    """Exclusive bounds here would refuse a value ingest had already accepted."""
    assert refuse_implausible(PM25_MIN, PM25_MIN, PM25_MAX, "m", "ug/m3") == PM25_MIN
    assert refuse_implausible(PM25_MAX, PM25_MIN, PM25_MAX, "m", "ug/m3") == PM25_MAX
    assert refuse_implausible(ELEC_MIN_MW, ELEC_MIN_MW, ELEC_MAX_MW, "m", "MW") \
        == ELEC_MIN_MW
