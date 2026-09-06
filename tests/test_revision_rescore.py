"""A revised observation must take the published error with it.

Both ingesters re-read the last RESCAN_DAYS days every run and upsert the scored
target, and CAMS reanalysis genuinely revises days inside that window. Neither
score.py revisited a filled row - SCORE_SQL fills `actual_* IS NULL` only - so the
observation a published row was scored against could move while the stored actual,
the error, the MAE and the RMSE computed from it kept the old number, and nothing
recorded that the ground truth had changed. That is a silent retro-fit of the
published record.

Two halves, tested separately because they fail separately:
  * ingest's report_revisions() makes the change loud. Pure function, no database.
  * score's reopen step makes the record follow it. That is SQL, so it is tested
    against a real Postgres, like tests/test_feature_alignment.py.

The database half WRITES a prediction and a model_performance row, so unlike that
file it refuses any non-local host outright rather than only when seeding: these
rows are indistinguishable in shape from the published record they would land in.
"""
import os
import sys
from datetime import date

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = os.getenv("CITY", "Nagpur")

# A date no upstream will ever serve and a model name nothing else writes, so these
# rows cannot collide with a seeded fixture or a real one.
DAY = date(2019, 1, 1)
MODEL = "revision_rescore_test"

PREDICTED, SCORED_AGAINST, REVISED = 50.0, 40.0, 44.0


# --- the loud half: ingest names what it overwrote ---------------------------------

def obs_record(date_str, pm2_5):
    """One insert_observations() record, in column order."""
    return (CITY, date_str, pm2_5, 80.0, 31.0, 11.0, 0.0)


def test_ingest_names_a_revised_day(capsys):
    from vericast.pm25.ingest import report_revisions

    report_revisions({"2026-08-20": 40.0}, [obs_record("2026-08-20", 44.0)])

    out = capsys.readouterr().out
    assert "[revised] 2026-08-20: 40.00 -> 44.00" in out, out
    assert "1 day(s) had their pm2_5 REVISED" in out


def test_ingest_is_silent_when_the_value_did_not_move(capsys):
    """The re-scan re-submits 30 unchanged days every run; those are not revisions."""
    from vericast.pm25.ingest import report_revisions

    report_revisions({"2026-08-20": 40.0}, [obs_record("2026-08-20", 40.0)])

    assert capsys.readouterr().out == ""


def test_ingest_does_not_call_a_first_ingest_a_revision(capsys):
    """A day with no stored value, or a NULL one, is being filled - not overwritten."""
    from vericast.pm25.ingest import report_revisions

    report_revisions({}, [obs_record("2026-08-21", 44.0)])
    report_revisions({"2026-08-22": None}, [obs_record("2026-08-22", 44.0)])

    assert capsys.readouterr().out == ""


def test_ingest_names_a_day_revised_away_to_null(capsys):
    """The direction that loses ground truth: a scored day whose observation is
    withdrawn. score.py cannot re-score it, so naming it is all there is."""
    from vericast.pm25.ingest import report_revisions

    report_revisions({"2026-08-23": 40.0}, [obs_record("2026-08-23", None)])

    assert "[revised] 2026-08-23: 40.00 -> NULL" in capsys.readouterr().out


@pytest.mark.parametrize("module_path,table,target", [
    ("vericast.pm25.ingest", "observations", "pm2_5"),
    ("vericast.elec.ingest", "electricity_observations", "peak_demand_mw"),
])
def test_the_upsert_skips_a_row_it_would_not_change(module_path, table, target):
    """Without this the 30-day re-scan rewrites every day it re-reads, so `rowcount`
    cannot tell a revision from a no-op and created_at stops meaning anything.
    Asserted on the SQL text, where the omission would be, so it runs without a DB."""
    import importlib
    import inspect

    source = inspect.getsource(
        importlib.import_module(module_path).insert_observations)
    assert f"{table}.{target} IS DISTINCT FROM EXCLUDED.{target}" in source, (
        f"{module_path}'s upsert no longer guards its DO UPDATE; an unchanged "
        f"re-scan day is rewritten and a real revision is indistinguishable from it")


# --- the decision the reopen step makes, without SQL ------------------------------

class FakeCursor:
    """Just enough cursor for reopen_revised_actuals: one canned SELECT result, and a
    record of whether the UPDATE was issued."""

    def __init__(self, diverged):
        self._diverged = diverged
        self.executed = []
        self.rowcount = len(diverged)

    def execute(self, sql, params):
        self.executed.append(sql.split()[0])

    def fetchall(self):
        return self._diverged


def test_a_frozen_row_is_reported_but_never_re_opened(capsys):
    """The one case where the published record legitimately keeps a stale actual: the
    observation is gone. Re-opening it would clear the actual and SCORE_SQL could not
    refill it, so a verified day would land back at unverified permanently."""
    from vericast import reopen_revised_actuals, revision_sql

    diverged, reopen = revision_sql("predictions", "observations", "city",
                                    actual="actual_pm2_5",
                                    predicted="predicted_pm2_5", observed="pm2_5")
    cur = FakeCursor([(DAY, MODEL, SCORED_AGAINST, None)])

    assert reopen_revised_actuals(cur, diverged, reopen, CITY, "ug/m3") == 0
    assert cur.executed == ["SELECT"], "the UPDATE ran on a row with no ground truth"
    assert "[frozen]" in capsys.readouterr().out


def test_a_moved_observation_is_reported_and_re_opened(capsys):
    from vericast import reopen_revised_actuals, revision_sql

    diverged, reopen = revision_sql("predictions", "observations", "city",
                                    actual="actual_pm2_5",
                                    predicted="predicted_pm2_5", observed="pm2_5")
    cur = FakeCursor([(DAY, MODEL, SCORED_AGAINST, REVISED)])

    assert reopen_revised_actuals(cur, diverged, reopen, CITY, "ug/m3") == 1
    assert cur.executed == ["SELECT", "UPDATE"]
    assert "[revised]" in capsys.readouterr().out


def test_nothing_diverged_costs_one_query(capsys):
    """The every-run case. It must not issue the UPDATE, which would rewrite rows."""
    from vericast import reopen_revised_actuals, revision_sql

    diverged, reopen = revision_sql("predictions", "observations", "city",
                                    actual="actual_pm2_5",
                                    predicted="predicted_pm2_5", observed="pm2_5")
    cur = FakeCursor([])

    assert reopen_revised_actuals(cur, diverged, reopen, CITY, "ug/m3") == 0
    assert cur.executed == ["SELECT"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("table,key,actual,predicted,observed", [
    ("predictions", "city", "actual_pm2_5", "predicted_pm2_5", "pm2_5"),
    ("electricity_predictions", "state", "actual_demand_mw",
     "predicted_demand_mw", "peak_demand_mw"),
])
def test_the_reopen_query_can_only_clear_a_row_score_sql_will_refill(
        table, key, actual, predicted, observed):
    """Each predicate omitted here erases an actual with no path back, so they are
    asserted rather than trusted to survive an edit."""
    from vericast import revision_sql

    _, reopen = revision_sql(table, table.replace("predictions", "observations"), key,
                             actual=actual, predicted=predicted, observed=observed)

    assert f"SET {actual} = NULL" in reopen
    assert "p.source = 'daily'" in reopen, "a backtest actual could be erased"
    assert f"p.{predicted} IS NOT NULL" in reopen, "SCORE_SQL would not refill it"
    assert f"o.{observed} IS NOT NULL" in reopen, "nothing to re-score against"


# --- the half that keeps the record honest: a real Postgres ------------------------

def _blocked():
    """Why this test cannot run here, or None."""
    if not DATABASE_URL:
        return "DATABASE_URL not set"
    # Before the connection attempt, not after: the point is to never touch it.
    host = conninfo_to_dict(DATABASE_URL).get("host", "")
    if host not in ("localhost", "127.0.0.1", "::1", ""):
        return (f"refusing to write test predictions into a remote database "
                f"(host {host!r}); point DATABASE_URL at a local throwaway Postgres")
    try:
        psycopg.connect(DATABASE_URL, connect_timeout=5).close()
    except Exception as exc:
        return f"database unreachable: {type(exc).__name__}"
    return None


_SKIP = _blocked()


@pytest.fixture
def scored_day():
    """One published, scored day: prediction, its actual, and the perf row from it.

    Committed, because score_pending_predictions() opens its own connection and
    would not otherwise see any of it.
    """
    if _SKIP:
        pytest.skip(_SKIP)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (city, as_of, pm2_5) VALUES (%s, %s, %s) "
                "ON CONFLICT (city, as_of) DO UPDATE SET pm2_5 = EXCLUDED.pm2_5",
                (CITY, DAY, SCORED_AGAINST))
            cur.execute(
                "INSERT INTO predictions "
                "  (city, forecast_date, model, predicted_pm2_5, actual_pm2_5, source) "
                "VALUES (%s, %s, %s, %s, %s, 'daily') "
                "ON CONFLICT (city, forecast_date, model) DO UPDATE SET "
                "  predicted_pm2_5 = EXCLUDED.predicted_pm2_5, "
                "  actual_pm2_5 = EXCLUDED.actual_pm2_5, source = 'daily'",
                (CITY, DAY, MODEL, PREDICTED, SCORED_AGAINST))
            stale = abs(PREDICTED - SCORED_AGAINST)
            cur.execute(
                "INSERT INTO model_performance "
                "  (score_date, model, mae, rmse, sample_size, source) "
                "VALUES (%s, %s, %s, %s, 1, 'daily') "
                "ON CONFLICT (score_date, model) DO UPDATE SET "
                "  mae = EXCLUDED.mae, rmse = EXCLUDED.rmse, source = 'daily'",
                (DAY, MODEL, stale, stale))
            conn.commit()
        try:
            yield conn
        finally:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions WHERE city = %s AND model = %s",
                            (CITY, MODEL))
                cur.execute("DELETE FROM model_performance WHERE model = %s", (MODEL,))
                cur.execute("DELETE FROM observations WHERE city = %s AND as_of = %s",
                            (CITY, DAY))
                conn.commit()


def _revise(conn, to):
    with conn.cursor() as cur:
        cur.execute("UPDATE observations SET pm2_5 = %s WHERE city = %s AND as_of = %s",
                    (to, CITY, DAY))
        conn.commit()


def _row(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT actual_pm2_5 FROM predictions "
                    "WHERE city = %s AND forecast_date = %s AND model = %s",
                    (CITY, DAY, MODEL))
        actual = cur.fetchone()[0]
        cur.execute("SELECT mae, rmse, source FROM model_performance "
                    "WHERE score_date = %s AND model = %s", (DAY, MODEL))
        return (actual, *cur.fetchone())


def test_a_revised_observation_moves_the_actual_and_the_error(scored_day):
    """The finding, end to end: the ground truth moves, so the published error must.

    Before the reopen step, actual_pm2_5 stayed at 40 and mae stayed at 10 - an error
    measured against a number the observations table no longer held, with nothing
    anywhere recording that it had moved.
    """
    from vericast.pm25.score import score_pending_predictions

    _revise(scored_day, REVISED)
    score_pending_predictions()

    actual, mae, rmse, source = _row(scored_day)
    expected = abs(PREDICTED - REVISED)
    assert actual == pytest.approx(REVISED), "the actual did not follow the observation"
    assert mae == pytest.approx(expected), f"mae is {mae}, still measured against {actual}"
    assert rmse == pytest.approx(expected)
    assert source == "daily", "a rescore must not relabel a verified row"


def test_an_unrevised_day_is_left_alone(scored_day):
    """The rescore may only touch days that actually moved: re-opening a row whose
    observation is unchanged would rewrite created_at on the whole record every run."""
    from vericast.pm25.score import score_pending_predictions

    before = _row(scored_day)
    score_pending_predictions()

    assert _row(scored_day) == before


def test_an_observation_revised_to_null_leaves_the_published_actual_standing(scored_day):
    """The dangerous direction. Clearing the actual here would return a published,
    verified row to unverified with no path back - SCORE_SQL cannot refill it, because
    there is no observation to refill it from. So it stays, and ingest says so."""
    from vericast.pm25.score import score_pending_predictions

    _revise(scored_day, None)
    score_pending_predictions()

    actual, mae, _, _ = _row(scored_day)
    assert actual == pytest.approx(SCORED_AGAINST)
    assert mae == pytest.approx(abs(PREDICTED - SCORED_AGAINST))


def test_a_backtest_row_is_never_re_opened(scored_day):
    """Backtest rows are a closed set, seeded with their actuals already in hand, and
    SCORE_SQL filters them out - so re-opening one erases an actual permanently."""
    from vericast.pm25.score import score_pending_predictions

    with scored_day.cursor() as cur:
        cur.execute("UPDATE predictions SET source = 'backtest' "
                    "WHERE city = %s AND model = %s", (CITY, MODEL))
        scored_day.commit()

    _revise(scored_day, REVISED)
    score_pending_predictions()

    actual, *_ = _row(scored_day)
    assert actual == pytest.approx(SCORED_AGAINST), "a backtest actual was re-opened"
