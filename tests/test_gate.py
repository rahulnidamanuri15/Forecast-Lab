"""The retrain gate's branches that `gate.demo()` does not reach.

`python -m vericast.gate` is the calibration self-check - six deliberately-broken
models refused, an honest fit accepted, the sidecar round-tripping - and ci.yml
runs it as its own step. Re-asserting that here would pay the training cost twice
for one claim, so this file covers only what demo() never passes in: an incumbent
artifact on disk, a training-window sidecar beside it, and a holdout with no
variance to predict.

The comparability line is the reason the file exists. It decides whether the
incumbent's printed MAE means anything at all, and it decides it with a lexical
`<` on two date strings - correct for ISO dates, silently wrong for any other
format, with nothing but one printed sentence to notice by.
"""
import os
import re
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np

from vericast import ROOT, MODEL_ELEC, MODEL_PM25, gate

PARAMS = {"objective": "regression", "verbose": -1, "num_leaves": 7,
          "min_data_in_leaf": 5, "random_state": 0}
ROUNDS = 10


def series(n=120, flat_tail=0):
    """demo()'s synthetic shape: mean-reverting with an exogenous driver.

    Column 0 is the persistence feature - y(t), forecasting y(t+1) - which is
    what `baseline_col=0` means to challenger_ships. `flat_tail` pins the last
    values so the holdout has no variance left to predict.
    """
    rng = np.random.default_rng(0)
    driver = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 20.0
    for t in range(1, n):
        y[t] = 20.0 + 0.5 * (y[t - 1] - 20.0) + 5.0 * driver[t] + rng.normal()
    if flat_tail:
        y[-flat_tail:] = 20.0
    X = np.column_stack([np.roll(y, 1), driver, rng.normal(size=n)])
    X[0, 0] = y[0]
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    return X, y, dates


def incumbent(tmp_path, X, y, name="incumbent.txt"):
    """A saved artifact for challenger_ships to score for drift."""
    path = str(tmp_path / name)
    lgb.train(PARAMS, lgb.Dataset(X, label=y),
              num_boost_round=ROUNDS).save_model(path)
    return path


def head(dates):
    """The dates a challenger is fit on, i.e. everything before the holdout."""
    return dates[:len(dates) - gate.HOLDOUT_DAYS]


def test_a_window_ending_before_the_holdout_is_comparable(tmp_path, capsys):
    """The case the sidecar was added for: a retrain skipped for a week or more
    leaves an incumbent that genuinely never saw these rows, so its number can
    be read straight instead of being caveated away."""
    X, y, dates = series()
    path = incumbent(tmp_path, X, y)
    gate.record_training_window(path, head(dates), len(head(dates)))

    gate.challenger_ships(X, y, PARAMS, ROUNDS, 0,
                          incumbent_path=path, dates=dates)

    out = capsys.readouterr().out
    assert f"before the holdout opens at {dates[len(head(dates))]} - comparable" in out
    assert "not comparable" not in out


def test_a_window_covering_the_holdout_is_not_comparable(tmp_path, capsys):
    """One day's difference flips the verdict, which is the whole comparison."""
    X, y, dates = series()
    path = incumbent(tmp_path, X, y)
    gate.record_training_window(path, dates, len(dates))  # trained on everything

    gate.challenger_ships(X, y, PARAMS, ROUNDS, 0,
                          incumbent_path=path, dates=dates)

    out = capsys.readouterr().out
    assert "covers the holdout - not comparable" in out


def test_a_missing_sidecar_assumes_the_incumbent_saw_the_holdout(tmp_path, capsys):
    """Every artifact predating the sidecar is in this state, and the honest
    reading of an unknown window is the pessimistic one."""
    X, y, dates = series()
    path = incumbent(tmp_path, X, y)

    gate.challenger_ships(X, y, PARAMS, ROUNDS, 0,
                          incumbent_path=path, dates=dates)

    assert "no recorded training window, assume it saw them" in capsys.readouterr().out


def test_an_incumbent_on_a_different_feature_set_is_not_scored(tmp_path, capsys):
    """A feature added or dropped makes the two numbers incomparable in a way no
    date window can rescue, so the score is not printed at all."""
    X, y, dates = series()
    path = incumbent(tmp_path, X[:, :2], y)

    gate.challenger_ships(X, y, PARAMS, ROUNDS, 0,
                          incumbent_path=path, dates=dates)

    out = capsys.readouterr().out
    assert "feature set changed" in out
    assert "on the same rows" not in out


def test_a_flat_holdout_skips_the_two_unmeasurable_checks(capsys):
    """Spread and correlation both divide by the actuals' own spread.

    Zero variance is not a broken model, so the gate says the two checks are
    unmeasurable and drops them rather than dividing into a false verdict.
    """
    X, y, _ = series(flat_tail=gate.HOLDOUT_DAYS + 1)

    gate.challenger_ships(X, y, PARAMS, ROUNDS, 0)

    out = capsys.readouterr().out
    assert "flat over the holdout" in out
    assert "prediction spread" not in out
    assert "correlation with actuals" not in out


def test_the_workflow_stages_the_sidecar_the_gate_writes():
    """weekly-retrain.yml must stage a directory, not the four filenames.

    Naming them is what it did, and it never worked: a refused retrain writes no
    sidecar, `git add` exits 128 on a pathspec matching nothing, and under
    `bash -e` that aborts the step before its own "nothing to commit" branch -
    throwing away the sibling target's accepted artifact. So this reads the
    pathspec the workflow actually stages and asserts it covers what
    window_path() builds, rather than agreeing with a filename spelled twice.
    """
    with open(ROOT / ".github" / "workflows" / "weekly-retrain.yml",
              encoding="utf-8") as fh:
        workflow = fh.read()

    staged = re.findall(r"^\s*git add (\S+)$", workflow, re.MULTILINE)
    assert staged, "weekly-retrain.yml stages nothing"

    for path in (MODEL_PM25, MODEL_ELEC):
        sidecar = os.path.relpath(gate.window_path(path), ROOT).replace(os.sep, "/")
        assert any(sidecar.startswith(spec) for spec in staged), (
            f"weekly-retrain.yml stages {staged}, which does not cover "
            f"{sidecar}; the retrain would commit an artifact whose recorded "
            f"training window is the old one")
