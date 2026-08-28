"""Retrain gate: refuse to ship a model that is broken.

weekly-retrain.yml used to retrain, commit the bytes, and push - no holdout, no
check, no rollback. A Sunday where upstream shipped garbage silently replaced a
working model and the daily pipeline picked it up on next checkout.

The gate holds out the tail of the series, fits a challenger on the head only,
and scores it on rows it has never seen. A refused retrain is not a failure: the
caller keeps the old artifact, prints why, and exits 0, so the workflow's
existing `git diff --cached --quiet` branch reports "nothing to commit" on its
own.

Why this does NOT vote on "challenger beats the incumbent artifact", which is
the obvious design and was the first one here: the incumbent on disk was trained
on *all* rows, including the ones held out. Measured on the live data that
advantage is 1.5x on PM2.5 (3.33 vs 4.93 ug/m3) and 2.0x on electricity (585 vs
1148 MW) - the incumbent scores better on the holdout than on its own training
head - so an honest challenger essentially cannot win and the gate freezes the
incumbent forever instead of protecting against a bad one. The incumbent's
training range is not stored anywhere, so the comparison cannot be made fair.
Its number is still printed, for drift, but it does not vote.

What votes instead is three absolute checks, calibrated against six
deliberately-broken models (shuffled labels, 1-round stump, constant label,
sign-flipped, 3x-scaled) on both live targets. No single check covers both
targets: MAE-vs-persistence misses every degenerate elec model (a constant-label
fit scores 0.78x of persistence there), and correlation misses a pure scale bug.
Together, each of the six fails at least one check while both honest fits pass
all three with margin.

ponytail: MAE on one trailing block, not k-fold or a significance test. 30 days
is enough to catch garbage, which is what this gate is for. Head-only rather
than walk-forward for the same reason: refitting per holdout day costs 30 fits
(2s PM2.5, 6s elec) and moved the ratio by 0.02x. Reach for walk-forward
(experiments/save_elec_backtest_results.py has the mechanics) only if borderline
retrains start getting refused for noise.
"""
import os

import numpy as np
import lightgbm as lgb

HOLDOUT_DAYS = 30
MIN_TRAIN_ROWS = 60      # below this a 30-day holdout leaves too little to fit

# Bars, with the measured margins that set them. Honest fits land at 1.03x
# (PM2.5) and 0.85x (elec) of persistence; the worst garbage that MAE is the
# only check to catch is 7.7x / 37x. 2.0x sits clear of both, and covers the
# 2.1x window-to-window swing an honest fit shows across 8 past holdouts.
MAX_BASELINE_RATIO = 2.0
# Honest predictions vary by 1.22x (PM2.5) / 0.48x (elec) of the holdout
# actuals' own spread; a constant fit gives 0.00x, a 1-round stump 0.08x / 0.05x.
MIN_SPREAD_FRACTION = 0.2
# Honest r is +0.32 / +0.51; shuffled and sign-flipped fits go negative (-0.26
# to -0.51). Sign alone separates them, so the bar is zero.
MIN_CORRELATION = 0.0

def _mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def challenger_ships(X, y, params, num_boost_round, baseline_col,
                     incumbent_path=None, feature_names=None,
                     holdout_days=HOLDOUT_DAYS, unit=""):
    """True if a model retrained today is fit to overwrite the artifact on disk.

    `baseline_col` is the column index of the persistence feature - y(t), used to
    forecast y(t+1) - which is the do-nothing forecast every model must beat.
    Callers pass `FEATURE_COLUMNS.index("pm2_5_lag_1")` or its elec equivalent.

    Passes (True) when there is too little data to hold anything out; refusing
    then would mean never shipping a first model. X must be ordered oldest-first,
    which DATASET_SQL's `ORDER BY f.as_of` already guarantees.
    """
    if len(X) < MIN_TRAIN_ROWS + holdout_days:
        print(f"[gate] Only {len(X)} rows; need "
              f"{MIN_TRAIN_ROWS + holdout_days} to hold out {holdout_days}. "
              "Accepting without a gate.")
        return True

    split = len(X) - holdout_days
    X_head, y_head = X[:split], y[:split]
    X_hold, y_hold = X[split:], y[split:]

    challenger = lgb.train(
        params,
        lgb.Dataset(X_head, label=y_head, feature_name=feature_names or "auto"),
        num_boost_round=num_boost_round,
    )
    pred = challenger.predict(X_hold)

    suffix = f" {unit}" if unit else ""
    print(f"[gate] Last {holdout_days} rows held out (fit on {len(X_head)}):")

    checks = []

    baseline_mae = _mae(X_hold[:, baseline_col], y_hold)
    challenger_mae = _mae(pred, y_hold)
    ratio = challenger_mae / baseline_mae if baseline_mae else float("inf")
    checks.append((
        ratio <= MAX_BASELINE_RATIO,
        f"MAE {challenger_mae:.4f}{suffix} vs persistence {baseline_mae:.4f}"
        f"{suffix} = {ratio:.2f}x (bar {MAX_BASELINE_RATIO:.1f}x)"))

    # A holdout with no variance to predict makes the next two unmeasurable. Say
    # so and skip rather than dividing by zero into a false verdict.
    actual_sd = float(np.std(y_hold))
    if actual_sd < 1e-9:
        print("[gate]   actuals are flat over the holdout; spread and "
              "correlation not measurable, skipped")
    else:
        spread = float(np.std(pred)) / actual_sd
        checks.append((
            spread >= MIN_SPREAD_FRACTION,
            f"prediction spread {spread:.2f}x the actuals' "
            f"(bar {MIN_SPREAD_FRACTION:.2f}x)"))

        r = (float(np.corrcoef(pred, y_hold)[0, 1])
             if np.std(pred) > 1e-9 else 0.0)
        checks.append((
            r > MIN_CORRELATION,
            f"correlation with actuals r={r:+.3f} (bar > {MIN_CORRELATION:.1f})"))

    for ok, detail in checks:
        print(f"[gate]   [{'OK' if ok else 'FAIL'}] {detail}")

    # Drift only, no vote - see the module docstring for why it cannot vote.
    if incumbent_path and os.path.exists(incumbent_path):
        incumbent = lgb.Booster(model_file=incumbent_path)
        if incumbent.num_feature() == X.shape[1]:
            print("[gate]   (incumbent scores "
                  f"{_mae(incumbent.predict(X_hold), y_hold):.4f}{suffix} on the "
                  "same rows, but it trained on them - not comparable)")
        else:
            print(f"[gate]   (incumbent expects {incumbent.num_feature()} "
                  f"features, data has {X.shape[1]}; feature set changed)")

    ships = all(ok for ok, _ in checks)
    print(f"[gate] {'ACCEPT' if ships else 'REJECT'} - "
          f"{'challenger is fit to ship' if ships else 'keeping the incumbent'}")
    return ships


def demo():
    """Self-check: the six broken models the bars were calibrated against must
    all be refused, and a fit on real signal must pass."""
    rng = np.random.default_rng(0)
    n = 300
    # Mean-reverting with an exogenous driver, like both live targets (weather
    # moves PM2.5 and demand). A pure random walk would make persistence
    # near-optimal and nothing would clear the MAE bar, honest fits included.
    driver = rng.normal(size=n)
    y = np.empty(n)
    y[0] = 20.0
    for t in range(1, n):
        y[t] = 20.0 + 0.5 * (y[t - 1] - 20.0) + 5.0 * driver[t] + rng.normal()
    # Column 0 is the persistence feature: y(t), used to forecast y(t+1).
    X = np.column_stack([np.roll(y, 1), driver, rng.normal(size=n)])
    X[0, 0] = y[0]
    params = {"objective": "regression", "verbose": -1, "num_leaves": 7,
              "min_data_in_leaf": 5, "random_state": 0}
    head = n - HOLDOUT_DAYS

    def ships(mangle=None, rounds=100):
        labels = y.copy()
        if mangle is not None:
            labels[:head] = mangle(labels[:head])
        return challenger_ships(X, labels, params, rounds, 0)

    assert ships(), "an honest fit on real signal must ship"
    assert not ships(lambda a: rng.permutation(a)), \
        "shuffled labels must be refused"
    assert not ships(lambda a: -a + 2 * a.mean()), \
        "sign-flipped labels must be refused"
    assert not ships(lambda a: np.full(len(a), a.mean())), \
        "a constant label must be refused"
    assert not ships(lambda a: a * 3), "a 3x scale bug must be refused"
    assert not ships(rounds=1), "a 1-round stump must be refused"
    assert challenger_ships(X[:50], y[:50], params, 100, 0), \
        "too few rows to hold out must accept"

    print("[OK] gate self-check passed")


if __name__ == "__main__":
    demo()
