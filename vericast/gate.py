"""Retrain evaluation gate: validate candidate model against calibrated quality bars.

Evaluates a challenger model trained on the training head against a 30-day holdout block.
Rejects degraded or broken models using three calibrated criteria:
  1. MAE vs persistence baseline <= 2.0x (prevents massive error regressions).
  2. Prediction spread >= 0.2x of actuals spread (rejects constant/collapsed predictions).
  3. Pearson correlation r > 0.0 with actuals (rejects sign-inverted predictions).

If rejected, the pipeline keeps the incumbent artifact and exits 0 cleanly.
"""
import json
import os

import numpy as np
import lightgbm as lgb

HOLDOUT_DAYS = 30
MIN_TRAIN_ROWS = 60      # below this a 30-day holdout leaves too little to fit

# Calibrated acceptance bars:
MAX_BASELINE_RATIO = 2.0     # Challenger MAE must not exceed 2x persistence baseline
MIN_SPREAD_FRACTION = 0.2    # Standard deviation of predictions must be >= 20% of actuals std dev
MIN_CORRELATION = 0.0        # Pearson correlation between predictions and actuals must be positive

def _mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def window_path(model_path):
    """Sidecar path for `model_path`'s training window."""
    return f"{model_path}.window.json"


def record_training_window(model_path, dates, rows):
    """Record the date range and sample count used to train the model artifact."""
    if not dates:
        return None
    payload = {"first": str(dates[0]), "last": str(dates[-1]), "rows": int(rows)}
    with open(window_path(model_path), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[gate] Recorded training window {payload['first']} -> "
          f"{payload['last']} ({payload['rows']} rows) beside the artifact.")
    return payload


def read_training_window(model_path):
    """Read the recorded training window sidecar, or None if unavailable."""
    try:
        with open(window_path(model_path), encoding="utf-8") as fh:
            payload = json.load(fh)
        return {"first": payload["first"], "last": payload["last"],
                "rows": int(payload["rows"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def challenger_ships(X, y, params, num_boost_round, baseline_col,
                     incumbent_path=None, feature_names=None,
                     holdout_days=HOLDOUT_DAYS, unit="", dates=None):
    """Validate challenger model on holdout set against quality bars before shipping."""
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
            incumbent_mae = _mae(incumbent.predict(X_hold), y_hold)
            # The sidecar is what makes this line honest. Without it every
            # incumbent score had to be caveated as not comparable; with it, an
            # incumbent whose window ends before the holdout starts never saw
            # these rows and its number means what it looks like.
            window = read_training_window(incumbent_path)
            holdout_start = str(dates[split]) if dates is not None else None
            comparable = bool(window and holdout_start
                              and window["last"] < holdout_start)
            if comparable:
                note = (f"trained through {window['last']}, before the holdout "
                        f"opens at {holdout_start} - comparable")
            elif window:
                note = (f"trained through {window['last']}, which covers the "
                        f"holdout - not comparable")
            else:
                note = "no recorded training window, assume it saw them"
            print(f"[gate]   (incumbent scores {incumbent_mae:.4f}{suffix} on the "
                  f"same rows; {note})")
        else:
            print(f"[gate]   (incumbent expects {incumbent.num_feature()} "
                  f"features, data has {X.shape[1]}; feature set changed)")

    ships = all(ok for ok, _ in checks)
    print(f"[gate] {'ACCEPT' if ships else 'REJECT'} - "
          f"{'challenger is fit to ship' if ships else 'keeping the incumbent'}")
    return ships


def demo():
    """Self-check: the six broken models the bars were calibrated against must
    all be refused, and a fit on real signal must pass.

    ponytail: bare `assert` is right here and wrong in the pipeline gates. This is
    a __main__ self-check whose whole job is to fail loudly under CI's plain
    `python -m vericast.gate`; under -O it simply is not the check anymore. The
    gates that guard *data* raise instead, since there -O would silently turn them
    into no-ops that still exit 0.
    """
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

    # The sidecar round-trips, and a missing or corrupt one reads as None rather
    # than raising - the printed comparability line must never fail a retrain.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        artifact = os.path.join(tmp, "model.txt")
        assert read_training_window(artifact) is None, "absent sidecar must be None"
        dates = [f"2026-08-{d:02d}" for d in range(1, 11)]
        record_training_window(artifact, dates, len(dates))
        assert read_training_window(artifact) == {
            "first": "2026-08-01", "last": "2026-08-10", "rows": 10}
        with open(window_path(artifact), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        assert read_training_window(artifact) is None, "corrupt sidecar must be None"
        assert record_training_window(artifact, [], 0) is None, \
            "an empty date list has no window to record"

    print("[OK] gate self-check passed")


if __name__ == "__main__":
    demo()
