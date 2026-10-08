"""Manually backfill Maharashtra demand observations from official NLDC PSP reports.

Escape hatch for when the Grid-Sentinel CSV mirror stalls: its aggregation last
committed 2026-09-30 (data through 2026-09-29) while the underlying NLDC daily
PSP reports keep publishing. The operator reads Maharashtra's "Maximum Demand
met" (MW) off the official report for each missing day and passes DATE:MW pairs:

    python -m vericast.elec.backfill 2026-09-30:24500 2026-10-01:25123.5
    python -m vericast.elec.backfill 2026-09-30:24500:312.4  # ...with energy MU

Everything except the demand figure itself reuses ingest.py: `plausible_mw`
rejects unit slips, temperatures come from the Open-Meteo archive (full history,
unlike the mirror), and `insert_observations` writes through the same upsert +
revision log. A backfilled row is therefore as trustworthy as a mirror row -
provided the MW values are genuinely read off the NLDC report. A
plausible-but-wrong number can never be detected afterwards, so do not estimate,
interpolate, or carry values forward: leave a day out rather than invent it.

Afterwards run the normal chain so lags, features, scores and the forecast all
advance together:

    python -m vericast.elec.features
    python -m vericast.elec.leakage_test
    python -m vericast.elec.score
    python -m vericast.elec.predict
    python -m vericast.elec.diagnose

Backfill contiguously through yesterday: features are windowed lags
(demand_lag_6 needs t-6 present), so a skipped day NULLs every downstream
window and predict.py skips those model arms with a WARN instead of publishing.
"""
import argparse
import math
import os
from datetime import datetime
from dotenv import load_dotenv

from vericast import (
    local_time,
    require_database_url,
)
from vericast.elec.ingest import (
    INITIAL_START,
    STATE,
    fetch_temperature,
    insert_observations,
    plausible_mw,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

EARLIEST = datetime.strptime(INITIAL_START, "%Y-%m-%d").date()


def parse_entry(raw):
    """Parse one DATE:MW[:MU] entry. Raises ValueError naming the problem."""
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"{raw!r}: expected DATE:MW or DATE:MW:MU (e.g. 2026-09-30:24500)"
        )
    date_str, mw_str = parts[0], parts[1]
    mu_str = parts[2] if len(parts) == 3 else None
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"{raw!r}: date {date_str!r} is not YYYY-MM-DD")
    try:
        mw = float(mw_str)
    except ValueError:
        raise ValueError(f"{raw!r}: demand {mw_str!r} is not a number")
    mu = None
    if mu_str is not None:
        try:
            mu = float(mu_str)
        except ValueError:
            raise ValueError(f"{raw!r}: energy {mu_str!r} is not a number")
        if not (math.isfinite(mu) and mu >= 0):
            raise ValueError(f"{raw!r}: energy {mu_str!r} must be >= 0")
    return day, mw, mu


def validate_entries(raw_entries, yesterday):
    """Validate every entry before anything is written (all-or-nothing).

    Returns [(date, mw, mu)] sorted by date. Raises RuntimeError listing every
    bad entry at once, so one typo cannot land while another aborts the run.
    """
    parsed = []
    errors = []
    for raw in raw_entries:
        try:
            parsed.append((raw, *parse_entry(raw)))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(
            "Refusing backfill; fix these entries:\n  " + "\n  ".join(errors)
        )

    problems = []
    seen = set()
    for raw, day, mw, _ in parsed:
        if day in seen:
            problems.append(f"{raw!r}: duplicate date {day}")
        seen.add(day)
        if day < EARLIEST:
            problems.append(f"{raw!r}: {day} predates {EARLIEST} series start")
        if day > yesterday:
            problems.append(
                f"{raw!r}: {day} is after yesterday ({yesterday}); "
                f"observations cover completed days only"
            )
        if not plausible_mw(mw):
            problems.append(
                f"{raw!r}: {mw} MW is outside the plausible Maharashtra range "
                f"(unit slip? read the MW column, not kW/GW)"
            )
    if problems:
        raise RuntimeError(
            "Refusing backfill; fix these entries:\n  " + "\n  ".join(problems)
        )

    entries = sorted(
        ((day, mw, mu) for _, day, mw, mu in parsed), key=lambda e: e[0]
    )
    for (prev, _, _), (day, _, _) in zip(entries, entries[1:]):
        if (day - prev).days > 1:
            print(f"  [warn] gap {prev} -> {day}: windowed lags NULL across it, "
                  f"so predict.py will skip model arms anchored there. Fill "
                  f"every missing day contiguously when you can.")
            break
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backfill Maharashtra demand observations from official "
                    "NLDC PSP figures (see module docstring).")
    parser.add_argument("entries", nargs="+", help="DATE:MW[:MU] per missing day")
    args = parser.parse_args(argv)

    require_database_url(DATABASE_URL)
    yesterday = local_time.yesterday("Asia/Kolkata")
    entries = validate_entries(args.entries, yesterday)
    start, end = entries[0][0], entries[-1][0]
    print(f"Backfilling {len(entries)} day(s) for {STATE}: {start} -> {end}")

    try:
        temps = fetch_temperature(start, end)
    except RuntimeError as exc:
        print(f"  [warn] temperature backfill failed ({exc}); storing NULL "
              f"temperatures. naive_baseline and seasonal_naive can still "
              f"publish, but LightGBM will skip until temps exist.")
        temps = {}

    records = [
        (STATE, day.isoformat(), mw, mu, *temps.get(day.isoformat(), (None, None)))
        for day, mw, mu in entries
    ]
    insert_observations(records)
    print("Backfill complete. Now run: features, leakage_test, score, "
          "predict, diagnose (see module docstring).")


if __name__ == "__main__":
    main()
