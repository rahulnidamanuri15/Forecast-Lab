"""VeriCast pipeline package: one subpackage per target, plus shared `local_time`.

Run every script as a module from the repo root, e.g.
    python -m vericast.pm25.ingest

Model artifact paths are resolved from this file, not from the working
directory, so `python -m ...`, a cron job and the container all find the same
file. They stay committed to the repo (weekly-retrain.yml pushes them back).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PM25 = str(ROOT / "models" / "lightgbm_model.txt")
MODEL_ELEC = str(ROOT / "models" / "lightgbm_elec_model.txt")

# How many days behind an upstream source may fall before it counts as stalled.
# One definition per target, imported by every consumer, because the number is
# read in three different roles that have to agree: the diagnostic gate that
# stops the daily pipeline, the /health flag the dashboard's LIVE pill shows, and
# verify_deployment_readiness.py's go-live check. When those drifted apart the
# go-live check failed on data the pipeline gate passed and /health called fresh.
#
# The two numbers differ because the sources differ, and that is the whole reason
# they are named separately rather than folded into one STALE_LIMIT_DAYS:
#   PM2.5 - Open-Meteo publishes yesterday by ~05:00 UTC, so 1 stale day is the
#           steady state and 2 allows one dropped cron run.
#   Elec  - the HalcyonVector demand mirror normally runs 2-4 days behind real
#           time, so only past 5 has it actually stalled.
#
# Why any limit at all: both predict.py modules anchor forecast_date to
# latest_obs + 1 day, so a stalled source slides the anchor with it and every
# internal-consistency check downstream still passes. These are the only checks
# that look at the clock.
PM25_STALE_LIMIT_DAYS = 2
ELEC_STALE_LIMIT_DAYS = 5

