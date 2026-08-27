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
