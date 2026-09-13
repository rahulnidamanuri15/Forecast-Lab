"""Single-file model bundles; legacy text models remain readable.

A bundle contains the model and its metadata in the same JSON document. Readers
open it once, so an atomic replacement cannot mix two generations. New bundles
are authoritative; a corrupt bundle never silently falls back to a legacy model.
"""
import json
import os
import tempfile
from pathlib import Path

import lightgbm as lgb


def bundle_path(model_path):
    return f"{model_path}.bundle.json"


def model_exists(model_path):
    return os.path.isfile(bundle_path(model_path)) or os.path.isfile(model_path)


def read_bundle(model_path):
    try:
        with open(bundle_path(model_path), encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    if (not isinstance(payload, dict) or payload.get("version") != 1
            or not isinstance(payload.get("model"), str) or not payload["model"]
            or not isinstance(payload.get("window"), dict)):
        raise ValueError("Invalid model bundle")
    window = payload["window"]
    if (not all(isinstance(window.get(k), str) for k in ("first", "last"))
            or not isinstance(window.get("rows"), int) or window["rows"] <= 0):
        raise ValueError("Invalid model training window")
    return payload


def load_model(model_path):
    payload = read_bundle(model_path)
    if payload is not None:
        return lgb.Booster(model_str=payload["model"])
    return lgb.Booster(model_file=str(model_path))


def save_bundle(model, model_path, dates, rows):
    if not dates or rows <= 0 or len(dates) != rows:
        raise ValueError("Training dates must match the positive row count")
    window = {"first": str(dates[0]), "last": str(dates[-1]), "rows": int(rows)}
    model_text = model.model_to_string()
    # Validate serialization before publishing the new generation.
    lgb.Booster(model_str=model_text)
    payload = {"version": 1, "model": model_text, "window": window}
    destination = Path(bundle_path(model_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent, prefix=".bundle-",
                                         suffix=".tmp", delete=False) as fh:
            temporary = fh.name
            json.dump(payload, fh, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, destination)
        temporary = None
        # Persist the directory entry on platforms supporting directory fsync.
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return window
