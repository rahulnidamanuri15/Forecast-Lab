"""Layout invariants — these fail if the package structure drifts.

Cheap and DB-free, so CI runs them. They catch the two things the restructure
made possible to break silently: a model artifact path that no longer resolves,
and a predict module re-declaring FEATURE_COLUMNS instead of importing the
train module's single definition.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vericast import MODEL_ELEC, MODEL_PM25
from vericast.elec import predict as elec_predict, train as elec_train
from vericast.pm25 import predict as pm25_predict, train as pm25_train


def test_model_artifacts_resolve():
    for path in (MODEL_PM25, MODEL_ELEC):
        assert os.path.isfile(path), f"model artifact missing: {path}"
        assert os.path.getsize(path) > 0, f"model artifact empty: {path}"


def test_feature_columns_have_one_definition_per_target():
    # Identity, not equality: a copied list would pass ==, which is exactly the
    # drift these modules used to warn about in comments.
    assert pm25_predict.FEATURE_COLUMNS is pm25_train.FEATURE_COLUMNS
    assert elec_predict.FEATURE_COLUMNS is elec_train.FEATURE_COLUMNS
    assert len(pm25_train.FEATURE_COLUMNS) == 15
    assert len(elec_train.FEATURE_COLUMNS) == 14


def test_predict_uses_the_shared_artifact_path():
    assert pm25_predict.MODEL_PATH == MODEL_PM25
    assert elec_predict.MODEL_PATH == MODEL_ELEC
