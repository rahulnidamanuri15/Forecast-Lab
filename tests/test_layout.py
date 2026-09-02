"""Layout invariants — these fail if the package structure drifts.

Cheap and DB-free, so CI runs them. They catch the three things the restructure
made possible to break silently: a model artifact path that no longer resolves,
an artifact whose bytes no longer load or whose feature order no longer matches
the list it was trained from, and a predict module re-declaring FEATURE_COLUMNS
instead of importing the train module's single definition.
"""
import os
import sys

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vericast import MODEL_ELEC, MODEL_PM25
from vericast.elec import predict as elec_predict, train as elec_train
from vericast.pm25 import predict as pm25_predict, train as pm25_train


def test_model_artifacts_resolve():
    for path in (MODEL_PM25, MODEL_ELEC):
        assert os.path.isfile(path), f"model artifact missing: {path}"
        assert os.path.getsize(path) > 0, f"model artifact empty: {path}"


def test_model_artifacts_load_and_agree_on_feature_order():
    """The check that was missing when CRLF made both artifacts unloadable.

    LightGBM's parser is line-ending sensitive, so `models/*.txt` checked out
    with CRLF fails every tree header - with a clean `git status` and no other
    test noticing until the pipeline tried to predict. `.gitattributes` marks
    them `-text` to prevent the conversion; this asserts the bytes on disk
    actually load, and that the artifact's own feature names still match the
    list order it was trained from (a reorder would silently mislabel features).
    """
    for path, columns in ((MODEL_PM25, pm25_train.FEATURE_COLUMNS),
                          (MODEL_ELEC, elec_train.FEATURE_COLUMNS)):
        with open(path, "rb") as fh:
            assert b"\r" not in fh.read(), (
                f"{path} has CR bytes; LightGBM cannot parse it. "
                "Check .gitattributes and core.autocrlf.")
        booster = lgb.Booster(model_file=path)
        assert booster.feature_name() == columns, (
            f"{path} feature order differs from FEATURE_COLUMNS")


def test_feature_columns_have_one_definition_per_target():
    # Identity, not equality: a copied list would pass ==, which is exactly the
    # drift these modules used to warn about in comments.
    assert pm25_predict.FEATURE_COLUMNS is pm25_train.FEATURE_COLUMNS
    assert elec_predict.FEATURE_COLUMNS is elec_train.FEATURE_COLUMNS
    assert len(pm25_train.FEATURE_COLUMNS) == 15
    assert len(elec_train.FEATURE_COLUMNS) == 14


def test_the_publish_gates_check_the_columns_the_model_reads():
    """diagnose.py's features-NULL check must generate its column list, not copy it.

    pm25/diagnose.py spelled the 15 columns out as a SQL literal. That passed every
    test here while checking whatever set was current when it was written: rename or
    reorder a feature and the gate goes on asserting the old one, reporting PASS on
    the exact NULL that makes predict.py skip its lightgbm arm.

    Imported inside the test, as tests/test_mirror_guards.py does - these modules
    read CITY at import and refuse a city other than Nagpur.
    """
    from vericast.elec import diagnose as elec_diagnose
    from vericast.pm25 import diagnose as pm25_diagnose

    assert pm25_diagnose.FEATURE_COLUMNS is pm25_train.FEATURE_COLUMNS
    assert elec_diagnose.FEATURE_COLUMNS is elec_train.FEATURE_COLUMNS

    # And the check itself still exists in both. elec/diagnose.py had no
    # features-NULL check at all: a date gap left predict.py skipping LightGBM with
    # a WARN and exit 0, and the only complaint was "lightgbm did not publish" -
    # the symptom, two checks later, with nothing naming the NULL column.
    # A text assertion because the logic sits inside main(), which needs a database;
    # same reason tests/test_features_sql.py asserts against a SQL template.
    for module in (pm25_diagnose, elec_diagnose):
        with open(module.__file__, encoding="utf-8") as fh:
            source = fh.read()
        assert "Latest features row has no NULLs" in source, (
            f"{module.__name__} no longer gates on a NULL feature")


def test_predict_uses_the_shared_artifact_path():
    assert pm25_predict.MODEL_PATH == MODEL_PM25
    assert elec_predict.MODEL_PATH == MODEL_ELEC
