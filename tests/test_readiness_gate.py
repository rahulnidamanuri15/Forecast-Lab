"""The go-live gate's own wiring, checked without a database or a live server.

verify_deployment_readiness.py cannot run in ci.yml: its DB checks need a
populated instance and its HTTP checks a server on API_BASE. It runs weekly in
.github/workflows/readiness-gate.yml against the live database and the deployed
API; what is left over for CI is this file. A renamed check, a check written and
never wired into the list, or a TARGETS key dropped from the descriptor all
surfaced as a traceback on the run that was supposed to be the safety net.

What is checkable here is everything except the queries themselves: the list is
complete, both targets get the four DB checks the elec half used to lack, every
check degrades to False rather than raising when its dependency is absent, the
gate is actually scheduled somewhere, and the README's advertised count still
matches the code.
"""
import os
import re
from unittest.mock import MagicMock, patch

import httpx
import pytest

import verify_deployment_readiness as gate
from vericast import PM25_CITY_OF_RECORD, ROOT, require_city_of_record

# Every key the parameterised checks index out of a TARGETS descriptor. Spelled
# out rather than derived from the descriptors themselves, which would make the
# test agree with whatever is there - including a key dropped from both.
DESCRIPTOR_KEYS = {"artifact", "observations", "features", "feature_columns",
                   "predictions", "key", "key_value", "value", "fmt", "unit",
                   "models"}


def wired_check_names():
    """The check functions CHECKS actually calls, lambdas included.

    A lambda's co_names holds the name it closes over, so
    `lambda: check_model_artifact("electricity")` reports check_model_artifact.
    """
    names = set()
    for _, fn in gate.CHECKS:
        if fn.__name__ == "<lambda>":
            names.update(fn.__code__.co_names)
        else:
            names.add(fn.__name__)
    return names


def test_every_check_function_is_wired_into_the_list():
    """A check that exists but is not in CHECKS is dead code that gates nothing.

    Silent in the only direction that matters: the summary still prints ALL CHECKS
    PASSED, one check short.
    """
    defined = {name for name in vars(gate)
               if name.startswith("check_") and callable(getattr(gate, name))}
    assert defined == wired_check_names()


def test_both_targets_get_the_four_database_checks():
    """The gap fix #10 closed: the elec artifact, its daily prediction and its
    forecast-date anchor were unverified, and a missing lightgbm_elec_model.txt
    passed go-live to show up as a 404 on /electricity/forecast. The features NULL
    check was the one left behind - hardcoded to PM2.5, so a NULL in
    electricity_features passed go-live the same way."""
    labels = [name for name, _ in gate.CHECKS]
    for target in gate.TARGETS:
        for kind in ("artifact exists", "prediction exists", "Forecast date logic",
                     "features have no NULLs"):
            assert any(kind in label and target in label for label in labels), \
                f"no {kind!r} check for {target}"


def test_every_descriptor_carries_every_key_the_checks_read():
    """An f-string over a missing key is a KeyError mid-gate, not a FAIL line."""
    for target, descriptor in gate.TARGETS.items():
        assert DESCRIPTOR_KEYS <= set(descriptor), \
            f"{target} descriptor is missing {DESCRIPTOR_KEYS - set(descriptor)}"
        assert descriptor["models"], f"{target} has no models to check for"


def test_no_check_raises_when_its_dependency_is_missing(monkeypatch):
    """Every check must return a bool, so the summary can name what failed.

    main() catches exceptions, but only as "Unexpected error in check" - which
    tells whoever is deploying that something broke, not that the database is
    unreachable. Run the whole list with no DSN, no server and a failing
    subprocess: the artifact checks legitimately pass, everything else must FAIL
    cleanly.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)

    failed_subprocess = MagicMock(returncode=1, stdout="", stderr="no database")
    with patch.object(gate.httpx, "get",
                      side_effect=httpx.ConnectError("nothing listening")), \
         patch.object(gate.subprocess, "run", return_value=failed_subprocess):
        for name, check in gate.CHECKS:
            result = check()
            assert isinstance(result, bool), f"{name} returned {result!r}, not a bool"


def test_the_gate_is_actually_scheduled_somewhere():
    """A 22-check gate nothing runs is documentation, not a gate.

    This file can only assert the list is wired correctly; the checks themselves
    need a populated database and a live API_BASE, which is why they run in their
    own workflow rather than in ci.yml. If that workflow stops invoking the
    script, every test here still passes and nothing executes a single check.
    """
    workflow = ROOT / ".github" / "workflows" / "readiness-gate.yml"
    assert workflow.is_file(), "readiness-gate.yml is gone; nothing runs the gate"

    text = workflow.read_text(encoding="utf-8")
    assert os.path.basename(gate.__file__) in text, (
        "readiness-gate.yml no longer invokes the gate script")
    # Both halves of what makes the run meaningful: a schedule so it happens
    # unattended, and API_BASE so the HTTP checks reach the deploy rather than a
    # localhost that isn't listening.
    assert "schedule:" in text and "cron:" in text, (
        "the gate workflow has no schedule; it would only ever run by hand, "
        "which is the state having a workflow was meant to fix")
    assert "API_BASE" in text, (
        "no API_BASE in the gate workflow: the six HTTP checks would hit the "
        "localhost default and FAIL on a healthy deploy")


def test_the_readme_advertises_the_number_of_checks_that_exist():
    """The count drifted from 17 to 21 unnoticed, because it is hand-copied prose.

    ponytail: a regex over the README rather than generating that line. One
    sentence in one file; a docs generator for it would be the larger thing.
    """
    with open(ROOT / "README.md", encoding="utf-8") as fh:
        readme = fh.read()

    stated = re.search(r"verify_deployment_readiness\.py.*?\((\d+) checks\)", readme)
    assert stated, "README no longer states the gate's check count"
    assert int(stated.group(1)) == len(gate.CHECKS)


def test_the_gate_runs_only_under_the_city_of_record():
    """A gate that passes under a CITY the API refuses to boot under is worse
    than no gate: model_performance has no city column, so another city's rows
    land on Nagpur's keys and overwrite the published record in place."""
    assert gate.CITY == PM25_CITY_OF_RECORD

    with pytest.raises(RuntimeError, match="single-city"):
        require_city_of_record("Mumbai")


def test_the_readiness_gate_is_excluded_from_the_image():
    """It imports httpx and shells out to the leakage tests; neither belongs in
    the served container, and .dockerignore is the only thing keeping it out."""
    with open(ROOT / ".dockerignore", encoding="utf-8") as fh:
        assert os.path.basename(gate.__file__) in fh.read()
