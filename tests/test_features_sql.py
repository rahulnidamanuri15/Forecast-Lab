"""Rolling-window guards must count the column they average, not rows.

`COUNT(*) OVER w7 = 7` asks "seven rows exist"; `AVG` skips NULLs, so a
thin-hours day (a row with a NULL value - ingest.py's MIN_HOURS_PER_DAY) makes a
six-value mean pass as a seven-day one. vericast/pm25/leakage_test.py asserts the
other definition (`len(values) == days`, NULLs excluded), so the two silently
disagreed on exactly the day the guard exists for.

A text assertion, same reason as tests/test_leaderboard.py and
tests/test_dashboard.py: the defect is one token in a SQL template, and this runs
in CI with no database. tests/test_feature_alignment.py executes the real frames
but seeds 40 consecutive *complete* days, so it cannot reach this case.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vericast.elec.features import ENGINEER_SQL as ELEC_SQL
from vericast.pm25.features import ENGINEER_SQL as PM25_SQL

# CASE WHEN COUNT(x) OVER wN = N THEN AGG(y) OVER wN END
GUARD = re.compile(
    r"COUNT\((?P<counted>[\w*]+)\)\s+OVER\s+(?P<win>w\d+)\s*=\s*(?P<n>\d+)\s+"
    r"THEN\s+(?:AVG|MAX|MIN|SUM)\((?P<agg>\w+)\)\s+OVER\s+(?P=win)\b")

SQL = {"pm25": PM25_SQL, "elec": ELEC_SQL}


def test_every_rolling_guard_counts_its_own_column():
    for name, sql in SQL.items():
        guards = GUARD.finditer(sql)
        found = 0
        for m in guards:
            found += 1
            assert m["counted"] == m["agg"], (
                f"{name}: COUNT({m['counted']}) guards {m['agg']} over "
                f"{m['win']} - count the averaged column, or a NULL value "
                f"yields a short mean labelled as a full window")
            assert m["win"] == f"w{m['n']}", (
                f"{name}: {m['win']} compared against {m['n']}")
        assert found >= 3, f"{name}: only {found} rolling guard(s) matched"


def test_no_rolling_guard_counts_rows():
    for name, sql in SQL.items():
        assert "COUNT(*) OVER" not in sql, (
            f"{name}: COUNT(*) OVER counts rows, not values")
