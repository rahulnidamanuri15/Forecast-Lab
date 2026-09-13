"""Run real CLI checks and HTTP readiness against a local nowcast-only deployment."""
import math
import os
import socket
import subprocess
import sys
import time

import httpx
import psycopg
import pytest

from vericast import local_time
from datetime import timedelta
from test_nowcast_integration import isolated_db


def test_local_deployment_commands_with_nowcasts_only(isolated_db, tmp_path):
    env = {**os.environ, "DATABASE_URL": isolated_db, "PYTHON_DOTENV_DISABLED": "1",
           "ALERT_WEBHOOK_URL": "", "GITHUB_STEP_SUMMARY": "", "PYTHONIOENCODING": "utf-8",
           "FRONTEND_ORIGIN": "http://localhost:8000", "CITY": "Nagpur", "STATE": "Maharashtra"}

    def run(*args):
        result = subprocess.run([sys.executable, *args], env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run("-m", "vericast.schema")
    run("-m", "vericast.schema")
    anchor = local_time.today() - timedelta(days=2)
    with psycopg.connect(isolated_db) as conn:
        for n in range(60):
            day = anchor - timedelta(days=59 - n)
            conn.execute("INSERT INTO observations (city, as_of, pm2_5, pm10, "
                         "temperature_2m_mean, wind_speed_10m_max, precipitation_sum) "
                         "VALUES ('Nagpur', %s, %s, 80, 30, 10, 0)", (day, 50 + 20 * math.sin(n / 4)))
            conn.execute("INSERT INTO electricity_observations (state, as_of, peak_demand_mw, "
                         "energy_met_mu, temperature_2m_mean, temperature_2m_max) "
                         "VALUES ('Maharashtra', %s, %s, 480, 30, 36)", (day, 25000 + 2500 * math.sin(n / 6)))
        # Prior estimates with arriving actuals supply each nowcast leaderboard.
        for model in ("lightgbm", "naive_baseline"):
            conn.execute("INSERT INTO predictions (city, forecast_date, model, predicted_pm2_5, source) "
                         "VALUES ('Nagpur', %s, %s, 50, 'nowcast')", (anchor, model))
            conn.execute("INSERT INTO electricity_predictions "
                         "(state, forecast_date, model, predicted_demand_mw, source) "
                         "VALUES ('Maharashtra', %s, %s, 25000, 'nowcast')", (anchor, model))
    for domain in ("pm25", "elec"):
        for stage in ("features", "leakage_test", "score", "predict", "diagnose"):
            run("-m", f"vericast.{domain}.{stage}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env["API_BASE"] = f"http://127.0.0.1:{port}"
    with (tmp_path / "uvicorn.log").open("w+", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                                   "--port", str(port)], env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    if httpx.get(env["API_BASE"] + "/health", timeout=1).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if server.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("local uvicorn did not become ready")
            assert server.poll() is None, "local uvicorn exited unexpectedly"
            output = run("verify_deployment_readiness.py")
            assert "Total: 23/23 checks passed" in output
        finally:
            server.terminate()
            server.wait(timeout=15)
