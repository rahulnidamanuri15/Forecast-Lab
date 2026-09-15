# VeriCast — Published-Then-Verified Next-Day Forecasting

[![CI](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml)
[![Daily Pipeline](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml)
[![Readiness Gate](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml)
[![Weekly Retrain](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

VeriCast publishes next-day forecasts for Nagpur PM2.5 and Maharashtra peak power demand, then scores every prediction against arriving observations. Advance forecasts, delayed estimates, and backtests are stored and reported separately — never averaged.

Live API: `https://forecast-lab-2l0q.onrender.com` · Interactive docs: `/docs` · Dashboard: `/dashboard`

## Contents

- [Target domains](#target-domains)
- [How it works](#how-it-works)
- [Model performance](#model-performance)
- [API reference](#api-reference)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Automation](#automation)
- [Provenance and limitations](#provenance-and-limitations)
- [Operations](#operations)
- [Security and license](#security-and-license)
- [Roadmap](#roadmap)

---

## Target domains

| Target | Series | Unit | Models | Upstream source |
|---|---|---|---|---|
| Air Quality | Nagpur PM2.5 | μg/m³ | `lightgbm`, `naive_baseline` | Open-Meteo CAMS Reanalysis |
| Electricity | Maharashtra Peak Demand | MW | `lightgbm`, `naive_baseline`, `seasonal_naive` | Grid-India PSP Reports |

Both targets use isolated tables, daily jobs, and API routes. Both wait for a shared schema-migration job; after that, a stall in one domain does not block the other.

---

## How it works

```
[Upstream Data]
Open-Meteo / Grid-India mirror
       │
       ▼
[Observations Ingest] ──► observations / electricity_observations
       │
       ▼
[Feature Engineering] ──► features / electricity_features (date-addressed SQL windows)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ML Inference]                            [Daily Scorer]
LightGBM + Baselines                      Scores pending rows, re-opens revisions
       │                                  atomically
       ▼                                         │
[Predictions Table] ◄────────────────────────────┘
(source: 'daily', 'nowcast', or 'backtest')
       │
       ▼
[FastAPI Backend] (pool + rate limit + CSP headers)
       │
       ▼
[Dashboard] (Chart.js, dark mode, provenance filter)
```

Core guarantees:

- **No leakage by construction.** SQL windows use date-addressed intervals, not row offsets:
  ```sql
  MAX(peak_demand_mw) OVER (ORDER BY as_of
      RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING)
  ```
  Calendar gaps yield `NULL`, never shifted history.
- **Revision-safe scoring.** Upstream corrections within 30 days re-open the affected rows and re-score actuals + errors in one transaction.
- **Gated retraining.** Weekly challengers must beat persistence by 5% (`vericast/gate.py`), keep ≥20% of actual spread, and correlate positively (`r > 0`) on a 30-day holdout before promotion.
- **Resilient serving.** Pooled Postgres with 10s statement timeouts, 120 req/min per-IP limit (single worker), 5-minute CDN `Cache-Control`, strict CSP.

---

## Model performance

VeriCast reports **advance forecasts** (`verified`), **delayed estimates** (`nowcast`), and **walk-forward backtests** separately. They are never combined. For compatibility, `daily` rows are returned as `verified`; `publication_source` exposes the stored name.

Backtests below are launch records, not live-forecast evidence. Query `GET /evaluation` and `GET /electricity/evaluation` for live metrics.

### Nagpur PM2.5 — backtest 700 days

| Model | Provenance | Scored days | MAE (μg/m³) | RMSE (μg/m³) | Note |
|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 700 | **9.58** | **12.51** | **+13.7% vs baseline** |
| `naive_baseline` | `backtest` | 700 | 11.10 | 14.49 | Persistence ($y_t \to y_{t+1}$) |

### Maharashtra Peak Demand — backtest 1,238 days

| Model | Provenance | Scored days | MAE (MW) | RMSE (MW) | MAPE | Note |
|---|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 1,238 | **773.54** | **1036.30** | **3.00%** | **+21.2% vs baseline** |
| `naive_baseline` | `backtest` | 1,238 | 981.15 | 1309.45 | 3.79% | Persistence |
| `seasonal_naive` | `backtest` | 1,238 | 1154.08 | 1590.27 | 4.48% | Same weekday last week ($y_{t-6}$) |

Committed `.window.json` counts (730 / 1268) include warm-up rows never scored — hence the delta vs scored days.

---

## API reference

All routes are read-only `GET`. Interactive docs at `/docs`.

| Endpoint | Parameters | Description |
|---|---|---|
| `/` | — | Service index + route map |
| `/health` | — | PM2.5 freshness, staleness days, source status |
| `/forecast` | `model=lightgbm`, `source=latest\|daily\|nowcast\|backtest` (default `latest`) | Newest live PM2.5 prediction, never backtest |
| `/history` | `days=30` (1–365) | Recent observations, oldest-first |
| `/leaderboard` | `source=daily` | Latest scored day per model, one provenance |
| `/evaluation` | `days` (optional) | MAE/RMSE split by `verified` / `nowcast` / `backtest` |
| `/predictions` | `model`, `limit` (1–500), `offset`, `scored_only`, `source`, `start_date`, `end_date` | Log with `predictions`, `count` (page), `total` (all matches) |
| `/diagnostics` | — | Bundle status, training window, horizon |
| `/electricity/health` | — | Demand freshness (2–4 day lag is normal) |
| `/electricity/forecast` | `model=lightgbm`, `source=latest\|...` | Newest live demand prediction |
| `/electricity/history` | `days=30` | Peak demand, energy met, temperature |
| `/electricity/leaderboard` | `source=daily` | Latest scored demand day per model |
| `/electricity/evaluation` | `days` (optional) | MAE/RMSE/**MAPE** by provenance |
| `/electricity/predictions` | same as `/predictions` + `error_pct` | Demand log with `count` + `total` |
| `/electricity/diagnostics` | — | Electricity bundle status |
| `/dashboard` | — | Dashboard HTML with CSP headers |

Conventions:

- `source=latest` means newest issuance by `(forecast_date, created_at)` across `daily` + `nowcast`, never backtest. It does **not** prefer `daily` on tied dates. Use `source=daily` for advance-only reads; a 404 then means "no advance forecast on record" — expected when whole-day features arrive late.
- `status=pending|verified` tracks actual availability, not issuance timing. `timing_status=advance_forecast|delayed_estimate`, `publication_source`, `issued_at`, `cutoff`, `target_timezone`, `horizon_days=1`, and `provenance_consistent` describe issuance.
- `source=daily` filters in SQL before `LIMIT`. Post-filtering the payload shrinks the window silently because backtest rows outnumber live rows.
- Empty logs return `200 {"predictions": [], "count": 0, "total": 0}`. Record routes (`/forecast`, `/history`, `/leaderboard`, `/evaluation`) return 404 when the pipeline has produced nothing.
- Limits: **120 req/min per IP** (`429` with `Retry-After`), **5-minute `Cache-Control`** on `200 GET`. The limiter is per-process — deploy one worker (`--workers 1`, the Docker default) or front with Redis/CDN rate limiting when scaling.

---

## Quickstart

### Prerequisites

- Python **3.11** (see `.python-version`; CI, workflows, and Docker all pin it)
- Node.js 18+ (dashboard `node --test` only)
- PostgreSQL 17 (local, Docker, or Neon)

### Option A — Docker Compose (fastest)

```bash
git clone https://github.com/rahulnidamanuri15/Forecast-Lab.git
cd Forecast-Lab
docker compose up --build
# API: http://localhost:8000/dashboard
# DB:  127.0.0.1:5432/vericast (user/pass: vericast/vericast)
```

### Option B — Local virtualenv

```bash
git clone https://github.com/rahulnidamanuri15/Forecast-Lab.git
cd Forecast-Lab
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.lock
```

### Environment

Create `.env` (never commit — already in `.gitignore`):

```env
DATABASE_URL=postgresql://vericast:vericast@127.0.0.1:5432/vericast?sslmode=disable
CITY=Nagpur
STATE=Maharashtra
FRONTEND_ORIGIN=http://localhost:8000
```

See [Configuration](#configuration) for all variables.

### Initialize and verify

```bash
python -m vericast.schema
python -m vericast.schema  # second run proves idempotency
python -m pytest tests -q -ra --cov=vericast --cov=app --cov-fail-under=70
node --test tests/dashboard_nowcasts.test.cjs
python -m vericast.gate
python -m vericast.local_time
```

Against a running API with populated data:

```bash
API_BASE=http://localhost:8000 python verify_deployment_readiness.py
```

### Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
# open http://localhost:8000/dashboard
```

```bash
docker build -t vericast-api .
docker run -p 8000:8000 --env-file .env vericast-api
```

Suite: **219 tests** (Python + Node). Integration tests needing a live DB skip automatically on remote DSNs; CI runs the full suite against a throwaway Postgres 17 service.

### Migration rollout

1. Verify locally, then on CI. Schema migration runs twice in CI before tests.
2. Pause old writers, back up production, record row counts by source. Old writers are incompatible with source-aware keys.
3. Run `python -m vericast.schema` against production, deploy API + workers + dashboard together. The daily workflow's `migrate` job must succeed before ingestion jobs start.
4. Check both `/diagnostics` routes and evaluations/leaderboards, then resume schedules. No retraining required.

---

## Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Postgres DSN. Read lazily; missing value fails requests with a clear error, not an import crash. |
| `CITY` | No | `Nagpur` | Pinned. Any other value is refused until the city migration is universal. |
| `STATE` | No | `Maharashtra` | Pinned. Temperature coordinates and plausibility bounds assume it. |
| `FRONTEND_ORIGIN` | Yes (prod) | `""` | Comma-separated allowed browser origins. Empty allows none — dashboard fetches will be CORS-blocked. |
| `PORT` | No | `8000` | Served port (Render injects its own). |
| `TRUSTED_PROXY_COUNT` | No | `1` | Proxies in front of the API; rightmost-untrusted XFF entry is the client identity. |
| `ALERT_WEBHOOK_URL` | No | — | Slack/Discord webhook for pipeline failures and partial publishes. |
| `API_BASE` | No | `http://localhost:8000` | Readiness gate target. CI default points at production. |

Direct DB connections (tests/scripts) and pooled connections both enforce `statement_timeout=10s`.

---

## Automation

| Workflow | Schedule | Purpose |
|---|---|---|
| `daily-pipeline.yml` | 10:47 + 14:12 IST | Migrate → ingest → features → leakage → score → predict → diagnose, per target. Second cron is catch-up; upserts make it a no-op when the first lands. |
| `weekly-retrain.yml` | Sun + Wed 02:00 UTC | Retrain both targets behind the 5% gate; commits `models/*.bundle.json` back. |
| `readiness-gate.yml` | Sun 06:23 UTC | `verify_deployment_readiness.py` (23 checks) against live DB + API (SELECT/GET only). |
| `ci.yml` | push/PR + Mon 06:00 UTC | Schema ×2, 219 tests, coverage ≥70%, Docker build, gate + timezone self-checks. |
| `scorecard.yml` / `static.yml` | scheduled | Supply-chain and static analysis. |

All Python jobs install `requirements.lock` so CI green implies pipeline identical. The Docker image installs `requirements.txt` (same direct pins, no dev runner) to keep the image lean.

Artifacts: `models/lightgbm_model.txt.bundle.json` and `models/lightgbm_elec_model.txt.bundle.json` are authoritative. Legacy bare `.txt` files are readable for backward compatibility but labeled `legacy`; a corrupt bundle never silently falls back. Retraining deletes the stale legacy file once the bundle lands.

---

## Provenance and limitations

- Horizon is strictly `features(t) → target(t+1)`. Values and issuance timestamps are immutable; actuals/errors update on upstream revisions.
- PM2.5 target days are **UTC**; electricity days are **Asia/Kolkata**.
- Whole-day `features(t)` complete only after day `t` ends, so live output is structurally a **delayed estimate** (`nowcast`) more often than an advance forecast. This is expected, not a failure.
- Electricity upstream lags 2–4 days normally (`ELEC_STALE_LIMIT_DAYS=5`); PM2.5 limit is 2 days. `stale_days` of 2–4 on electricity is healthy.
- `0.0` PM2.5 is rejected as an empty-field sentinel, not clean air. Bounds `1–500 μg/m³` and `15,000–40,000 MW` catch unit changes (mg/m³, kW/GW) before they enter the scored record.
- Longer horizons need separate training + backtesting; this release does not attempt them.

---

## Operations

- **Scaling.** One worker is the supported topology (`--workers 1`). The limiter is in-memory; each extra worker/instance multiplies the effective 120/min budget. Add Redis or CDN limiting before scaling out.
- **Monitoring.** Watch `/health`, `/electricity/health`, both `/diagnostics`, and the readiness gate. Set `ALERT_WEBHOOK_URL` for pipeline/partial-publish alerts (`GITHUB_STEP_SUMMARY` also receives them in Actions).
- **Troubleshooting.**
  - `404 /forecast?source=daily` with `200 ?source=latest` → only nowcasts exist. Normal.
  - `503` on first boot → DB unreachable; pool boot check logs a warning and serves through it.
  - Migration `Provenance migration blocked` → a daily row collides with an existing nowcast. Resolve duplicates before re-running `python -m vericast.schema`. Never roll old writers forward onto the new schema.
  - CORS-blocked dashboard but healthy `curl` → set `FRONTEND_ORIGIN` to the exact browser origin.

---

## Security and license

- Report vulnerabilities privately per [.github/SECURITY.md](.github/SECURITY.md).
- License: [MIT](LICENSE).

---

## Roadmap

- [ ] Coverage 70 → 80, mutation testing on scoring paths
- [ ] Redis-backed rate limiting + structured JSON logs / OpenTelemetry
- [ ] Staging environment + backup/restore drill automation
- [ ] Calibration + prediction intervals per target
- [ ] Multi-horizon models as separate trained artifacts
- [ ] Dashboard accessibility audit and visual regression tests

See [CHANGELOG.md](CHANGELOG.md) for release history.
