# VeriCast — Published-Then-Verified Next-Day Forecasting

[![CI](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml)
[![Daily Pipeline](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml)
[![Readiness Gate](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml)
[![Weekly Retrain](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Production machine learning predictions retain the trained **features(t) → target(t+1)** horizon. Runs issued before target-day midnight are advance forecasts (`daily`); runs at or after that cutoff are **nowcasts / delayed estimates** (`nowcast`), even if the target day has already ended. PM2.5 uses UTC target days; electricity uses Asia/Kolkata. Prediction values and issuance timestamps are immutable on retry. Actuals and errors can be updated when upstream observations are revised.

Structural note: whole-day features for day `t` are only complete after `t` ends, so a whole-day `features(t) → target(t+1)` model cannot issue before the target midnight in production — live output is structurally a delayed estimate. `/forecast` therefore defaults to `source=latest` (newest daily/nowcast row, never backtest); pass `source=daily` explicitly for advance-only reads and treat a 404 as "no advance forecast on record."

The electricity source normally lags by 2–4 days, so its t+1 output is generally a delayed estimate, not tomorrow's forecast. Extending the horizon requires separate training and backtesting; this release does not do that.

---

## 🎯 Target Domains

| Target | Series | Unit | Models Evaluated | Upstream Source |
|---|---|---|---|---|
| **Air Quality** | Nagpur PM2.5 | μg/m³ | `lightgbm`, `naive_baseline` | Open-Meteo CAMS Reanalysis |
| **Electricity** | Maharashtra Peak Demand | MW | `lightgbm`, `naive_baseline`, `seasonal_naive` | Grid-India PSP Reports |

Both pipelines use isolated tables, daily jobs, and API routes. Both wait for a shared schema-migration prerequisite; after that, a source stall in one domain does not block the other.

---

## 🏛️ System Architecture

```
[Upstream Data]
Open-Meteo / Grid-Sentinel
       │
       ▼
[Observations Ingest] ──► observations / electricity_observations
       │
       ▼
[Feature Engineering] ──► features / electricity_features (Strict date-addressed SQL window frames)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ML Inference]                            [Daily Scorer]
LightGBM + Baselines                      Evaluates pending rows
       │                                  Atomically handles revisions
       ▼                                         │
[Predictions Table] ◄────────────────────────────┘
(source: 'daily', 'nowcast', or 'backtest')
       │
       ▼
[FastAPI Backend] ──(Connection Pool + Rate Limiting + CSP)
       │
       ▼
[Interactive Dashboard] (Responsive Chart.js UI, Dark Mode, Provenance Filter)
```

---

## 📊 Model Performance Benchmarks

VeriCast separates **advance forecasts**, **nowcasts / delayed estimates**, and **walk-forward backtests**. Their evaluations and leaderboard records are **never combined or averaged**. For API compatibility, the `daily` source is returned as `verified`; this provenance alias is independent of whether a row has been scored. `publication_source` exposes the stored source name.

The historical backtest benchmarks below are not advance-forecast results. Previously quoted live metrics must be re-read from the API after the provenance migration; they are not evidence that predictions preceded the cutoff.

### 1. Nagpur PM2.5 (Air Quality)
*Backtest: 700 days | Verified: Live continuous tracking*

| Model | Provenance | Scored Days | MAE (μg/m³) | RMSE (μg/m³) | Performance Note |
|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 700 | **9.58** | **12.51** | **+13.7% better than baseline** |
| `naive_baseline` | `backtest` | 700 | 11.10 | 14.49 | Persistence baseline ($y_t \to y_{t+1}$) |

### 2. Maharashtra Peak Electricity Demand
*Backtest: 1,238 days | Verified: Live continuous tracking*

| Model | Provenance | Scored Days | MAE (MW) | RMSE (MW) | MAPE | Performance Note |
|---|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 1,238 | **773.54** | **1036.30** | **3.00%** | **+21.2% better than baseline** |
| `naive_baseline` | `backtest` | 1,238 | 981.15 | 1309.45 | 3.79% | Persistence ($y_t \to y_{t+1}$) |
| `seasonal_naive` | `backtest` | 1,238 | 1154.08 | 1590.27 | 4.48% | Same weekday last week ($y_{t-6}$) |

> Call `GET /evaluation` and `GET /electricity/evaluation` for live, real-time metrics.
> Scored days above are walk-forward evaluated days; committed `.window.json` row counts (730 / 1268) include warm-up/training rows that were never scored — hence the delta.

---

## 🛡️ Core Engineering Guarantees

- **No Future Leakage by Construction**: SQL window frames use date-addressed intervals rather than row offsets:
  ```sql
  MAX(peak_demand_mw) OVER (ORDER BY as_of
      RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING)
  ```
  Any calendar gap naturally results in `NULL` instead of reaching into invalid history.
- **Transparent Revision Handling**: Upstream revisions within 30 days are automatically detected and re-opened. Errors and actuals are recalculated within a single database transaction so the published record always matches verified ground truth.
- **Automated Quality Gates**: Weekly retrain challenger models (`vericast.gate`) must beat persistence baselines, maintain variance ($\ge 20\%$ of actuals), and correlate positively ($r > 0$) on a 30-day holdout before deployment.
- **Operational Resiliency**:
  - `psycopg_pool.ConnectionPool` (Neon serverless PostgreSQL) with 10s statement timeouts.
  - In-memory rate limiter (120 req/min per IP, rightmost-XFF identity, CORS-aware 429s) and 5-minute `Cache-Control` response headers (CDN; no server-side cache — every uncached load hits Postgres).
  - Security headers enforced (CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`).

---

## 📁 Repository Structure

```
├── app.py                            # FastAPI application serving both targets
├── index.html                        # Chart.js frontend dashboard
├── verify_deployment_readiness.py    # Pre-flight deployment readiness gate (23 checks)
├── models/                           # Serialized LightGBM models, bundles + window metadata
│   ├── lightgbm_model.txt[.bundle.json][.window.json]       # Nagpur PM2.5 model
│   └── lightgbm_elec_model.txt[.bundle.json][.window.json]  # Maharashtra electricity model
├── vericast/                         # Shared core package
│   ├── __init__.py                   # Locks, plausibility bounds, revision/alignment SQL
│   ├── local_time.py                 # Timezone authority (Asia/Kolkata)
│   ├── gate.py                       # Retrain evaluation and artifact release gate
│   ├── schema.py                     # Idempotent database DDL & migrations
│   ├── publication.py                # Advance/nowcast issuance policy
│   ├── artifacts.py                  # Atomic model bundles + metadata
│   ├── pm25/                         # PM2.5 pipeline modules
│   │   ├── ingest.py                 # Fetches CAMS reanalysis & weather
│   │   ├── features.py               # SQL window feature engineering
│   │   ├── leakage_test.py           # Temporal leakage validation
│   │   ├── train.py                  # Model training & feature column authority
│   │   ├── predict.py                # Daily forecasting & plausibility filters
│   │   ├── score.py                  # Verification scoring & rescoring
│   │   └── diagnose.py               # Health & freshness validation
│   └── elec/                         # Electricity pipeline modules (identical symmetry)
│       └── [ingest, features, leakage_test, train, predict, score, diagnose].py
├── experiments/                      # Walk-forward backtest seeders (not run in CI)
│   ├── save_backtest_results.py      # PM2.5 backtest
│   └── save_elec_backtest_results.py # Electricity backtest
└── tests/                            # Unit & integration suite (205 tests) + dashboard_nowcasts.test.cjs
```

---

## 🔌 API Reference

All routes are read-only (`GET`) and served under `/`:

| Endpoint | Parameters | Description |
|---|---|---|
| `/` | — | Service index + route map |
| `/health` | — | PM2.5 data freshness, staleness days, and source status |
| `/forecast` | `model=lightgbm`, `source=latest\|daily\|nowcast\|backtest` (default `latest`) | Newest live prediction (daily preferred on tied dates, never backtest), with issuance and scoring status |
| `/history` | `days=30` | Recent raw observations (oldest-first for charting) |
| `/leaderboard` | `source=daily` | Most recent scored day per model, isolated by provenance |
| `/evaluation` | `days=30` | Separate `verified`, `nowcast`, and `backtest` metrics (sorted on verified MAE only) |
| `/predictions` | `model`, `limit`, `source`, `scored_only` | Queryable historical prediction log with absolute and percentage errors |
| `/diagnostics` | — | Current PM2.5 bundle status, training window, horizon |
| `/electricity/health` | — | Electricity data freshness and mirror status |
| `/electricity/forecast` | `model=lightgbm`, `source=latest\|daily\|nowcast\|backtest` (default `latest`) | Newest live demand prediction (daily preferred on tied dates, never backtest), with timing |
| `/electricity/history` | `days=30` | Historical peak demand, energy met, and temperature observations |
| `/electricity/leaderboard` | `source=daily` | Latest scored performance for demand models, isolated by provenance |
| `/electricity/evaluation` | `days=30` | Complete metrics including **MAPE**, MAE, and RMSE by provenance |
| `/electricity/predictions` | `model`, `limit`, `source`, `scored_only` | Historical demand predictions log with error metrics |
| `/electricity/diagnostics` | — | Current electricity bundle status, training window, horizon |
| `/dashboard` | — | Serves the HTML/JS dashboard directly with CSP headers |

*Interactive Swagger documentation is available at `/docs`.*

### Timing and model metadata

- `/forecast` and `/electricity/forecast` default to `source=latest` (newest daily/nowcast row, daily preferred on tied dates; never backtests). Use `source=daily` for advance-only reads — an advance-only 404 is normal when only nowcasts exist, which is the structural steady state for whole-day features(t).
- Both `/leaderboard` routes accept `source=daily|nowcast|backtest`, defaulting to `daily`. Both `/evaluation` routes return separate `verified`, `nowcast`, and `backtest` blocks; no combined score exists.
- Prediction payloads expose `publication_source`, `timing_status`, `issued_at`, `cutoff`, `target_timezone`, `feature_as_of`, `horizon_days`, and `provenance_consistent`. `status=pending|verified` describes actual availability, not advance issuance. Missing historical issuance remains `unknown`.
- `/diagnostics` and `/electricity/diagnostics` validate the authoritative `.txt.bundle.json` when present and expose its version, training window, feature count, and horizon. A corrupt bundle cannot fall back to a legacy `.txt`. Existing legacy artifacts remain readable but are explicitly labelled `legacy` with unavailable bundle metadata. Metadata describes the **current deployment**, not the model used for historical predictions.
- The dashboard keeps nowcast evaluation, leaderboard, and ledger separate from advance metrics and charts.

---

## 🚀 Quickstart & Development

### 1. Prerequisites
- Python 3.11+
- PostgreSQL database (or Neon serverless DSN)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/rahulnidamanuri15/Forecast-Lab.git
cd Forecast-Lab

# Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: .\myenv\Scripts\activate

# Install production and development dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
CITY=Nagpur
STATE=Maharashtra
FRONTEND_ORIGIN=http://localhost:8000
```

### 4. Running Tests & Validation
Point `DATABASE_URL` at a disposable local Postgres (PostgreSQL 17), not Neon. CI uses `127.0.0.1:5432/vericast`; the historical local default was `127.0.0.1:5433/vericast_test` — either works as long as it is throwaway. Use `PYTHON_DOTENV_DISABLED=1` to avoid loading local dotenv configuration during verification.

```bash
python -m vericast.schema
python -m vericast.schema  # verify idempotency
python -m pytest tests -q -ra --cov=vericast --cov=app --cov-fail-under=60
node --test tests/dashboard_nowcasts.test.cjs
python -m vericast.gate
python -m vericast.local_time

# Against your running API and populated target database:
python verify_deployment_readiness.py
```

The test suite includes migrations from the old source-blind keys, same-date provenance isolation, cutoff rollback, real model inference, and a nowcast-only local HTTP deployment. Integration fixtures use random schemas and clean up afterward. Node tests execute dashboard JavaScript with a minimal DOM adapter; they are not visual browser tests.

### Migration rollout
1. Verify on the local disposable database first, then the GitHub Actions CI PostgreSQL 17 service. CI runs schema migration twice before the tests; the tests also exercise upgrades with existing rows.
2. Before the first production migration, pause old daily/scoring/backtest writers and drain in-flight jobs. Take a database backup and record row counts by source. **Old writers' conflict targets are incompatible with the new keys.**
3. After CI passes and rollout is approved, run `python -m vericast.schema` against the intended production environment, then deploy the matching API/workers/dashboard together. The daily workflow's migration job must succeed before either ingestion job starts.
4. Check both diagnostics routes and separate evaluations/leaderboards, then resume schedules. No retraining or horizon change is required.

The migration changes keys to include source and corrects legacy `daily` predictions issued at/after local target midnight (or with unknown issuance), together with their matching single-day scores. It preserves values and timestamps. Conflicts abort the transaction instead of deleting records. Do not roll back to old writers against the new schema; use a reviewed database restore/rollback plan if needed.

### 5. Running the API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000/dashboard` in your browser.

### 6. Docker
```bash
docker build -t vericast-api .
docker run -p 8000:8000 --env-file .env vericast-api
```

---

## ⚡ Automation Workflows

- **Daily Pipeline** (`.github/workflows/daily-pipeline.yml`): Runs daily at off-peak minutes (`10:47 IST` and `14:12 IST`). Runs `python -m vericast.schema` first; dependent PM2.5 and electricity jobs then ingest, compute features, test leakage, score prior predictions, and publish t+1 predictions classified as advance forecasts or delayed estimates.
- **Weekly Retraining** (`.github/workflows/weekly-retrain.yml`): Runs every Sunday to retrain LightGBM models on accumulating observations, guarded by holdout quality gates (`vericast.gate`).
- **Continuous Gate** (`.github/workflows/readiness-gate.yml`): Runs the 23-point system verification against live production environments.

---

## 🔒 Security & License

- **Security**: Please report security vulnerabilities privately according to [SECURITY.md](.github/SECURITY.md).
- **License**: Released under the [MIT License](LICENSE).
