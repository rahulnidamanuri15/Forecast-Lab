# VeriCast — Published-Then-Verified Next-Day Forecasting

[![CI](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/ci.yml)
[![Daily Pipeline](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/daily-pipeline.yml)
[![Readiness Gate](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/readiness-gate.yml)
[![Weekly Retrain](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml/badge.svg)](https://github.com/rahulnidamanuri15/Forecast-Lab/actions/workflows/weekly-retrain.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Next-day production machine learning forecasts **published first and verified later**. Every forecast is committed to the database before its ground truth is knowable, then scored against arriving observations. Nothing is retro-fitted or silently re-computed.

---

## 🎯 Target Domains

| Target | Series | Unit | Models Evaluated | Upstream Source |
|---|---|---|---|---|
| **Air Quality** | Nagpur PM2.5 | μg/m³ | `lightgbm`, `naive_baseline` | Open-Meteo CAMS Reanalysis |
| **Electricity** | Maharashtra Peak Demand | MW | `lightgbm`, `naive_baseline`, `seasonal_naive` | Grid-India PSP Reports |

Both pipelines run fully independently with isolated tables, daily automation jobs, and API routes. A stall in one domain cannot block or affect the other.

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
(source: 'daily' vs 'backtest')
       │
       ▼
[FastAPI Backend] ──(Connection Pool + Rate Limiting + CSP)
       │
       ▼
[Interactive Dashboard] (Responsive Chart.js UI, Dark Mode, Provenance Filter)
```

---

## 📊 Model Performance Benchmarks

VeriCast enforces a strict distinction between **Verified Live** (predictions written before ground truth existed) and **Backtest** (walk-forward seeding with known ground truth). The two are **never combined or averaged**.

### 1. Nagpur PM2.5 (Air Quality)
*Backtest: 700 days | Verified: Live continuous tracking*

| Model | Provenance | Scored Days | MAE (μg/m³) | RMSE (μg/m³) | Performance Note |
|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 700 | **9.58** | **12.51** | **+13.7% better than baseline** |
| `naive_baseline` | `backtest` | 700 | 11.10 | 14.49 | Persistence baseline ($y_t \to y_{t+1}$) |
| **LightGBM** | `verified` | Live | **4.15** | **4.75** | Published prior to observation |
| `naive_baseline` | `verified` | Live | 4.39 | 5.28 | Published prior to observation |

### 2. Maharashtra Peak Electricity Demand
*Backtest: 1,238 days | Verified: Live continuous tracking*

| Model | Provenance | Scored Days | MAE (MW) | RMSE (MW) | MAPE | Performance Note |
|---|---|---|---|---|---|---|
| **LightGBM** | `backtest` | 1,238 | **773.54** | **1036.30** | **3.00%** | **+21.2% better than baseline** |
| `naive_baseline` | `backtest` | 1,238 | 981.15 | 1309.45 | 3.79% | Persistence ($y_t \to y_{t+1}$) |
| `seasonal_naive` | `backtest` | 1,238 | 1154.08 | 1590.27 | 4.48% | Same weekday last week ($y_{t-6}$) |
| **LightGBM** | `verified` | Live | ~1106 | ~1319 | 3.84% | Published prior to observation |

> Call `GET /evaluation` and `GET /electricity/evaluation` for live, real-time metrics.

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
  - In-memory rate limiter (120 req/min per IP) and 5-minute HTTP response caching.
  - Security headers enforced (CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`).

---

## 📁 Repository Structure

```
├── app.py                            # FastAPI application serving both targets
├── index.html                        # Chart.js frontend dashboard
├── verify_deployment_readiness.py    # Pre-flight deployment readiness gate (23 checks)
├── models/                           # Serialized LightGBM models and window metadata
│   ├── lightgbm_model.txt            # Nagpur PM2.5 model
│   └── lightgbm_elec_model.txt       # Maharashtra electricity model
├── vericast/                         # Shared core package
│   ├── local_time.py                 # Timezone authority (Asia/Kolkata)
│   ├── gate.py                       # Retrain evaluation and artifact release gate
│   ├── schema.py                     # Idempotent database DDL & migrations
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
└── tests/                            # Comprehensive unit & integration test suite (169 tests)
```

---

## 🔌 API Reference

All routes are read-only (`GET`) and served under `/`:

| Endpoint | Parameters | Description |
|---|---|---|
| `/health` | — | PM2.5 data freshness, staleness days, and source status |
| `/forecast` | `model=lightgbm` | Latest verified forecast (`source='daily'`) with pending/verified state |
| `/history` | `days=30` | Recent raw observations (oldest-first for charting) |
| `/leaderboard` | — | Most recent scored day per model from `model_performance` |
| `/evaluation` | `days=30` | Aggregate metrics broken down into `verified` and `backtest` blocks |
| `/predictions` | `model`, `limit`, `source`, `scored_only` | Queryable historical prediction log with absolute and percentage errors |
| `/electricity/health` | — | Electricity data freshness and mirror status |
| `/electricity/forecast` | `model=lightgbm` | Latest peak demand forecast in MW |
| `/electricity/history` | `days=30` | Historical peak demand, energy met, and temperature observations |
| `/electricity/leaderboard` | — | Latest scored performance for demand models |
| `/electricity/evaluation` | `days=30` | Complete metrics including **MAPE**, MAE, and RMSE by provenance |
| `/electricity/predictions` | `model`, `limit`, `source`, `scored_only` | Historical demand predictions log with error metrics |
| `/dashboard` | — | Serves the HTML/JS dashboard directly with CSP headers |

*Interactive Swagger documentation is available at `/docs`.*

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
```bash
# Run unit and integration tests (169 test cases)
python -m pytest

# Execute deployment readiness verification (23 checks)
python verify_deployment_readiness.py
```

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

- **Daily Pipeline** (`.github/workflows/daily-pipeline.yml`): Runs daily at off-peak minutes (`10:47 IST` and `14:12 IST`) with parallel jobs for PM2.5 and Electricity. Ingests data, validates freshness, computes features, tests for leakage, scores previous predictions, and publishes next-day forecasts.
- **Weekly Retraining** (`.github/workflows/weekly-retrain.yml`): Runs every Sunday to retrain LightGBM models on accumulating observations, guarded by holdout quality gates (`vericast.gate`).
- **Continuous Gate** (`.github/workflows/readiness-gate.yml`): Runs the 23-point system verification against live production environments.

---

## 🔒 Security & License

- **Security**: Please report security vulnerabilities privately according to [SECURITY.md](.github/SECURITY.md).
- **License**: Released under the [MIT License](LICENSE).
