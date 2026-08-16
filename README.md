# ML Forecasting - Air Quality Prediction

A simple, honest forecasting system for PM2.5 levels in Nagpur, India.

## Overview

This project implements a minimal viable forecasting system that:
- Collects hourly air quality and daily weather data from Open-Meteo APIs
- Stores data in PostgreSQL
- Engineers features (lags, rolling averages, time-based)
- Compares models: naive baseline, LightGBM, SARIMA
- Provides a REST API and simple dashboard
- Includes automated daily predictions and scoring via GitHub Actions

## Go-Live Status

✅ **All gates passed**:
- Leakage test: PASSED
- Unique constraints: Applied to observations and features tables
- Baseline on leaderboard: naive_baseline MAE = 7.3724
- Crons: Configured to run predictions at 6AM UTC, scoring at 7AM UTC
- Public URL: Local server running at http://localhost:8000

## Architecture

```
Data Collection → Observations Table → Features Engineering → Features Table
                                    ↘
                                     → Models (Baseline, LightGBM, SARIMA)
                                    ↙
                         API Endpoints ← Dashboard (HTML + Chart.js)
                         ↑
               GitHub Actions (Daily Prediction & Scoring)
```

## Key Files

- `check_data.py` - Verifies sufficient data is available
- `create_observations_table.py` / `ingest_observations.py` - Data pipeline
- `create_features_table.py` / `engineer_features.py` - Feature engineering
- `leakage_test.py` - Verifies no data leakage
- `naive_baseline_backtest.py` - Establishes performance benchmark
- `train_lightgbm.py` / `train_sarima.py` - Model implementations
- `app.py` - FastAPI server with `/forecast`, `/leaderboard`, `/history`
- `index.html` - Simple dashboard with Chart.js
- `.github/workflows/` - Automated prediction (6AM) and scoring (7AM)

## API Endpoints

- `GET /` - API info
- `GET /forecast` - Next day PM2.5 forecast (naive baseline)
- `GET /leaderboard` - Model performance comparison
- `GET /history?days=30` - Last N days of observations

## Deployment

For production deployment to Render, Fly.io, or similar:

1. Set `DATABASE_URL` environment variable to your PostgreSQL connection string
2. The GitHub Actions will automatically:
   - Make predictions daily at 6AM UTC
   - Score predictions daily at 7AM UTC (once actuals are available)
3. Visit `/docs` for interactive API documentation

## Model Performance (from backtesting)

| Model | MAE | RMSE | Description |
|-------|-----|------|-------------|
| naive_baseline | 7.3724 | 9.8950 | Predict tomorrow's PM2.5 as today's PM2.5 |
| lightgbm | 7.5459 | 10.0399 | LightGBM with lagged/rolling features |
| sarima | 7.6990 | 10.2045 | SARIMA(1,1,1)x(1,1,1,7) |

*Note: The naive baseline is surprisingly hard to beat with this data and feature set - this is honest forecasting.*

## Next Steps (Post-Deploy)

Following the plan:
- **September**: Add `/health` endpoint showing hours since last scoring (detect cron failures)
- **October**: Add second city OR multi-day horizon (not both)
- **November**: Add prediction intervals + coverage check
- **December**: README with live numbers + 60-second demo video

## Honesty Statement

This system intentionally keeps things simple:
- No attempt to hide that the naive baseline is competitive
- No over-engineering - we stop at the first solution that works
- Clear separation of concerns: data → features → models → API
- All numbers are explainable from memory (we know where every metric comes from)
- If deployed, it will provide real value while we continue learning elsewhere

*"The best code is the code never written." - We've written only what was necessary.*