# VeriCast — PM2.5 forecasting for Nagpur

Next-day PM2.5 concentration forecasts that are **published first and verified later**.
Every forecast is written to the database before its actual is knowable, then scored
against the observation when it arrives. Nothing is retro-fitted.

## Architecture

```
Open-Meteo  →  observations  →  engineer_features.py  →  features
                                                            ↓
                                    LightGBM / naive baseline
                                                            ↓
                        predictions  →  FastAPI  →  dashboard (Chart.js)
                             ↑
                    score_predictions.py (joins observations on forecast_date)
```

All application-level date decisions ("today", "yesterday", forecast date, scoring
date) go through `local_time.py`, which is fixed to **Asia/Kolkata**. PostgreSQL
timestamps stay timezone-aware.

## What the ground truth actually is

PM2.5 observations come from Open-Meteo's air-quality endpoint, which serves
**CAMS reanalysis output — a model, not a ground station**. So the target this
system is scored against is itself a model estimate, not a physical measurement.

That is a deliberate tradeoff, not an oversight. CAMS is gap-free and
consistently defined over the full 2023→present window, which is what makes a
clean publish-then-verify record possible at all; CPCB station feeds have gaps
and station-level discontinuities that would contaminate the scoring loop.
The claim this repo makes is "the forecasting and verification loop is honest",
not "these numbers are measured air".

Swapping to real sensor readings (OpenAQ → CPCB) touches exactly one function,
`fetch_and_aggregate_data` in `ingest_observations.py` — nothing else in the
pipeline reads the AQ API. Deferred on purpose: the loop matters more than the
sensor.

## Model performance

Live full-record numbers, served by `GET /evaluation` with no `days` parameter
(2023-09-02 → present). Regenerate the table below by calling that endpoint —
it is the same query, not a hand-maintained copy:

| Model | Scored | MAE (μg/m³) | RMSE (μg/m³) | Description |
|-------|--------|------|------|-------------|
| lightgbm | 709 | **9.51** | 12.44 | LightGBM on lagged + rolling + weather features |
| naive_baseline | 710 | 11.01 | 14.40 | Predict tomorrow's PM2.5 as today's PM2.5 |

LightGBM beats the naive baseline by **~13.7% MAE**. That margin is the whole point:
persistence is a genuinely hard baseline for daily air quality, and a model that
can't beat it isn't worth deploying.

`GET /leaderboard` is a different question: it reports each model's *most recent
scored day* from `model_performance`, so its `sample_size` is normally 1. Use
`/evaluation` for accuracy claims and `/leaderboard` for "how did yesterday go".

Reproduce the frozen walk-forward benchmark with `python experiments/compare_models.py`
(n=700, MAE 9.5798 vs 11.0962 — consistent with the live record above).

## Files

Production path (root):

- `ingest_observations.py` — pull new Open-Meteo data into `observations`
- `engineer_features.py` — build lag/rolling/calendar features into `features`
- `leakage_test.py` — assert no feature row sees data past its own `as_of`
- `train_production_model.py` — retrain and write `lightgbm_model.txt`
- `make_prediction.py` — write tomorrow's forecast for both models
- `score_predictions.py` — fill in `actual_pm2_5` for every pending row, upsert `model_performance` (the only writer of actuals)
- `diagnose_lightgbm_forecast.py` — refuse to publish an unfit forecast (non-zero exit)
- `app.py` — FastAPI service
- `index.html` — dashboard
- `local_time.py` — the only source of "today"
- `verify_deployment_readiness.py` — the single go-live gate

Not on the production path: `experiments/` (`compare_models.py`,
`naive_baseline_backtest.py`, `train_lightgbm.py`, `train_sarima.py`,
`save_backtest_results.py`) and `tests/`.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Latest observation date + staleness in days |
| `GET /forecast?model=lightgbm` | Latest stored forecast, with `status: pending\|verified` |
| `GET /history?days=30` | Recent observations |
| `GET /leaderboard` | Most recent scored day per model (`sample_size` is normally 1) |
| `GET /evaluation` | Full-record MAE/RMSE over every prediction ever published |
| `GET /evaluation?days=30` | Same, restricted to a rolling window |
| `GET /predictions?model=lightgbm&limit=12` | Prediction log with errors |

Interactive docs at `/docs`. All values are raw PM2.5 concentration in μg/m³ —
**not** AQI. No AQI transform is computed anywhere in this system.

## Automation

- `.github/workflows/daily-pipeline.yml` — one sequential job, fail-fast:
  ingest → engineer features → leakage test → score pending → make forecast → verify.
  Scoring runs *before* forecasting because yesterday's actual has to exist first.
- `.github/workflows/weekly-retrain.yml` — Sundays: retrain and commit
  `lightgbm_model.txt` back to the repo, which the daily pipeline picks up on its
  next checkout.

`score_predictions.py` scores *every* pending row, not just yesterday's, so a
missed run self-heals on the next one instead of leaving a permanent NULL.

## Running it

```bash
cp .env.example .env          # then fill in DATABASE_URL
pip install -r requirements.txt
python -m pytest tests -q
uvicorn app:app --host 0.0.0.0 --port 8000
python verify_deployment_readiness.py   # must print ALL CHECKS PASSED
```

Docker:

```bash
docker build -t vericast-api .
docker run -p 8000:8000 -e DATABASE_URL=... -e CITY=Nagpur -e FRONTEND_ORIGIN=... vericast-api
```

Environment variables: `DATABASE_URL` (required), `CITY` (default `Nagpur`),
`FRONTEND_ORIGIN` (comma-separated allowed origins; empty means no browser origin
is allowed — there is no `*` fallback). `API_BASE` optionally points
`verify_deployment_readiness.py` at a deployed instance instead of localhost.

## Security

Report vulnerabilities privately — see [`.github/SECURITY.md`](.github/SECURITY.md).
Do not open a public issue.

## License

[MIT](LICENSE).

