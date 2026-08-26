# VeriCast — published-then-verified next-day forecasting

Next-day forecasts that are **published first and verified later**. Every forecast is
written to the database before its actual is knowable, then scored against the
observation when it arrives. Nothing is retro-fitted.

Two targets, one loop:

| Target | Series | Unit | Models |
|--------|--------|------|--------|
| Air quality | Nagpur PM2.5 | μg/m³ | lightgbm, naive_baseline |
| Electricity | Maharashtra peak demand met | MW | lightgbm, naive_baseline, seasonal_naive |

They share the discipline and nothing else: separate tables, separate daily jobs,
separate routes. A stall in one cannot block the other.

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

The electricity path mirrors it with `electricity_`-prefixed tables and `_elec_`
scripts, except that features are built by **one idempotent `INSERT ... SELECT`**
using Postgres date-addressed window frames rather than a Python row loop:

```sql
MAX(peak_demand_mw) OVER (ORDER BY as_of
    RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING)
```

`RANGE ... PRECEDING` addressed by date structurally cannot reference a future row,
and returns NULL across a date gap instead of silently reaching over it — so the
leakage guarantee is a property of the query rather than a test that has to pass.
`CASE WHEN COUNT(*) OVER w7 = 7` enforces minimum periods the same way, which is
stricter than a row-index check because it also nulls out gap-shortened windows.

All application-level date decisions ("today", "yesterday", forecast date, scoring
date) go through `local_time.py`, which is fixed to **Asia/Kolkata**. PostgreSQL
timestamps stay timezone-aware.

## What the ground truth actually is

### PM2.5

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

### Electricity demand

Peak demand met comes from a **community GitHub mirror of Grid-India's daily state
reports** ([`HalcyonVector/Grid-Sentinel`](https://github.com/HalcyonVector/Grid-Sentinel),
`Dataset/study3_states.csv`) — not from the operator directly. That matters enough
to state plainly rather than bury:

- **It is third-party.** Grid-India publishes daily PSP reports, but not at a stable
  machine-readable URL; every probe of `report.grid-india.in/.../{date}_NLDC_PSP.xlsx`
  returned nothing. The mirror is the only reliable programmatic access found.
- **It lags real time by 2–4 days.** So the electricity forecast is labelled "the day
  after the newest observation", not real-world tomorrow, and its freshness thresholds
  are 5 days rather than PM2.5's 1. A 2–4 day lag is the normal case here, not an
  incident — `GET /electricity/health` reports `source_lag_expected` for exactly this.
- **It is treated as untrusted input.** Blank demand values are skipped rather than
  coerced, and `diagnose_elec_forecast.py` refuses to publish outside 15,000–40,000 MW.

If the mirror stops updating, the electricity job fails its own freshness gate and
publishes nothing. It cannot affect the PM2.5 record.

Temperature for both targets comes from Open-Meteo archive, averaged unweighted
across Mumbai, Pune and Nagpur for the state-level demand model.

## Model performance

Live full-record numbers, served by `GET /evaluation` and `GET /electricity/evaluation`
with no `days` parameter. Regenerate the tables below by calling those endpoints —
same query, not a hand-maintained copy.

**PM2.5, Nagpur** (2023-09-02 → present):

| Model | Scored | MAE (μg/m³) | RMSE (μg/m³) | Description |
|-------|--------|------|------|-------------|
| lightgbm | 709 | **9.51** | 12.44 | LightGBM on lagged + rolling + weather features |
| naive_baseline | 710 | 11.01 | 14.40 | Predict tomorrow's PM2.5 as today's PM2.5 |

LightGBM beats the naive baseline by **~13.7% MAE**. That margin is the whole point:
persistence is a genuinely hard baseline for daily air quality, and a model that
can't beat it isn't worth deploying.

**Peak demand, Maharashtra** (2023-03-02 → present, seeded by a walk-forward backtest):

| Model | Scored | MAE (MW) | RMSE (MW) | MAPE | Description |
|-------|--------|----------|-----------|------|-------------|
| lightgbm | 1238 | **773.54** | 1036.30 | **3.00%** | 14 features: lagged demand, rolling aggregates, thermal, calendar |
| naive_baseline | 1238 | 981.15 | 1309.45 | 3.79% | Tomorrow's peak = today's peak |
| seasonal_naive | 1238 | 1154.08 | 1590.27 | 4.48% | Tomorrow's peak = the same weekday last week |

LightGBM beats persistence by **21.2% MAE** here — a wider margin than PM2.5's.

Two results worth reading carefully:

- **MAPE is the metric that travels.** A 774 MW error on a ~26 GW system is 3%; the
  same absolute number would be meaningless next to a PM2.5 figure. Cross-target
  comparisons should use MAPE, never MAE.
- **`seasonal_naive` came last, not first.** The design expectation was that a power
  grid's same-weekday-last-week value would beat plain persistence, because Sunday
  looks more like last Sunday than like Saturday. Measured over 1,238 days it is the
  worst of the three — a 6-day-old value carries too much drift for the weekly cycle
  to pay for. The weekly cycle is real (`day_of_week` ranks 3rd in feature importance,
  behind `demand_roll_7_mean` and `demand_lag_1`); LightGBM just extracts it better
  than a bare weekly lag does. It stays published as a baseline because a baseline
  that loses is still evidence, and removing it after seeing the result would be
  exactly the retro-fitting this project exists to avoid.

`GET /leaderboard` is a different question: it reports each model's *most recent
scored day* from `model_performance`, so its `sample_size` is normally 1. Use
`/evaluation` for accuracy claims and `/leaderboard` for "how did yesterday go".

Reproduce the frozen walk-forward benchmark with `python experiments/compare_models.py`
(n=700, MAE 9.5798 vs 11.0962 — consistent with the live record above).

## Files

Production path (root), PM2.5:

- `ingest_observations.py` — pull new Open-Meteo data into `observations`
- `engineer_features.py` — build lag/rolling/calendar features into `features`
- `leakage_test.py` — assert no feature row sees data past its own `as_of`
- `train_production_model.py` — retrain and write `lightgbm_model.txt`
- `make_prediction.py` — write tomorrow's forecast for both models
- `score_predictions.py` — fill in `actual_pm2_5` for every pending row, upsert `model_performance` (the only writer of actuals)
- `diagnose_lightgbm_forecast.py` — refuse to publish an unfit forecast (non-zero exit)

Production path, electricity (same order, `electricity_*` tables):

- `create_electricity_tables.py` — one-time DDL for all four tables
- `ingest_electricity.py` — demand mirror + Open-Meteo temperature → `electricity_observations`
- `engineer_elec_features.py` — one idempotent `INSERT ... SELECT`; no separate leakage test needed (see Architecture)
- `train_elec_model.py` — retrain and write `lightgbm_elec_model.txt`
- `make_elec_prediction.py` — write next-day forecasts for all three models
- `score_elec_predictions.py` — fill in `actual_demand_mw`, upsert `electricity_model_performance`
- `diagnose_elec_forecast.py` — 6-check publish gate (non-zero exit)

Shared:

- `app.py` — FastAPI service, both targets
- `index.html` — dashboard, one tab per target
- `local_time.py` — the only source of "today"
- `verify_deployment_readiness.py` — the single go-live gate

Not on the production path: `experiments/` (`compare_models.py`,
`naive_baseline_backtest.py`, `train_lightgbm.py`, `train_sarima.py`,
`save_backtest_results.py`, `save_elec_backtest_results.py`) and `tests/`.

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
| `GET /electricity/health` | Latest demand observation + `source_lag_expected` (5-day threshold) |
| `GET /electricity/forecast?model=lightgbm` | Latest stored demand forecast in MW |
| `GET /electricity/history?days=30` | Recent peak demand, energy met and temperature |
| `GET /electricity/evaluation` | Full-record MAE/RMSE/**MAPE** per model |
| `GET /electricity/predictions?model=lightgbm&limit=15` | Prediction log with `error` and `error_pct` |

Interactive docs at `/docs`. PM2.5 values are raw concentration in μg/m³ —
**not** AQI; no AQI transform is computed anywhere in this system. Electricity values
are peak demand met in MW and energy met in MU, as published upstream.

The two `model=` allowlists are deliberately separate: `seasonal_naive` is valid on
`/electricity/forecast` and a 400 on `/forecast`, because no such PM2.5 model is
published.

## Automation

- `.github/workflows/daily-pipeline.yml` — **two independent jobs, no `needs:`**:
  - `pipeline` (PM2.5): ingest → engineer features → leakage test → score pending → make forecast → verify.
  - `electricity`: ingest → engineer features → score pending → make forecast → verify.

  They run in parallel and neither gates the other. That is the point: the electricity
  source is a third-party mirror that can stall, and a stall there must not block a
  PM2.5 forecast whose own upstream is working. Within each job the steps are
  sequential and fail-fast — scoring runs *before* forecasting because yesterday's
  actual has to exist first.
- `.github/workflows/weekly-retrain.yml` — Sundays: retrain both models and commit
  `lightgbm_model.txt` and `lightgbm_elec_model.txt` back to the repo, which the daily
  pipeline picks up on its next checkout.

Both scoring scripts score *every* pending row, not just yesterday's, so a missed run
self-heals on the next one instead of leaving a permanent NULL.

## Running it

```bash
cp .env.example .env          # then fill in DATABASE_URL
pip install -r requirements.txt
python -m pytest tests -q
uvicorn app:app --host 0.0.0.0 --port 8000
python verify_deployment_readiness.py   # must print ALL CHECKS PASSED
```

First-time electricity setup (one-off, then the daily job takes over):

```bash
python create_electricity_tables.py
python ingest_electricity.py                          # ~1,300 rows, 2023-01-01 →
python engineer_elec_features.py
python train_elec_model.py
python experiments/save_elec_backtest_results.py      # seeds the launch record
```

Docker:

```bash
docker build -t vericast-api .
docker run -p 8000:8000 -e DATABASE_URL=... -e CITY=Nagpur -e FRONTEND_ORIGIN=... vericast-api
```

Environment variables: `DATABASE_URL` (required), `CITY` (default `Nagpur`),
`STATE` (default `Maharashtra`), `FRONTEND_ORIGIN` (comma-separated allowed origins;
empty means no browser origin is allowed — there is no `*` fallback). `API_BASE`
optionally points `verify_deployment_readiness.py` at a deployed instance instead of
localhost.

## Security

Report vulnerabilities privately — see [`.github/SECURITY.md`](.github/SECURITY.md).
Do not open a public issue.

## License

[MIT](LICENSE).

