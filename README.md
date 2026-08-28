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
Open-Meteo  →  observations  →  vericast/pm25/features.py  →  features
                                                            ↓
                                    LightGBM / naive baseline
                                                            ↓
                        predictions  →  FastAPI  →  dashboard (Chart.js)
                             ↑
                    vericast/pm25/score.py (joins observations on forecast_date)
```

The electricity path mirrors it file for file under `vericast/elec/`, with
`electricity_`-prefixed tables. Both feature stores are built the same way — **one
idempotent `INSERT ... SELECT`** using Postgres date-addressed window frames:

```sql
MAX(peak_demand_mw) OVER (ORDER BY as_of
    RANGE BETWEEN INTERVAL '1 day' PRECEDING AND INTERVAL '1 day' PRECEDING)
```

`RANGE ... PRECEDING` addressed by date structurally cannot reference a future row,
and returns NULL across a date gap instead of silently reaching over it — so the
leakage guarantee is a property of the query rather than a test that has to pass.
`CASE WHEN COUNT(*) OVER w7 = 7` enforces minimum periods the same way, which is
stricter than a row-index check because it also nulls out gap-shortened windows.
CI asserts it anyway, against a real Postgres: `tests/test_feature_alignment.py`
reproduces the 2025-05-21 → 05-24 Maharashtra gap and checks that `demand_lag_1`
nulls across it.

All application-level date decisions ("today", "yesterday", forecast date, scoring
date) go through `vericast/local_time.py`, which is fixed to **Asia/Kolkata**. PostgreSQL
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
`fetch_and_aggregate_data` in `vericast/pm25/ingest.py` — nothing else in the
pipeline reads the AQ API. Deferred on purpose: the loop matters more than the
sensor.

One more thing to state plainly, because it defines the target: **a daily PM2.5
mean is a UTC day, labelled with an IST-derived date.** The ingest call passes
`"timezone": "UTC"` and buckets hourly values by their UTC date, while the date
*range* requested comes from `local_time` (Asia/Kolkata). So `as_of = 2026-08-27`
means 2026-08-27 00:00–23:00 UTC — 05:30 that day to 04:30 the next, in IST.

That offset is consistent on both sides of the loop: features, training and
scoring all read the same column, so no model gets an advantage from it. It is
left as-is rather than corrected because switching to `Asia/Kolkata` would
redefine every historical actual on the next full re-ingest — moving numbers
already published and scored against, which is the retro-fitting this project
exists to avoid. An IST-day series would be a new city key, not a rewrite of
this one.

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
- **It is treated as untrusted input.** Blank *or unparseable* demand values are skipped
  rather than coerced — one bad cell loses its day, not the rows already accepted — and
  `vericast/elec/diagnose.py` refuses to publish outside 15,000–40,000 MW.

If the mirror stops updating, the electricity job fails its own freshness gate and
publishes nothing. It cannot affect the PM2.5 record.

Temperature for both targets comes from Open-Meteo archive, averaged unweighted
across Mumbai, Pune and Nagpur for the state-level demand model.

## Model performance

The tables below are a **dated snapshot, taken 2026-08-28** — the counts and MAEs are
frozen numbers checked into this file, not generated output. For current numbers call
`GET /evaluation` and `GET /electricity/evaluation` with no `days` parameter; the record
only grows, so the live figures move as days are scored.

**PM2.5, Nagpur** (2023-09-02 → present):

| Model | Scored | MAE (μg/m³) | RMSE (μg/m³) | Description |
|-------|--------|------|------|-------------|
| lightgbm | 711 | **9.49** | 12.42 | LightGBM on lagged + rolling + weather features |
| naive_baseline | 712 | 10.99 | 14.38 | Predict tomorrow's PM2.5 as today's PM2.5 |

LightGBM beats the naive baseline by **~13.7% MAE**. That margin is the whole point:
persistence is a genuinely hard baseline for daily air quality, and a model that
can't beat it isn't worth deploying.

**Peak demand, Maharashtra** (2023-03-02 → present, seeded by a walk-forward backtest):

| Model | Scored | MAE (MW) | RMSE (MW) | MAPE | Description |
|-------|--------|----------|-----------|------|-------------|
| lightgbm | 1239 | **773.90** | 1036.46 | **3.00%** | 14 features: lagged demand, rolling aggregates, thermal, calendar |
| naive_baseline | 1239 | 981.46 | 1309.49 | 3.79% | Tomorrow's peak = today's peak |
| seasonal_naive | 1239 | 1154.95 | 1590.89 | 4.49% | Tomorrow's peak = the same weekday last week |

LightGBM beats persistence by **21.2% MAE** here — a wider margin than PM2.5's.

Two results worth reading carefully:

- **MAPE is the metric that travels.** A 774 MW error on a ~26 GW system is 3%; the
  same absolute number would be meaningless next to a PM2.5 figure. Cross-target
  comparisons should use MAPE, never MAE.
- **`seasonal_naive` came last, not first.** The design expectation was that a power
  grid's same-weekday-last-week value would beat plain persistence, because Sunday
  looks more like last Sunday than like Saturday. Measured over 1,239 days it is the
  worst of the three — a 6-day-old value carries too much drift for the weekly cycle
  to pay for. The weekly cycle is real (`day_of_week` ranks 3rd in feature importance,
  behind `demand_roll_7_mean` and `demand_lag_1`); LightGBM just extracts it better
  than a bare weekly lag does. It stays published as a baseline because a baseline
  that loses is still evidence, and removing it after seeing the result would be
  exactly the retro-fitting this project exists to avoid.

`GET /leaderboard` is a different question: it reports each model's *most recent
scored day* from `model_performance`, so its `sample_size` is normally 1. Use
`/evaluation` for accuracy claims and `/leaderboard` for "how did yesterday go".
`/electricity/leaderboard` is the same question on the demand side, reading
`electricity_model_performance` and carrying `mape` through.

Reproduce the frozen walk-forward benchmark with `python experiments/compare_models.py`
(n=1062, MAE 9.3567 vs 10.7975 — consistent with the live record above). Its sample
count is smaller than the 1,092-row dataset because the first 30 days seed the
walk-forward window rather than being scored. It is the only backtest script kept:
the single-model ones it superseded (`naive_baseline_backtest.py`, `train_lightgbm.py`,
`train_sarima.py`) each re-derived the same dataset with their own hardcoded city and
their own baseline number to beat, so their printed comparisons drifted out of
agreement with this table. `compare_models.py` scores both models on identical
prediction dates in one pass, which is the only way the improvement percentage means
anything.

## Layout

```
app.py                            FastAPI service, both targets
index.html                        dashboard, one tab per target
verify_deployment_readiness.py    the single go-live gate (16 checks)
models/                           committed LightGBM artifacts
vericast/
├── __init__.py                   resolves models/ paths from the package, not cwd
├── local_time.py                 the only source of "today" (Asia/Kolkata)
├── gate.py                       retrain gate: refuse to ship a broken model
├── schema.py                     idempotent DDL for all eight tables
├── pm25/                         Nagpur PM2.5 (μg/m³)
│   ├── ingest.py                 Open-Meteo → observations
│   ├── features.py               one INSERT ... SELECT; lag/rolling/calendar → features
│   ├── leakage_test.py           assert no feature row sees data past its own as_of
│   ├── train.py                  retrain → models/lightgbm_model.txt; owns FEATURE_COLUMNS
│   ├── predict.py                write tomorrow's forecast for both models
│   ├── score.py                  fill actual_pm2_5 for every pending row (only daily-path writer of actuals)
│   └── diagnose.py               refuse to publish an unfit forecast (non-zero exit)
└── elec/                         Maharashtra peak demand (MW), same seven roles
    ├── ingest.py                 demand mirror + Open-Meteo temperature
    ├── features.py               one INSERT ... SELECT; no leakage test needed (see Architecture)
    ├── train.py                  retrain → models/lightgbm_elec_model.txt; owns FEATURE_COLUMNS
    ├── predict.py                three models, per-model publish guards
    ├── score.py                  fill actual_demand_mw, upsert MAE/RMSE/MAPE
    └── diagnose.py               6-check publish gate (non-zero exit)
```

Same seven filenames in both target packages, so `pm25/x.py` and `elec/x.py` always
do the same job — that symmetry is what makes the two pipelines readable side by side.
Each `train.py` is the single definition of its target's `FEATURE_COLUMNS`; `predict.py`
and the backtests import it, so the lists cannot drift (asserted in
`tests/test_layout.py`).

Every script runs as a module from the repo root:

```bash
python -m vericast.pm25.ingest      # …features, .leakage_test, .train, .predict, .score, .diagnose
python -m vericast.elec.ingest      # …same names under elec
python -m vericast.schema           # create any missing tables
```

Not on the production path: `experiments/` (`compare_models.py`,
`save_backtest_results.py`, `save_elec_backtest_results.py`) and `tests/`.

Two of those experiment scripts are the **one documented exception** to
"`score.py` is the only writer of actuals": `save_backtest_results.py` and
`save_elec_backtest_results.py` INSERT `actual_pm2_5` / `actual_demand_mw`
directly, because a walk-forward backtest already knows both sides of every pair.
They are how the launch record was seeded (see Setup below) and are run once, by
hand, never from the daily job. Everything after launch is `score.py`'s.

Those seeded rows are not otherwise marked: in `model_performance` /
`electricity_model_performance`, a backtest row is distinguishable from a daily
row only by `sample_size` — hundreds vs. the normal 1. No provenance column was
added, because adding one to a live table the published record reads from is
risk without a caller that needs it.

`model_performance` also has no `city` column (its electricity counterpart has
`state`), because it predates the second target. Consequence to know before
changing `CITY`: the PM2.5 leaderboard would silently mix cities, so a second
city needs a migration, not just a new env var.

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
| `GET /electricity/leaderboard` | Most recent scored day per model (`sample_size` is normally 1) |
| `GET /electricity/evaluation` | Full-record MAE/RMSE/**MAPE** per model |
| `GET /electricity/predictions?model=lightgbm&limit=15` | Prediction log with `error` and `error_pct` |

Interactive docs at `/docs`. PM2.5 values are raw concentration in μg/m³ —
**not** AQI; no AQI transform is computed anywhere in this system. Electricity values
are peak demand met in MW and energy met in MU, as published upstream.

The two `model=` allowlists are deliberately separate: `seasonal_naive` is valid on
`/electricity/forecast` and a 400 on `/forecast`, because no such PM2.5 model is
published.

Connections come from a `psycopg_pool.ConnectionPool` (min 1, max 8) opened in the
FastAPI lifespan, behind the same `get_db_connection()` every handler already
called — Neon is a network hop away and a fresh connect + TLS handshake + auth per
request was the largest slice of this API's latency. `check=check_connection` is
not optional against Neon, which closes idle connections server-side; without it
the pool eventually hands out a dead socket. Max 8 because this is a read-only GET
API and Neon's free tier caps concurrent connections, so a bigger pool only holds
server slots idle.

## Automation

- `.github/workflows/daily-pipeline.yml` — **two independent jobs, no `needs:`**:
  - `pipeline` (PM2.5): ingest → engineer features → leakage test → score pending → make forecast → verify.
  - `electricity`: ingest → engineer features → score pending → make forecast → verify.

  They run in parallel and neither gates the other. That is the point: the electricity
  source is a third-party mirror that can stall, and a stall there must not block a
  PM2.5 forecast whose own upstream is working. Within each job the steps are
  sequential and fail-fast — scoring runs *before* forecasting because yesterday's
  actual has to exist first.

  **Two crons, not one:** `17 5 * * *` (10:47 IST) and `42 8 * * *` (14:12 IST).
  GitHub's `schedule:` trigger is best-effort on a free public repo — queued at low
  priority, routinely delayed, and droppable outright. Measured on this repo: 20–32
  minutes of drift for a week, then 11 hours, then a run that never fired. Both minutes
  are deliberately off the top of the hour, which is the most contended slot. The second
  cron costs nothing when the first landed: every write is an `ON CONFLICT … DO UPDATE`
  upsert, ingest resumes from `MAX(as_of)`, and a same-day re-run finds nothing new.
- `.github/workflows/weekly-retrain.yml` — Sundays: retrain both models and commit
  `models/lightgbm_model.txt` and `models/lightgbm_elec_model.txt` back to the repo,
  which the daily pipeline picks up on its next checkout. Neither artifact is
  overwritten unconditionally: `vericast/gate.py` holds out the last 30 days, fits a
  challenger on the head only, and refuses to write unless it beats persistence,
  varies as much as the actuals do, and correlates with them. A refused retrain exits
  0 and leaves the artifact untouched, so the workflow reports "nothing to commit".

  The gate deliberately does *not* vote on "challenger beats the artifact on disk":
  that artifact was trained on the held-out rows, which on live data makes it look
  1.5x (PM2.5) to 2.0x (electricity) better than an honest challenger and would freeze
  the incumbent forever. Its score is printed for drift, not compared.

Both scoring scripts score *every* pending row, not just yesterday's, so a missed run
self-heals on the next one instead of leaving a permanent NULL.

## Running it

```bash
# DATABASE_URL is required; CITY=Nagpur, STATE=Maharashtra, FRONTEND_ORIGIN
# and API_BASE=http://localhost:8000 all default, so a .env with just the DSN works.
echo 'DATABASE_URL=' > .env    # then fill it in
pip install -r requirements.txt -r requirements-dev.txt   # -dev is pytest only
python -m pytest tests -q
uvicorn app:app --host 0.0.0.0 --port 8000
python verify_deployment_readiness.py   # must print ALL CHECKS PASSED
```

First-time electricity setup (one-off, then the daily job takes over):

```bash
python -m vericast.schema
python -m vericast.elec.ingest                        # ~1,300 rows, 2023-01-01 →
python -m vericast.elec.features
python -m vericast.elec.train
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

