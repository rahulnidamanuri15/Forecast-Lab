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
`CASE WHEN COUNT(peak_demand_mw) OVER w7 = 7` enforces minimum periods the same
way, which is stricter than a row-index check because it also nulls out
gap-shortened windows — and it counts the averaged column rather than `*`, so a
present row carrying a NULL value counts as absent too.
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

Both endpoints now return **two separate blocks per model**, `verified` and `backtest`,
and no combined figure at all. `verified` is the published-then-verified record: rows
written before the actual was knowable. `backtest` is the walk-forward launch record,
computed with the actual already in hand. They are never averaged, because averaging
them is exactly the retro-fitting this project exists to avoid — a 1,239-day backtest
would swamp a few dozen verified days and the headline number would silently become a
backtest average. The columns below say which block each figure came from.

**PM2.5, Nagpur** (2023-09-02 → present):

| Model | Scored | MAE (μg/m³) | RMSE (μg/m³) | Description |
|-------|--------|------|------|-------------|
| lightgbm | 711 | **9.49** | 12.42 | LightGBM on lagged + rolling + weather features |
| naive_baseline | 712 | 10.99 | 14.38 | Predict tomorrow's PM2.5 as today's PM2.5 |

LightGBM beats the naive baseline by **~13.7% MAE**. That margin is the whole point:
persistence is a genuinely hard baseline for daily air quality, and a model that
can't beat it isn't worth deploying.

Those counts are the **union of both blocks** as of the snapshot date — the launch
backtest seeded most of them and the daily job has been adding verified days since.
`GET /evaluation` splits them; this table does not, because the split moves every day
and a frozen number for it would be wrong within a week. Read the endpoint for the
verified-only figure.

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
`electricity_model_performance` and carrying `mape` through. Both feed the
dashboard's "Latest Scored Day" table, which carries that n≈1 caveat in its
subtitle so the single-day number cannot be read as the record.

Both endpoints filter `source = 'daily'`, and both `score.py` upserts force
`source = 'daily'` in their `DO UPDATE` branch rather than leaning on the column
DEFAULT — a DEFAULT applies to inserts only, so a daily score landing on a
`score_date` a backtest already wrote would otherwise stay labelled `'backtest'`
and be filtered out permanently.

The frozen walk-forward benchmark (n=1062, MAE 9.3567 vs 10.7975) is reproduced by
`python experiments/save_backtest_results.py`, which runs the walk-forward loop *and*
persists it. Those are the **`backtest` block alone**, which is why they differ from the
711/712 and 9.49 in the table above: that table is the union of both blocks at the
snapshot date, so it carries the verified days the daily job had added by then. Neither
figure is wrong; they answer different questions, and `/evaluation` is the one that
separates them. Its sample count is smaller than the 1,092-row dataset because the first
30 days seed the walk-forward window rather than being scored. A console-only twin of
that loop (`compare_models.py`) used to live beside it and was deleted: it re-derived the
same dataset with the same 30-day warmup and printed the same comparison without writing
anything, so the two could drift apart while both looked authoritative. The single-model
scripts they superseded (`naive_baseline_backtest.py`, `train_lightgbm.py`,
`train_sarima.py`) each hardcoded their own city and their own baseline to beat, which is
how the drift started. Scoring both models on identical prediction dates in one pass is
the only way the improvement percentage means anything.

## Layout

```
app.py                            FastAPI service, both targets
index.html                        dashboard, one tab per target
verify_deployment_readiness.py    the single go-live gate (22 checks)
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
└── elec/                         Maharashtra peak demand (MW), same eight roles
    ├── ingest.py                 demand mirror + Open-Meteo temperature
    ├── features.py               one INSERT ... SELECT; lag/rolling/calendar → electricity_features
    ├── leakage_test.py           assert no feature row sees data past its own as_of
    ├── train.py                  retrain → models/lightgbm_elec_model.txt; owns FEATURE_COLUMNS
    ├── predict.py                three models, per-model publish guards
    ├── score.py                  fill actual_demand_mw, upsert MAE/RMSE/MAPE
    └── diagnose.py               7-check publish gate (non-zero exit)
```

Same eight filenames in both target packages, so `pm25/x.py` and `elec/x.py` always
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

Not on the production path: `experiments/` (`save_backtest_results.py`,
`save_elec_backtest_results.py`) and `tests/`.

Both of those experiment scripts are the **one documented exception** to
"`score.py` is the only writer of actuals": `save_backtest_results.py` and
`save_elec_backtest_results.py` INSERT `actual_pm2_5` / `actual_demand_mw`
directly, because a walk-forward backtest already knows both sides of every pair.
They are how the launch record was seeded (see Setup below) and are run once, by
hand, never from the daily job. Everything after launch is `score.py`'s.

Those seeded rows carry `source = 'backtest'` in all four tables that have the column
(`predictions`, `electricity_predictions`, `model_performance`,
`electricity_model_performance`); every row the daily path writes is `'daily'`. Both
leaderboard handlers filter on `source = 'daily'`, and that filter is not
cosmetic: each backtest script upserts one aggregate row per model at its *last
evaluated date*, so re-running one today would carry the newest `score_date`,
win the `DISTINCT ON (model) ... ORDER BY score_date DESC`, and publish a
hundreds-deep backtest average as "how yesterday went". The column was added by
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `vericast/schema.py`, whose
`MIGRATIONS` tuple also backfills the pre-existing rows — `DEFAULT 'daily'`
would otherwise have labelled the launch backtest daily, leaving the hole open.

The label has to be forced in both directions, and `tests/test_leaderboard.py` asserts
both on the SQL text:

- **Every `DO UPDATE` branch on the daily path sets `source = 'daily'` explicitly.** A
  column DEFAULT applies to INSERT only, so a real forecast or score landing on a date
  the launch backtest already seeded would keep `source = 'backtest'` and never count
  towards the published record. Four writers need the line: both `score.py` upserts and
  both `predict.py` upserts.
- **Neither backtest seeder can overwrite a verified row.** Their `DO UPDATE` is guarded
  by `WHERE <table>.source = 'backtest'` and never assigns `actual_*`, so a re-run
  refreshes its own rows and skips every daily one instead of replacing a genuinely
  verified observation with a backtest-computed one.

`sample_size > 1` is the other tell on the performance tables: a daily row scores one
date against a `UNIQUE(city, forecast_date, model)` table, so its sample size is 1 by
construction.

`model_performance` also has no `city` column, which is the one migration a second
city would need — see Known limits below.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Latest observation date, staleness in days + `source_lag_expected` (2-day threshold) |
| `GET /forecast?model=lightgbm` | Latest **verified-provenance** forecast (`source = 'daily'`), with `status: pending\|verified` |
| `GET /history?days=30` | The 30 most recently stored observations, oldest first (`days` bounds **rows**, not the calendar) |
| `GET /leaderboard` | Most recent scored day per model (`sample_size` is normally 1) |
| `GET /evaluation` | Full record per model, split into `verified` and `backtest` blocks |
| `GET /evaluation?days=30` | Same, restricted to a rolling window (`days` is `ge=0`; a bad one is 422) |
| `GET /predictions?model=lightgbm&limit=12` | Prediction log with errors |
| `GET /electricity/health` | Latest demand observation + `source_lag_expected` (5-day threshold) |
| `GET /electricity/forecast?model=lightgbm` | Latest **verified-provenance** demand forecast in MW (`source = 'daily'`) |
| `GET /electricity/history?days=30` | Same row-bounded contract: recent peak demand, energy met and temperature |
| `GET /electricity/leaderboard` | Most recent scored day per model (`sample_size` is normally 1) |
| `GET /electricity/evaluation` | Same split, with **MAPE** alongside MAE/RMSE in each block |
| `GET /electricity/predictions?model=lightgbm&limit=15` | Prediction log with `error` and `error_pct` |

Every `days` parameter is validated by FastAPI rather than by a hand-rolled check, so
out-of-range values answer **422** across all four endpoints that take one — `/evaluation`
used to answer 400 for the same class of input, which made the contract something a
client had to special-case per route.

Both evaluation endpoints nest their metrics under a provenance key and publish **no
combined figure**:

```json
{"model": "lightgbm",
 "window_days": null,
 "verified": {"scored_count": 41, "pending_count": 1, "mae": 9.8, "rmse": 12.6},
 "backtest": {"scored_count": 1062, "pending_count": 0, "mae": 9.36, "rmse": 12.1}}
```

A model with no rows of one provenance simply has no block for it, rather than a
zero-filled one that would read as "measured, and it was 0". A `source` value nobody
planned for is reported under its own raw name instead of being dropped, so the counts
still add up. Sorting follows `verified` MAE, falling back to `backtest` MAE, with
unscored models last.

`/forecast` and `/electricity/forecast` **filter** to `source = 'daily'` rather than
merely labelling it, because they are the dashboard headline: the seeded backtest
record and the daily record overlap in `forecast_date`, so an unfiltered
`ORDER BY forecast_date DESC LIMIT 1` would present a walk-forward row fitted after
the fact as today's live forecast on any day the daily row is missing. Both echo
`source` anyway, so a caller never has to trust that the filter stayed. `/predictions`
and `/electricity/predictions` still label without filtering — there the backtest rows
are the point.

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
server slots idle. Every 200 on a GET carries `Cache-Control: public, max-age=300`,
which is what stops a looping crawler from occupying those 8 slots — the record it
would be re-reading only changes once a day.

## Automation

- `.github/workflows/daily-pipeline.yml` — **two independent jobs, no `needs:`**:
  - `pipeline` (PM2.5): ingest → engineer features → leakage test → score pending → make forecast → verify.
  - `electricity`: ingest → engineer features → leakage test → score pending → make forecast → verify.

  Each job's last step is its `diagnose.py`, and both now gate on upstream freshness:
  `vericast/pm25/diagnose.py` refuses at 2 days behind, `vericast/elec/diagnose.py` at 5
  (a mirror's normal lag, not a stall). Both numbers live in `vericast/__init__.py` as
  `PM25_STALE_LIMIT_DAYS` / `ELEC_STALE_LIMIT_DAYS`, imported by the gates, `/health`
  and `verify_deployment_readiness.py` — they used to be copied per file, and the
  go-live check's hardcoded `<= 1` failed on data the pipeline passed and `/health`
  called fresh. This is not a nicety — `predict.py` anchors
  `forecast_date = latest_obs + 1 day` on both sides, so a stalled source slides the
  anchor along with it and every other internal-consistency check still passes. Both
  jobs run `verify_alignment()` inside their "Engineer features" step, which is where
  the features(t) → target(t+1) join is asserted on the daily path, and both then run
  their own `leakage_test.py` as a separate step, which re-derives every stored feature
  value from the observations by calendar date. The two checks are complementary: one
  covers the join, the other the values.

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
  the incumbent forever. Its score is printed for drift, not compared. Each retrain
  writes a `<artifact>.window.json` sidecar recording the dates it was fit on, and the
  commit step stages `models/` as a directory so the sidecar travels with its artifact,
  so that printed line can say *whether*
  the incumbent saw the holdout instead of assuming it did — after a skipped week its
  window genuinely ends before the holdout opens and the comparison is fair. Staging
  the directory rather than the four paths by name is load-bearing: a refused retrain
  writes no sidecar, and `git add` on a pathspec that matches nothing exits non-zero
  under `bash -e`, which aborted the step before its own "nothing to commit" branch and
  discarded the sibling target's accepted artifact.

  **No sidecar exists yet.** `models/` holds the two `.txt` artifacts and nothing else:
  the committed models predate `record_training_window()`, and the sidecar is written by
  an *accepted* retrain, so the first one appears the first Sunday a challenger ships.
  Until then the gate prints "no recorded training window, assume it saw them" and
  compares conservatively. This is deliberately not backfilled — the dates those two
  artifacts were fit on are not recoverable from the files, and a hand-written window
  would be a claim about the record that nothing verified.

  Both retrain steps and the commit step carry `if: always()`, for the same reason the
  daily pipeline splits into two jobs: a raising PM2.5 retrain used to abort the
  electricity retrain *and* throw away whichever artifact had already been written. The
  push is a rebase-and-retry loop rather than a bare `git push` — `concurrency:` stops
  this workflow racing itself but not an unrelated push to `main` landing during the
  minutes of training, and a rejected push would discard the retrain.
- `.github/workflows/readiness-gate.yml` — Sundays 06:23 UTC: runs
  `verify_deployment_readiness.py`'s 22 checks against the live database and the deployed
  API. Scheduled between the weekly retrain's 04:00 commit and the daily pipeline's 08:42
  catch-up cron, so it gates the artifact that was just committed. This is the one gate
  nothing used to automate: `ci.yml` has a throwaway Postgres and no server, so the script
  whose whole job is catching a broken deployment depended on someone running it by hand.
  Read-only — every check is a `SELECT` or a `GET`, and the two leakage tests it shells
  out to only read. It wakes the API with one throwaway request first, because Render's
  free tier cold-starts past the gate's 10s timeout and would FAIL every HTTP check on a
  service that is fine.

Both scoring scripts score *every* pending row, not just yesterday's, so a missed run
self-heals on the next one instead of leaving a permanent NULL. Ingest self-heals the
other direction: each run re-scans the last 30 days for the earliest date with no
usable observation and restarts from there, so a day the upstream skipped or served
too thin is refetched once it lands rather than staying behind the resume point.

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

Python 3.11, pinned in `.python-version` — which is committed rather than ignored, so
`pyenv`/`uv` pick locally the same interpreter that `ci.yml`, `daily-pipeline.yml`,
`weekly-retrain.yml` and the `Dockerfile` all pin independently. Changing it means
changing all five.

`tests/test_feature_alignment.py` is the only module that touches a real database. With
no reachable `DATABASE_URL` it skips; against a database that already has rows it asserts
on real pipeline output; against an *empty* one it seeds 40 synthetic days first so CI's
throwaway Postgres still runs the window-frame queries. That seeding refuses any
non-local host, because writing fabricated observations into the managed instance would
contaminate the published record.

`verify_deployment_readiness.py` cannot run in `ci.yml` — its checks need a populated
database and a live server on `API_BASE`, and CI has a throwaway Postgres and no server.
It runs instead in `.github/workflows/readiness-gate.yml`, Sundays at 06:23 UTC against
the live database and the deployed API, after the weekly retrain has committed its
artifact and before the daily pipeline's catch-up cron. Every check is a `SELECT` or a
`GET`, so a failing run reports a problem rather than causing one. `tests/test_readiness_gate.py`
still covers what is checkable with neither dependency: that every `check_*` function is
wired into `CHECKS`, that both targets get the four database checks, that no check raises
instead of returning `False` when its dependency is absent, and that the count above still
matches the code.

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

## Known limits

Deliberately not built. Each line names the ceiling and what would justify crossing it.

- **Rate limiting is in-process, not distributed.** A fixed 120-requests-per-minute
  window per client, in stdlib, in front of `Cache-Control: public, max-age=300` and the
  bounded 8-slot pool. It covers the case the cache does not: a client looping
  `/evaluation` with a cache-busting query string can otherwise hold every pool slot
  until requests time out. The budget is per instance and resets on deploy, and the key
  comes from `X-Forwarded-For`, which the client supplies — so it is a capacity guard,
  not a security control, and a distributed or header-rotating caller is out of scope.
  `slowapi` + Redis is the upgrade at more than one instance. The absence of auth is
  separate and by design — see [`.github/SECURITY.md`](.github/SECURITY.md); this is a
  read-only public record.
- **Hole recovery is best-effort, bounded at 30 days.** Both `resolve_date_range()`s take
  the earlier of `MAX(as_of) + 1 day` and the earliest date in the last `RESCAN_DAYS`
  with no usable observation, so a skipped or too-thin day is refetched once the upstream
  serves it. Beyond that window a hole is still permanent (Maharashtra's
  2025-05-21 → 05-24 predates the change), which stays tolerable because the
  `RANGE ... PRECEDING` frames null out across a gap rather than reaching over it — a
  hole costs accuracy, never correctness. Widening the window only helps if a source
  starts revising in months rather than days.
- **One city, one state.** `model_performance` has no `city` column (its electricity
  counterpart has `state`), because it predates the second target. `CITY` other than
  `Nagpur` therefore fails loudly at import in both `app.py` and `vericast/pm25/score.py`
  rather than quietly overwriting the published leaderboard in place. Everything else is
  keyed on city; crossing this means a `city` column plus a
  `UNIQUE(city, score_date, model)` swap, which is a constraint change rather than an
  addition and so out of `vericast/schema.py`'s scope.
- **The demand mirror tracks a mutable branch.** `DEMAND_CSV_URL` points at
  Grid-Sentinel's `main`, not a commit SHA, deliberately: a pin can only serve dates that
  existed when it was taken, and this mirror backfills late, which is exactly what the
  30-day re-scan above depends on. Two guards stand in for it — a required-column check
  that fails naming the URL before any row is accepted, and a 15,000–40,000 MW
  plausibility bound that catches a unit change (kW, GW) a column-name check cannot see.
  Neither catches a plausible-but-wrong number, and nothing can: the record rests on the
  mirror being honest.
- **The dashboard's CSP is a `<meta http-equiv>`, not a header.** `index.html` is served
  as a static GitHub Pages file, so there is no server to set one. The meta form is
  weaker — `frame-ancestors` and `report-uri` are ignored in it — and `'unsafe-inline'`
  stays unavoidable while the styles, the script and the tab `onclick=` handlers are
  inline. The part that matters still holds: `script-src` pins executable code to this
  file plus the one pinned CDN entry, so an injected `<script src>` from anywhere else
  does not run, and `esc()` guards every interpolated row. Serving the page from
  somewhere with real headers is the upgrade.
- **The deployed hostname and repo slug still say `forecast-lab`.** The Render service is
  `forecast-lab-2l0q.onrender.com` and the remote is `Forecast-Lab`; the rename to
  VeriCast happened in the docs only. Left alone deliberately — rewriting a working
  hostname to match a README breaks the dashboard, and the private-reporting link in
  [`.github/SECURITY.md`](.github/SECURITY.md) 404s if it stops matching the remote. The
  fix is to rename the Render service and the GitHub repo first; the dashboard's
  `API_BASE`, its CSP `connect-src` and that security link all pin the same host and
  move together.
- **Plain `uvicorn`, not `uvicorn[standard]`.** The extra pulls in `uvloop`, `httptools`,
  `watchfiles`, `websockets` and `PyYAML` as unpinned transitive deps, and every other
  line in `requirements.txt` is pinned exactly on purpose. This API is read-only, cached
  for 5 minutes and bounded by an 8-slot pool, so it is nowhere near parser-bound — the
  swap would trade five unpinned dependencies in the image for throughput nothing is
  waiting on. Worth revisiting only if the pool stops being the limit.
- **`python -O` still disarms the self-checks, just not the data gates.** `assert` is
  erased by `-O` / `PYTHONOPTIMIZE=1`, so the checks that guard *data* — both
  `verify_alignment()`s, both `train.py` feature-count checks,
  `save_elec_backtest_results.py`'s — `raise AssertionError` explicitly instead, and stay
  armed however the interpreter is invoked. The remaining bare `assert`s are all in
  `__main__` self-checks (`vericast/gate.py`'s `demo()`, `vericast/local_time.py`), where
  being erased costs nothing because under `-O` the check simply is not the check anymore.
  Nothing in this repo runs with `-O`; the split exists so that adding it for speed can
  only cost self-checks, never a gate. Crossing this means dropping `assert` from the
  self-checks too, which buys nothing until something actually sets the flag.

## Security

Report vulnerabilities privately — see [`.github/SECURITY.md`](.github/SECURITY.md).
Do not open a public issue.

## License

[MIT](LICENSE).

