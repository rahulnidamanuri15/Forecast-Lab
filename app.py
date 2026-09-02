import logging
import os
from contextlib import asynccontextmanager
from time import monotonic

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from typing import Optional

from vericast import (
    ELEC_STALE_LIMIT_DAYS,
    PM25_STALE_LIMIT_DAYS,
    local_time,
    require_city_of_record,
)

load_dotenv()

# uvicorn owns the handlers and format; this module only needs a named logger so
# its records carry the level and timestamp the log viewer filters on.
log = logging.getLogger("vericast.api")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# model_performance has no city column (see vericast/__init__.py), so another CITY
# would serve Nagpur's accuracy record under its name. Refused at import: a wrong
# label on a published accuracy claim is worse than a dead deployment. One guard
# in vericast/__init__.py, called by all nine modules that read CITY.
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))
STATE = os.getenv("STATE", "Maharashtra")  # target #2: regional electricity demand
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")

# Connect + TLS handshake + auth per request is the largest slice of this API's
# latency, and Neon is a network hop away; the pool hands out an already-open
# connection instead. max_size=8 because Neon's free tier caps concurrent
# connections and this is a read-only GET API - a pool larger than the work queue
# only holds server slots idle. check=check_connection is not optional: Neon closes
# idle connections on its side, so an unchecked pool hands out a dead socket.
#
# Two separate questions, which `_pool is None` alone used to answer at once: "is
# there a usable pool" and "are we running under the lifespan at all". The test
# suite drives handlers through TestClient without a lifespan, so the second one
# has to be answerable - but overloading the first onto it meant a lifespan that
# failed to build a pool left the process serving requests unpooled *and*
# unthrottled, silently, as if it were a test run.
_pool: Optional[ConnectionPool] = None
_under_lifespan = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pool, _under_lifespan
    _under_lifespan = True
    try:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=8,
            timeout=10,                              # wait for a free slot, then fail
            check=ConnectionPool.check_connection,   # never hand out a dead socket
            open=False,
        )
        _pool.open()
        yield
    finally:
        if _pool is not None:
            _pool.close()
        _pool = None
        _under_lifespan = False


app = FastAPI(
    title="VeriCast API",
    description=(
        f"Read-only public record of next-day forecasts published before the "
        f"actual was knowable: {CITY} PM2.5 (ug/m3) and {STATE} peak demand met (MW)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Read-only public GET API: no cookies or auth headers, so allow_credentials stays
# off and only GET is permitted. FRONTEND_ORIGIN must name the real deployed origin
# in production - empty allows no browser origin at all, the safe default over "*".
origins = []
if FRONTEND_ORIGIN:
    origins = [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cache_control(request, call_next):
    """Let caches absorb repeat reads, so the 8-slot pool is not the only limit.

    Every endpoint reads a record the daily pipeline rewrites once a day, so 5
    minutes of staleness is invisible - including /health, whose staleness is
    measured in days.

    ponytail: one blanket max-age, no per-route tuning. Split it when an endpoint
    needs to be fresher than the pipeline that feeds it.
    """
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=300"
    return response


# A fixed-window cap, in stdlib, because the real exposure is the 8-slot pool: a
# client looping /evaluation with a cache-busting query string bypasses the
# Cache-Control above and can hold every slot until requests time out at 10s.
# 120/min is far above a dashboard load (~6 calls) and far below pool saturation.
#
# Not a security control: X-Forwarded-For is client-supplied, so a distributed or
# header-rotating caller is not covered, and nothing here needs it to be - this
# exists to stop one noisy client taking the record offline for everyone else.
#
# ponytail: in-process, so the budget is per instance and resets on deploy - fine
# at one Render instance, slowapi + Redis is the upgrade at two.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 120
# Bounds the counter dict: without a cap, a caller rotating the forwarded header
# turns a rate limiter into a memory leak. Past it, new keys share one bucket, so
# a flood degrades into throttling itself rather than into unbounded growth.
RATE_LIMIT_MAX_CLIENTS = 10_000

_rate_window_start = 0.0
_rate_hits: dict = {}


def over_rate_limit(key, now):
    """Count one request for `key` and report whether it exceeded the window.

    Fixed window rather than sliding: the whole dict is dropped on rollover, which
    is what bounds memory by one window's traffic instead of by every client ever
    seen. The cost is a caller spending two windows' budget across a boundary, not
    worth a deque per client for a cache-fronted read-only API.

    No lock: the event loop is single-threaded and there is no await between the
    read and the write below.
    """
    global _rate_window_start, _rate_hits
    if now - _rate_window_start >= RATE_LIMIT_WINDOW_SECONDS:
        _rate_window_start, _rate_hits = now, {}
    if key not in _rate_hits and len(_rate_hits) >= RATE_LIMIT_MAX_CLIENTS:
        key = ""  # shared overflow bucket; "" is not a reachable client key
    _rate_hits[key] = hits = _rate_hits.get(key, 0) + 1
    return hits > RATE_LIMIT_MAX_REQUESTS


def client_key(request):
    """Best available caller identity. Render terminates TLS at its proxy, so
    request.client.host is the proxy and would throttle every caller as one; the
    leftmost X-Forwarded-For entry is the originating client as that proxy saw it,
    and is spoofable - see the note above."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (
        request.client.host if request.client else "unknown")


@app.middleware("http")
async def rate_limit(request, call_next):
    # Skipped only outside the lifespan, which means the test suite driving ~50
    # requests through TestClient as one client. A running deployment whose pool
    # failed to open is still throttled: it is serving on one direct connection
    # per request, which is exactly when a runaway client does the most damage.
    if _under_lifespan and over_rate_limit(client_key(request), monotonic()):
        # Not cached: Cache-Control is only set on 200s.
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests; slow down and retry shortly."},
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )
    return await call_next(request)


def get_db_connection():
    """A pooled connection, as a context manager.

    Every caller already uses `with get_db_connection() as conn`, which is exactly
    pool.connection()'s contract. Falls back to a direct connect only outside the
    lifespan: the TestClient in tests/ drives handlers without running it.

    Under the lifespan a missing pool is a real fault, not a test - 503 rather
    than a silent per-request connect that works until Neon's connection cap
    turns it into a 500 for everyone.
    """
    if _pool is not None:
        return _pool.connection()
    if _under_lifespan:
        log.error("connection pool is absent under the lifespan")
        raise HTTPException(status_code=503, detail="Service unavailable")
    return psycopg.connect(DATABASE_URL)


def db_error(exc: Exception) -> HTTPException:
    """500 without leaking the raw database exception to the client.

    log.exception, not print: it carries the level and the traceback, so a pool
    timeout is greppable and distinguishable from a bad query in the log viewer.
    """
    log.exception("db error on request: %s", type(exc).__name__)
    return HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Forecasting API",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "leaderboard": "/leaderboard",
            "history": "/history",
            "predictions": "/predictions?model=lightgbm&limit=50&scored_only=false",
            "evaluation": "/evaluation?days=30",
            "electricity": {
                "health": "/electricity/health",
                "forecast": "/electricity/forecast?model=lightgbm",
                "history": "/electricity/history?days=30",
                "predictions": "/electricity/predictions?model=lightgbm&limit=15&scored_only=false",
                "evaluation": "/electricity/evaluation?days=30",
                "leaderboard": "/electricity/leaderboard",
            },
        }
    }

# The two published PM2.5 models. Module-level so /forecast and /predictions
# validate against the same set instead of two literals drifting apart; the
# electricity equivalent is ELEC_MODELS below.
PM25_MODELS = {"lightgbm", "naive_baseline"}

@app.get("/forecast")
async def get_forecast(model: str = "lightgbm"):
    """Return the latest forecast that was published before its actual existed.

    Filtered to source = 'daily', not merely labelled like /predictions. This is the
    dashboard headline: the backtest record runs to 2026-08-28 and the daily record
    is ahead of it, so an unfiltered `ORDER BY forecast_date DESC LIMIT 1` would
    serve a walk-forward row fitted after the fact on any day the daily row is
    missing. `source` is echoed anyway so the caller never has to trust the filter.
    """

    if model not in PM25_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        forecast_date,
                        predicted_pm2_5,
                        actual_pm2_5,
                        model,
                        created_at,
                        source
                    FROM predictions
                    WHERE city = %s
                      AND model = %s
                      AND source = 'daily'
                    ORDER BY forecast_date DESC
                    LIMIT 1
                    """,
                    (CITY, model),
                )

                row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"No {model} forecast found",
            )

        forecast_date, predicted, actual, model_name, created_at, source = row

        return {
            "city": CITY,
            "forecast_date": forecast_date.isoformat(),
            "forecast_pm2_5": float(predicted),
            "model": model_name,
            "actual_pm2_5": (
                float(actual) if actual is not None else None
            ),
            "status": "verified" if actual is not None else "pending",
            "created_at": created_at.isoformat(),
            "source": PROVENANCE.get(source, source),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise db_error(exc)

MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's PM2.5 as today's PM2.5",
    "lightgbm": "LightGBM with lagged and rolling features",
}

# Provenance buckets, in the order they are reported. 'daily' is renamed `verified`
# in the payload because that is what it means to a reader: the row was published
# before its actual existed.
PROVENANCE = {"daily": "verified", "backtest": "backtest"}


def _by_provenance(rows, metric_names, descriptions, window_days):
    """Fold (model, source, scored, pending, *metrics) rows into one entry per
    model with a separate block per provenance.

    Shared by /evaluation and /electricity/evaluation, which differ only in their
    metrics (electricity adds MAPE) and descriptions. The two provenances never
    merge here: there is deliberately no combined figure to quote.
    """
    by_model = {}
    for model_name, source, scored, pending, *metrics in rows:
        entry = by_model.setdefault(model_name, {
            "model": model_name,
            "window_days": window_days,
            "description": descriptions.get(model_name, ""),
        })
        # An unrecognised source is reported under its own raw name rather than
        # dropped: a row that reached the database has to appear somewhere, or the
        # counts silently stop adding up.
        entry[PROVENANCE.get(source, source)] = {
            "scored_count": scored,
            "pending_count": pending,
            **{name: float(v) if v is not None else None
               for name, v in zip(metric_names, metrics)},
        }

    # Sort on the verified MAE, the figure that matters; a model with only backtest
    # rows falls back to that, one with neither sorts last. inf rather than a tuple:
    # two Nones would compare None < None.
    def key(entry):
        for bucket in ("verified", "backtest"):
            mae = entry.get(bucket, {}).get("mae")
            if mae is not None:
                return mae
        return float("inf")

    return sorted(by_model.values(), key=key)


def _leaderboard(rows, metric_names, descriptions):
    """Fold (model, *metrics, sample_size, score_date) rows into a sorted list.

    Shared by /leaderboard and /electricity/leaderboard the way _by_provenance is
    shared by the /evaluation pair; they differ only in their metrics (electricity
    adds MAPE) and descriptions. Lower MAE is better, and mae is nullable, so an
    unscored model sorts last via inf: a None mae raises TypeError comparing
    float < None.
    """
    leaderboard = [
        {
            "model": model,
            # float() on every metric: the columns are FLOAT, so this is a no-op
            # until a NUMERIC migration touches one, at which point an uncoerced
            # Decimal reaches the response.
            **{name: float(v) if v is not None else None
               for name, v in zip(metric_names, metrics)},
            "sample_size": sample_size,
            "as_of": score_date.isoformat(),
            "description": descriptions.get(model, ""),
        }
        for model, *metrics, sample_size, score_date in rows
    ]

    leaderboard.sort(key=lambda x: float("inf") if x["mae"] is None else x["mae"])
    return leaderboard


@app.get("/leaderboard")
async def get_leaderboard():
    """
    Get the leaderboard of model performance, read live from model_performance.
    For each model, returns its most recent scored MAE/RMSE (i.e. the latest
    score_date on record for that model), not a fixed backtest snapshot.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # DISTINCT ON is Postgres-specific and exactly fits "latest per group".
                #
                # source = 'daily' is not optional: experiments/save_backtest_results.py
                # writes one aggregate row per model at the last evaluated date, so a
                # backtest re-run today would carry the most recent score_date and
                # become the published leaderboard with a sample_size in the hundreds.
                cur.execute("""
                    SELECT DISTINCT ON (model) model, mae, rmse, sample_size, score_date
                    FROM model_performance
                    WHERE source = 'daily'
                    ORDER BY model, score_date DESC
                """)
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        return {
            "leaderboard": _leaderboard(rows, ("mae", "rmse"), MODEL_DESCRIPTIONS),
            # One bad day can invert /evaluation's ranking at sample_size 1. Saying
            # so is the fix: the multi-day record is the claim, this is the last
            # data point.
            "note": "Each row is a model's most recently scored day, so a "
                    "sample_size of 1 can rank models differently from the "
                    "multi-day record at /evaluation. Lower MAE and RMSE are "
                    "better.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/predictions")
async def get_predictions(
    model: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    scored_only: bool = False,
):
    """
    Get individual prediction rows from the predictions table.

    - model: filter to a single model (e.g. 'lightgbm'). Omit for all models.
    - limit: max rows returned (1-500), most recent forecast_date first.
    - scored_only: if true, only return rows where actual_pm2_5 is known.
    """
    if model is not None and model not in PM25_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                clauses = ["city = %s"]
                params = [CITY]

                if model:
                    clauses.append("model = %s")
                    params.append(model)

                if scored_only:
                    clauses.append("actual_pm2_5 IS NOT NULL")

                # ponytail: f-string WHERE, not a query builder. Safe because every
                # fragment joined here is a literal in this file and the only user
                # value (`model`) is allowlisted above; real values still go through
                # %s. Revisit if a caller-supplied column or operator reaches this.
                where_clause = " AND ".join(clauses)
                params.append(limit)

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_pm2_5, actual_pm2_5, created_at, source
                    FROM predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model
                    LIMIT %s
                """, params)

                rows = cur.fetchall()

        # Serialization outside the `with`: the pool is max_size=8 and JSON-encoding
        # 500 rows does not need a connection held open for it.
        predictions = []
        for forecast_date, model_name, predicted, actual, created_at, source in rows:
            # Coerced at the unpack, not per output field, so `error` below is
            # covered too. Both columns are FLOAT today, but a NUMERIC migration
            # touching one and not the other makes `actual - predicted` a
            # Decimal-minus-float TypeError.
            predicted = float(predicted) if predicted is not None else None
            actual = float(actual) if actual is not None else None
            error = abs(actual - predicted) if actual is not None and predicted is not None else None
            predictions.append({
                "forecast_date": forecast_date.isoformat(),
                "model": model_name,
                "predicted_pm2_5": predicted,
                "actual_pm2_5": actual,
                "error": error,
                "created_at": created_at.isoformat() if created_at else None,
                # Labelled, not filtered: 'backtest' rows outnumber 'daily' ones
                # ~50:1 here. Filtering would change the existing contract, and a
                # label is enough to tell a published forecast from a launch record.
                "source": PROVENANCE.get(source, source),
            })

        return {
            "predictions": predictions,
            "count": len(predictions),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/evaluation")
async def get_evaluation(days: Optional[int] = Query(None, ge=0)):
    """
    Accuracy over published predictions, grouped by model and split by provenance.

    `verified` is the number the project's name is about: rows written by
    vericast/pm25/predict.py before the actual existed, then scored when the
    observation arrived. `backtest` is the launch record seeded by
    experiments/save_backtest_results.py, whose rows had their actual at write time.
    Both are reported; neither is averaged into the other.

    Default (no `days`) is the full record. Pass `days=N` for a rolling window
    instead (`days=0` is also the full record).
    """
    # ge=0 on the Query rather than a hand-rolled 400: /history already rejects its
    # out-of-range days with FastAPI's own 422, and two status codes for the same
    # class of bad input is a contract the client has to special-case. No upper
    # bound - the full-record branch is unbounded by design.

    full_record = not days  # None or 0

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Window anchored to the app timezone (see vericast/local_time.py),
                # not Postgres's CURRENT_DATE, which is GMT on Neon.
                #
                # Aggregated in SQL rather than fetched row-by-row: the full-record
                # branch has no window to bound it, so pulling every prediction ever
                # published grew with the record itself. COUNT/AVG return one row per
                # model regardless of size, and AVG(...) FILTER skips pending rows
                # without a second query. GROUP BY model, source keeps it to one scan.
                metrics = """
                    SELECT model, source,
                           COUNT(*) FILTER (WHERE predicted_pm2_5 IS NOT NULL
                                              AND actual_pm2_5 IS NOT NULL) AS scored,
                           COUNT(*) FILTER (WHERE predicted_pm2_5 IS NULL
                                               OR actual_pm2_5 IS NULL) AS pending,
                           AVG(ABS(actual_pm2_5 - predicted_pm2_5)) AS mae,
                           SQRT(AVG(POWER(actual_pm2_5 - predicted_pm2_5, 2))) AS rmse
                    FROM predictions
                    WHERE city = %s
                """
                if full_record:
                    cur.execute(metrics + " GROUP BY model, source", (CITY,))
                else:
                    cur.execute(
                        metrics + """
                          AND forecast_date >= %s - %s * INTERVAL '1 day'
                        GROUP BY model, source
                        """,
                        (CITY, local_time.today(), days),
                    )

                rows = cur.fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No predictions found" if full_record
                    else "No predictions found in that window"
                ),
            )

        evaluation = _by_provenance(
            rows,
            metric_names=("mae", "rmse"),
            descriptions=MODEL_DESCRIPTIONS,
            window_days=None if full_record else days,
        )

        return {
            "evaluation": evaluation,
            "note": ("`verified` covers forecasts published before the actual was "
                     "knowable - the record this project exists to keep. `backtest` "
                     "is the walk-forward launch record, measured with the actual "
                     "already in hand. Quote them separately."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/history")
async def get_history(days: int = Query(30, ge=1, le=365)):
    """
    Get the `days` most recently stored PM2.5 observations, oldest first.

    `days` bounds *rows*, not the calendar: the observations table has date gaps
    (see vericast/pm25/ingest.py), so days=30 over a gap reaches further back than
    30 calendar days. That is what a chart wants - N points, no holes at the
    right-hand edge - and why the count is reported as `days_returned`.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # DESC + LIMIT to take the newest rows, reversed below for charting.
                cur.execute("""
                    SELECT as_of, pm2_5, pm10, temperature_2m_mean,
                           wind_speed_10m_max, precipitation_sum
                    FROM observations
                    WHERE city = %s
                    ORDER BY as_of DESC
                    LIMIT %s
                """, (CITY, days))

                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No historical data found")

        # float() on every value column, as in /predictions: all five are FLOAT
        # today, so this only matters if one is ever migrated to NUMERIC.
        history = []
        for row in rows:
            history.append({
                "date": row[0].isoformat(),
                **{name: float(v) if v is not None else None
                   for name, v in zip(("pm2_5", "pm10", "temperature_2m_mean",
                                       "wind_speed_10m_max", "precipitation_sum"),
                                      row[1:])},
            })

        history.reverse()  # oldest first, for charting

        return {
            "historical_data": history,
            "days_returned": len(history),
            "city": CITY
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/health")
async def health():
    """Freshness of PM2.5 observations.

    `source_lag_expected` is the flag the dashboard's LIVE pill reads. Past
    PM25_STALE_LIMIT_DAYS the archive has stalled - and because predict.py anchors
    forecast_date to the latest observation, a stalled source keeps publishing a
    plausible forecast for a date that is no longer tomorrow.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
                latest = cur.fetchone()[0]
        # An empty table is not an outage: without this the arithmetic below raises
        # TypeError and the endpoint whose job is reporting faults reports the wrong one.
        if latest is None:
            return {"status": "no_data", "latest_observation": None, "stale_days": None,
                    "source_lag_expected": None,
                    "detail": f"no observations for {CITY}"}
        stale_days = (local_time.today() - latest).days
        return {"status": "ok", "latest_observation": latest.isoformat(),
                "stale_days": stale_days,
                "source_lag_expected": stale_days <= PM25_STALE_LIMIT_DAYS}
    except Exception as e:
        log.exception("health check failed: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# Target #2: Maharashtra regional electricity peak demand (MW)
#
# Separate routes rather than a `target=` parameter: the column names, units and
# metrics differ (MW vs ug/m3, MAPE only makes sense here), so one set of routes
# serving both would be a branch in every handler.
# ---------------------------------------------------------------------------

# seasonal_naive is electricity-only: a grid's same-weekday-last-week value is a
# real baseline, and PM2.5 has no equivalent published model.
ELEC_MODELS = {"lightgbm", "naive_baseline", "seasonal_naive"}

ELEC_MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's peak demand as today's peak demand",
    "seasonal_naive": "Predict tomorrow's peak demand as the same weekday last week",
    "lightgbm": "LightGBM with lagged demand, rolling aggregates, thermal and calendar features",
}


@app.get("/electricity/health")
async def electricity_health():
    """Freshness of electricity observations.

    `stale_days` of 2-4 is expected: unlike the air-quality archive, the upstream
    demand mirror lags real time. `source_lag_expected` says whether the current
    lag is within that normal band.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(as_of) FROM electricity_observations WHERE state = %s",
                    (STATE,),
                )
                latest = cur.fetchone()[0]
        if latest is None:
            return {"status": "no_data", "state": STATE, "latest_observation": None,
                    "stale_days": None, "source_lag_expected": None,
                    "detail": f"no electricity observations for {STATE}"}
        stale_days = (local_time.today() - latest).days
        return {
            "status": "ok",
            "state": STATE,
            "latest_observation": latest.isoformat(),
            "stale_days": stale_days,
            "source_lag_expected": stale_days <= ELEC_STALE_LIMIT_DAYS,
        }
    except Exception as e:
        log.exception("electricity health check failed: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/electricity/forecast")
async def get_electricity_forecast(model: str = "lightgbm"):
    """Latest peak-demand forecast (MW) published before its actual existed.

    Filtered to source = 'daily' for the same reason as /forecast: this feeds the
    dashboard headline, and an unfiltered LIMIT 1 would present a walk-forward
    backtest row as a live forecast whenever today's daily row is absent.
    """
    if model not in ELEC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT forecast_date, predicted_demand_mw, actual_demand_mw,
                           model, created_at, source
                    FROM electricity_predictions
                    WHERE state = %s AND model = %s AND source = 'daily'
                    ORDER BY forecast_date DESC
                    LIMIT 1
                    """,
                    (STATE, model),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No {model} forecast found")

        forecast_date, predicted, actual, model_name, created_at, source = row

        return {
            "state": STATE,
            "forecast_date": forecast_date.isoformat(),
            "forecast_demand_mw": float(predicted),
            "model": model_name,
            "actual_demand_mw": float(actual) if actual is not None else None,
            "status": "verified" if actual is not None else "pending",
            "created_at": created_at.isoformat(),
            "source": PROVENANCE.get(source, source),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise db_error(exc)


@app.get("/electricity/predictions")
async def get_electricity_predictions(
    model: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    scored_only: bool = False,
):
    """Individual prediction rows from electricity_predictions.

    `error_pct` is the per-row absolute percentage error, which is what the
    dashboard's status badges are banded on.
    """
    if model is not None and model not in ELEC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                clauses = ["state = %s"]
                params = [STATE]

                if model:
                    clauses.append("model = %s")
                    params.append(model)

                if scored_only:
                    clauses.append("actual_demand_mw IS NOT NULL")

                # ponytail: f-string WHERE, safe for the reason /predictions gives.
                where_clause = " AND ".join(clauses)
                params.append(limit)

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_demand_mw,
                           actual_demand_mw, created_at, source
                    FROM electricity_predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model
                    LIMIT %s
                """, params)

                rows = cur.fetchall()

        # Serialization outside the `with`, as in /predictions above.
        predictions = []
        for forecast_date, model_name, predicted, actual, created_at, source in rows:
            # Coerced at the unpack, as in /predictions: `error` and `error_pct`
            # both read these.
            predicted = float(predicted) if predicted is not None else None
            actual = float(actual) if actual is not None else None
            scored = actual is not None and predicted is not None
            error = abs(actual - predicted) if scored else None
            predictions.append({
                "forecast_date": forecast_date.isoformat(),
                "model": model_name,
                "predicted_demand_mw": predicted,
                "actual_demand_mw": actual,
                "error": error,
                "error_pct": (error / actual * 100) if scored and actual else None,
                "created_at": created_at.isoformat() if created_at else None,
                # Labelled, not filtered - see /predictions.
                "source": PROVENANCE.get(source, source),
            })

        return {"predictions": predictions, "count": len(predictions)}
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/evaluation")
async def get_electricity_evaluation(days: Optional[int] = Query(None, ge=0)):
    """Accuracy over published electricity predictions, by model and provenance.

    Same split as /evaluation: `verified` rows were published before the actual
    existed, `backtest` rows are the walk-forward launch record. Adds MAPE
    alongside MAE/RMSE, since a fixed MW error means different things at 20 GW
    and 32 GW.
    """
    # ge=0 for the same reason as /evaluation: one status code for a bad `days`
    # across every endpoint that takes one.

    full_record = not days  # None or 0

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Window anchored to the app timezone, not Postgres CURRENT_DATE
                # (GMT on Neon). Aggregated in SQL for the reason /evaluation gives.
                # MAPE's FILTER drops actual = 0 rather than dividing by it.
                metrics = """
                    SELECT model, source,
                           COUNT(*) FILTER (WHERE predicted_demand_mw IS NOT NULL
                                              AND actual_demand_mw IS NOT NULL) AS scored,
                           COUNT(*) FILTER (WHERE predicted_demand_mw IS NULL
                                               OR actual_demand_mw IS NULL) AS pending,
                           AVG(ABS(actual_demand_mw - predicted_demand_mw)) AS mae,
                           SQRT(AVG(POWER(actual_demand_mw - predicted_demand_mw, 2))) AS rmse,
                           AVG(ABS(actual_demand_mw - predicted_demand_mw)
                               / actual_demand_mw * 100)
                               FILTER (WHERE actual_demand_mw <> 0) AS mape
                    FROM electricity_predictions
                    WHERE state = %s
                """
                if full_record:
                    cur.execute(metrics + " GROUP BY model, source", (STATE,))
                else:
                    cur.execute(
                        metrics + """
                          AND forecast_date >= %s - %s * INTERVAL '1 day'
                        GROUP BY model, source
                        """,
                        (STATE, local_time.today(), days),
                    )

                rows = cur.fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=("No predictions found" if full_record
                        else "No predictions found in that window"),
            )

        evaluation = _by_provenance(
            rows,
            metric_names=("mae", "rmse", "mape"),
            descriptions=ELEC_MODEL_DESCRIPTIONS,
            window_days=None if full_record else days,
        )

        return {
            "state": STATE,
            "evaluation": evaluation,
            "note": ("`verified` covers forecasts published before the actual was "
                     "knowable. `backtest` is the walk-forward launch record. "
                     "Quote them separately."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/leaderboard")
async def get_electricity_leaderboard():
    """Most recent scored MAE/RMSE/MAPE per electricity model.

    The counterpart of /leaderboard, over electricity_model_performance. Same
    caveat: this is the latest scored *day*, so sample_size is normally 1 -
    /electricity/evaluation is the accuracy claim.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # DISTINCT ON as in /leaderboard, plus a STATE filter this table can
                # apply and model_performance cannot. source = 'daily' keeps a
                # backtest re-run out of the published leaderboard.
                cur.execute("""
                    SELECT DISTINCT ON (model)
                           model, mae, rmse, mape, sample_size, score_date
                    FROM electricity_model_performance
                    WHERE state = %s AND source = 'daily'
                    ORDER BY model, score_date DESC
                """, (STATE,))
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        return {
            "state": STATE,
            "leaderboard": _leaderboard(rows, ("mae", "rmse", "mape"),
                                        ELEC_MODEL_DESCRIPTIONS),
            "note": ("Each row is a model's most recently scored day, so a "
                     "sample_size of 1 can rank models differently from the "
                     "multi-day record at /electricity/evaluation. Lower MAE, "
                     "RMSE and MAPE are better."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)

@app.get("/electricity/history")
async def get_electricity_history(days: int = Query(30, ge=1, le=365)):
    """The `days` most recently stored observations, oldest first.

    Peak demand (MW), energy met (MU) and temperature. `days` bounds rows, not
    the calendar - same contract, and same reason, as /history.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT as_of, peak_demand_mw, energy_met_mu,
                           temperature_2m_mean, temperature_2m_max
                    FROM electricity_observations
                    WHERE state = %s
                    ORDER BY as_of DESC
                    LIMIT %s
                """, (STATE, days))
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No historical data found")

        # float() as in /history.
        history = [{
            "date": row[0].isoformat(),
            **{name: float(v) if v is not None else None
               for name, v in zip(("peak_demand_mw", "energy_met_mu",
                                   "temperature_2m_mean", "temperature_2m_max"),
                                  row[1:])},
        } for row in rows]

        history.reverse()  # chronological, for charting

        return {
            "historical_data": history,
            "days_returned": len(history),
            "state": STATE,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


if __name__ == "__main__":
    # Development only; production runs `uvicorn app:app --host 0.0.0.0 --port 8000`.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)