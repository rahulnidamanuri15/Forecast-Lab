from datetime import date
import logging
import os
from contextlib import asynccontextmanager
from time import monotonic

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from typing import Any, Optional

from vericast import (
    ELEC_STALE_LIMIT_DAYS,
    PM25_STALE_LIMIT_DAYS,
    MODEL_PM25,
    MODEL_ELEC,
    local_time,
    require_city_of_record,
    require_state_of_record,
)

from vericast.artifacts import model_metadata
from vericast.publication import publication_timing
from vericast.pm25.train import FEATURE_COLUMNS as PM25_FEATURE_COLUMNS
from vericast.elec.train import FEATURE_COLUMNS as ELEC_FEATURE_COLUMNS

load_dotenv()

# uvicorn owns the handlers and format; this module only needs a named logger so
# its records carry the level and timestamp the log viewer filters on.
log = logging.getLogger("vericast.api")

# Read lazily via get_database_url() so `import app` works for tooling/tests
# without a live DSN; request paths and lifespan fail fast with a clear error.
DATABASE_URL = os.getenv("DATABASE_URL")


def get_database_url() -> str:
    """Return the configured DSN or raise the actionable error."""
    url = DATABASE_URL or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url

# Validate single-city / single-state constraints (Nagpur / Maharashtra).
CITY = require_city_of_record(os.getenv("CITY", "Nagpur"))
STATE = require_state_of_record(os.getenv("STATE", "Maharashtra"))  # Target #2: regional electricity demand
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")

# Neon serverless connection pool (max 8 connections).
_pool: Optional[ConnectionPool] = None
_under_lifespan = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _pool, _under_lifespan
    _under_lifespan = True
    try:
        database_url = get_database_url()
        # Preserve DSN options (notably search_path) so pooled API reads use
        # the same schema as workers. Enforce the timeout last.
        try:
            base_options = (conninfo_to_dict(database_url).get("options", "") or "").strip()
        except Exception:
            base_options = ""
        pool_options = f"{base_options} -c statement_timeout=10000".strip() if base_options else "-c statement_timeout=10000"
        _pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=8,
            timeout=10,                              # wait for a free slot, then fail
            check=ConnectionPool.check_connection,   # verify connection health on checkout
            open=False,
            kwargs={"options": pool_options},
        )
        _pool.open()
        # Fail-fast boot check: a bad DSN previously surfaced only on the first
        # request as a 500. Log loudly here so a misconfigured deploy is obvious
        # in the startup logs instead of masquerading as runtime flakiness.
        # Non-fatal: Render cold-starts the web service before the DB is reachable,
        # so a hard raise here would crash-loop a healthy deploy.
        try:
            with _pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception as exc:
            log.warning("database boot check failed (%s); serving, requests may 503/500 "
                        "until the database is reachable", type(exc).__name__)
        # Single-process rate limiter (see below): warn when running multi-worker,
        # where each worker enforces its own independent bucket.
        if os.getenv("WEB_CONCURRENCY", "1") not in ("1",):
            log.warning("WEB_CONCURRENCY=%s: in-memory rate limiter is per-process; "
                        "effective limit scales with workers. Use 1 worker or an "
                        "external store.", os.getenv("WEB_CONCURRENCY"))
        yield
    finally:
        if _pool is not None:
            _pool.close()
        _pool = None
        _under_lifespan = False


app = FastAPI(
    title="VeriCast API",
    description=(
        f"Read-only t+1 prediction record with separate advance forecasts, delayed "
        f"estimates, and backtests: {CITY} PM2.5 (ug/m3) and {STATE} peak demand met (MW)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict CORS to specified frontend origin(s).
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
    """Attach 5-minute Cache-Control headers to successful GETs (data updates daily).

    Headers only — no server-side cache, no ETag. Correct behind a CDN (query
    strings are in the URL, allow_credentials=False); with no CDN in front every
    load still hits Postgres. Add a shared cache or CDN for edge caching.
    """
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=300"
    return response


# In-memory rate limiting (120 req/min per client IP) to protect the DB connection pool.
# SINGLE-PROCESS ONLY: buckets live in this process's dict, reset on restart, and
# are NOT shared across uvicorn --workers > 1 or multiple Render instances. Deploy
# with 1 worker (current default: `uvicorn app:app` with no --workers flag). If you
# scale horizontally, replace this with a shared store (Redis) - otherwise the
# effective limit multiplies by the worker/instance count.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 120
RATE_LIMIT_MAX_CLIENTS = 10_000

_rate_window_start = 0.0
_rate_hits: dict = {}


def over_rate_limit(key, now):
    """Return True if client has exceeded the request threshold in the current window."""
    global _rate_window_start, _rate_hits
    # Tolerance of 1 microsecond avoids IEEE-754 precision loss where (now + 60) - now < 60
    if now - _rate_window_start >= RATE_LIMIT_WINDOW_SECONDS - 1e-6:
        _rate_window_start, _rate_hits = now, {}
    # Bound memory without collapsing strangers into one shared bucket (which
    # let a rotating-IP attacker throttle innocents). Evict the oldest key to
    # make room; dicts preserve insertion order.
    if key not in _rate_hits and len(_rate_hits) >= RATE_LIMIT_MAX_CLIENTS:
        _rate_hits.pop(next(iter(_rate_hits)))
    _rate_hits[key] = hits = _rate_hits.get(key, 0) + 1
    return hits > RATE_LIMIT_MAX_REQUESTS


def client_key(request):
    """Best available caller identity. Render terminates TLS at its proxy, so
    request.client.host is the proxy and would throttle every caller as one.
    Use the rightmost X-Forwarded-For entry (the hop closest to us, appended by
    the trusted proxy) rather than the leftmost, which is entirely
    attacker-controlled and trivially spoofed to rotate identities. Truncated
    to 45 chars (max IPv6) so a crafted header can't bloat the rate-limit dict.
    Deployments behind N trusted proxies should take entry -(N+1); with the
    default single-proxy setup the rightmost entry is the client as the proxy
    saw it. Set TRUSTED_PROXY_COUNT if you front with additional hops."""
    forwarded = request.headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if parts:
        try:
            hops = int(os.getenv("TRUSTED_PROXY_COUNT", "1"))
        except ValueError:
            hops = 1
        # Rightmost-untrusted: with `hops` trusted proxies appending, the
        # client IP sits `hops` positions from the right.
        idx = max(0, len(parts) - max(1, hops))
        key = parts[idx] if len(parts) > max(1, hops) - 1 else parts[-1]
        # Fall back to the rightmost entry when the chain is shorter than the
        # trusted count (e.g. direct local requests with a spoofed single entry).
        if not key:
            key = parts[-1]
    else:
        key = request.client.host if request.client else "unknown"
    return key[:45] or "unknown"


@app.middleware("http")
async def rate_limit(request, call_next):
    """Enforce per-client request limits under active lifespan."""
    if _under_lifespan and over_rate_limit(client_key(request), monotonic()):
        # 429 short-circuits before CORSMiddleware (added earlier, runs later),
        # so attach the CORS allow-origin header here or cross-origin clients
        # see a network error with no .status and retry 3x across ~10 endpoints,
        # amplifying load exactly when shedding. Mirror the configured origins.
        cors_headers = {"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)}
        origin = request.headers.get("origin", "")
        if origin and (not origins or origin in origins):
            cors_headers["Access-Control-Allow-Origin"] = origin
            cors_headers["Vary"] = "Origin"
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests; slow down and retry shortly."},
            headers=cors_headers,
        )
    return await call_next(request)


def get_db_connection():
    """Return a pooled connection context manager, falling back to direct connection for tests."""
    if _pool is not None:
        return _pool.connection()
    if _under_lifespan:
        log.error("connection pool is absent under the lifespan")
        raise HTTPException(status_code=503, detail="Service unavailable")
    # Direct path (tests / scripts): same 10s statement timeout as the pool.
    return psycopg.connect(get_database_url(), options="-c statement_timeout=10000")


def db_error(exc: Exception) -> HTTPException:
    """Log exception details and return sanitized 500 HTTPException."""
    log.exception("db error on request: %s", type(exc).__name__)
    return HTTPException(status_code=500, detail="Internal server error")


# Pydantic response shapes for the queryable logs. List endpoints return
# `predictions` (this page) plus `count` (this page) and `total` (all rows
# matching the filters, ignoring limit/offset) so clients can paginate.
class PredictionPage(BaseModel):
    predictions: list[dict[str, Any]]
    count: int
    total: int

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Forecasting API",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "leaderboard": "/leaderboard",
            "history": "/history",
            "predictions": "/predictions?model=lightgbm&limit=50&scored_only=false&source=daily",
            "evaluation": "/evaluation?days=30",
            "electricity": {
                "health": "/electricity/health",
                "forecast": "/electricity/forecast?model=lightgbm",
                "history": "/electricity/history?days=30",
                "predictions": "/electricity/predictions?model=lightgbm&limit=15&scored_only=false&source=daily",
                "evaluation": "/electricity/evaluation?days=30",
                "leaderboard": "/electricity/leaderboard",
            },
        }
    }


DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
# Cache dashboard HTML at process start; re-read only when mtime changes so
# every /dashboard hit doesn't pay a file open + 2MB read.
_dashboard_cache: dict = {}


def _dashboard_html() -> str:
    try:
        mtime = os.path.getmtime(DASHBOARD_FILE)
    except OSError:
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    cached = _dashboard_cache.get("entry")
    if cached and cached[0] == mtime:
        return cached[1]
    with open(DASHBOARD_FILE, "r", encoding="utf-8") as fh:
        content = fh.read()
    _dashboard_cache["entry"] = (mtime, content)
    return content


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the VeriCast dashboard with backend-enforced security headers."""
    content = _dashboard_html()
    response = HTMLResponse(content=content)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # connect-src uses a wildcard for onrender.com (plus 'self' and localhost) so
    # a window.API_BASE override (used by the readiness gate and self-hosters)
    # is not blocked by a pinned production hostname. 'unsafe-inline' stays because
    # the dashboard is a single static file with inline <script>/<style> and no
    # build step to hash; object-src/frame-ancestors close the embedding vectors
    # that inline allowances would otherwise open.
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' https://*.onrender.com https://cdn.jsdelivr.net http://localhost:8000 http://127.0.0.1:8000; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'none'"
    )
    return response

# The two published PM2.5 models. Module-level so /forecast and /predictions
# validate against the same set instead of two literals drifting apart; the
# electricity equivalent is ELEC_MODELS below.
PM25_MODELS = {"lightgbm", "naive_baseline"}

@app.get("/forecast")
def get_forecast(model: str = "lightgbm", source: str = "latest"):
    """Latest live prediction; defaults to newest daily/nowcast row, never backtest.

    Default is `latest` (daily/nowcast) because whole-day features(t) are only
    complete after target midnight, so production issuance is structurally a
    delayed estimate (`nowcast`). `latest` returns the newest issuance by
    `(forecast_date, created_at)` — it does not prefer `daily` on tied dates.
    Advance-only callers pass `source=daily` explicitly and treat a 404 as
    "no advance forecast on record".
    Scoring status is separate from timing_status.
    """
    provenance_clause = source_filter(source, allow_latest=True)

    if model not in PM25_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
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
                      AND {provenance_clause}
                    ORDER BY forecast_date DESC, created_at DESC
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

        # predicted_pm2_5 is nullable in the schema; a NULL row is unscoreable
        # and must not 500 on float(None). Treat as absent record.
        if predicted is None:
            raise HTTPException(
                status_code=404,
                detail=f"No {model} forecast found",
            )

        return {
            "city": CITY,
            "forecast_date": forecast_date.isoformat(),
            "forecast_pm2_5": float(predicted),
            "model": model_name,
            "actual_pm2_5": (
                float(actual) if actual is not None else None
            ),
            "status": "verified" if actual is not None else "pending",
            "created_at": created_at.isoformat() if created_at else None,
            "source": PROVENANCE.get(source, source),
            **timing_payload(forecast_date, created_at, source, "UTC"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise db_error(exc)

MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's PM2.5 as today's PM2.5",
    "lightgbm": "LightGBM with lagged and rolling features",
}

# Keep the legacy API alias `verified` for daily provenance. It means issued
# before target midnight, NOT that an actual has arrived (see `status`).
# `publication_source` exposes the stored name without breaking existing clients.
PROVENANCE = {"daily": "verified", "nowcast": "nowcast", "backtest": "backtest"}


def source_filter(source, allow_latest=False):
    """SQL fragments from a closed allowlist; never interpolate caller input."""
    clauses = {
        "daily": "source = 'daily'",
        "nowcast": "source = 'nowcast'",
        "backtest": "source = 'backtest'",
    }
    if allow_latest:
        clauses["latest"] = "source IN ('daily', 'nowcast')"
    if source not in clauses:
        raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")
    return clauses[source]


def timing_payload(forecast_date, created_at, source, zone):
    timing = publication_timing(forecast_date, zone, created_at)
    classified_source = timing.pop("source")
    if source == "backtest" or created_at is None:
        timing.update(issued_at=created_at,
                      timing_status="backtest" if source == "backtest" else "unknown")
    return {"publication_source": source, **timing,
            "provenance_consistent": (classified_source == source
                                      if created_at is not None and source != "backtest"
                                      else None)}


def artifact_diagnostics(path, features, zone):
    try:
        artifact = model_metadata(path, len(features))
    except Exception as exc:
        log.warning("artifact diagnostic failed: %s", type(exc).__name__)
        artifact = {"status": "error", "detail": "Model artifact missing or invalid"}
    return {"model_bundle": artifact, "horizon_days": 1, "target_timezone": zone,
            "publication_policy": "Before target midnight: daily; at/after: nowcast"}


@app.get("/diagnostics")
def diagnostics():
    return artifact_diagnostics(MODEL_PM25, PM25_FEATURE_COLUMNS, "UTC")


@app.get("/electricity/diagnostics")
def electricity_diagnostics():
    return artifact_diagnostics(MODEL_ELEC, ELEC_FEATURE_COLUMNS, "Asia/Kolkata")


def _by_provenance(rows, metric_names, descriptions, window_days):
    """Fold (model, source, scored, pending, *metrics) rows into one entry per
    model with a separate block per provenance.

    Shared by /evaluation and /electricity/evaluation, which differ only in their
    metrics (electricity adds MAPE) and descriptions. The three provenances never
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

    # Sort on the verified MAE only, the figure that matters. A backtest-only
    # model sorts last (inf) rather than ranking its fitted-after-the-fact MAE
    # against live verified MAEs — the docstring above promises provenances never
    # merge, and sort order is part of that promise.
    def key(entry):
        mae = entry.get("verified", {}).get("mae")
        return mae if mae is not None else float("inf")

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
def get_leaderboard(source: str = "daily"):
    """Latest scored day per model in exactly one provenance, never combined."""
    provenance_clause = source_filter(source)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT DISTINCT ON (model) model, mae, rmse, sample_size, score_date
                    FROM model_performance
                    WHERE {provenance_clause} AND city = %s
                    ORDER BY model, score_date DESC
                """, (CITY,))
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        return {
            "source": PROVENANCE[source],
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


@app.get("/predictions", response_model=PredictionPage)
def get_predictions(
    model: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    scored_only: bool = False,
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get individual prediction rows from the predictions table.

    - model: filter to a single model (e.g. 'lightgbm'). Omit for all models.
    - limit: max rows returned (1-500), most recent forecast_date first.
    - offset: rows to skip for pagination (default: 0).
    - scored_only: if true, only return rows where actual_pm2_5 is known.
    - source: filter to one provenance, 'daily' or 'backtest'. Omit for both.
    - start_date / end_date: filter forecast_date range (YYYY-MM-DD).

    `source` filters in SQL, before the LIMIT. That distinction is the whole
    point of the parameter: a caller that wants only published rows and filters
    the payload itself gets `limit` rows of interleaved provenance and keeps
    whatever fraction survives, so its window silently shrinks as the two
    records overlap in forecast_date. Values are the stored ones ('daily'),
    not the renamed ones the payload reports ('verified').

    No rows is **200 with an empty list**, not the 404 the record routes answer.
    Deliberate, and the dividing line for the whole API: this is a filtered log,
    so "nothing matches model=X, source=Y, scored_only=Z" is an answer about the
    filter the caller composed. /forecast, /history, /leaderboard and /evaluation
    each return *the* record, where absence means the pipeline has not produced
    one yet - a fault worth a status code. `count` is this page, `total` is all
    rows matching the filters ignoring limit/offset.
    """
    if model is not None and model not in PM25_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if start_date:
        try:
            date.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Expected YYYY-MM-DD.")
    if end_date:
        try:
            date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Expected YYYY-MM-DD.")
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date.")

    # Allowlisted rather than passed through: an unrecognised source is a 400
    # here instead of an empty `predictions` list, which a caller cannot tell
    # apart from "nothing published yet".
    if source is not None and source not in PROVENANCE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source: {source}. Expected one of {sorted(PROVENANCE)}.",
        )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                clauses = ["city = %s"]
                params = [CITY]

                if model:
                    clauses.append("model = %s")
                    params.append(model)

                if source:
                    clauses.append("source = %s")
                    params.append(source)

                if scored_only:
                    clauses.append("actual_pm2_5 IS NOT NULL")

                if start_date:
                    clauses.append("forecast_date >= %s")
                    params.append(start_date)

                if end_date:
                    clauses.append("forecast_date <= %s")
                    params.append(end_date)

                # Safe parameterized query with allowlisted clauses
                where_clause = " AND ".join(clauses)
                count_params = list(params)
                params.extend([limit, offset])

                cur.execute(f"""
                    SELECT COUNT(*) FROM predictions WHERE {where_clause}
                """, count_params)
                total_row = cur.fetchone()
                total = int(total_row[0]) if total_row else 0

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_pm2_5, actual_pm2_5, created_at, source
                    FROM predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model, source
                    LIMIT %s OFFSET %s
                """, params)

                rows = cur.fetchall()

        # Serialization outside the `with`: the pool is max_size=8 and JSON-encoding
        # 500 rows does not need a connection held open for it.
        predictions = []
        # row_source, not `source`: that name is the query parameter now, and a
        # loop that rebinds it would hand the next reader the last row's
        # provenance where they expected the caller's filter.
        for forecast_date, model_name, predicted, actual, created_at, row_source in rows:
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
                # Labelled, not filtered by default: 'backtest' rows outnumber
                # 'daily' ones ~50:1 here, and a caller may legitimately want the
                # launch record. A caller that does not passes ?source=daily,
                # which filters before the LIMIT rather than after it.
                "source": PROVENANCE.get(row_source, row_source),
                **timing_payload(forecast_date, created_at, row_source, "UTC"),
            })

        return {
            "predictions": predictions,
            "count": len(predictions),
            "total": total,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/evaluation")
def get_evaluation(days: Optional[int] = Query(None, ge=0)):
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
                #
                # `pending` means "published, still waiting for its actual", so it
                # requires a prediction - the same predicted_* IS NOT NULL that
                # vericast/pm25/score.py's SCORE_SQL requires before it will ever fill
                # one in. Counting `predicted IS NULL OR actual IS NULL` instead
                # labelled a NULL-prediction row as pending when nothing can ever
                # score it. Such a row is unscoreable and counts in neither column, so
                # scored + pending is deliberately <= the row total.
                metrics = """
                    SELECT model, source,
                           COUNT(*) FILTER (WHERE predicted_pm2_5 IS NOT NULL
                                              AND actual_pm2_5 IS NOT NULL) AS scored,
                           COUNT(*) FILTER (WHERE predicted_pm2_5 IS NOT NULL
                                              AND actual_pm2_5 IS NULL) AS pending,
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
            "note": ("`verified` covers advance forecasts issued before target midnight. "
                     "`nowcast` covers delayed estimates issued at or after cutoff. "
                     "`backtest` is the walk-forward launch record. All metrics remain separate."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/history")
def get_history(days: int = Query(30, ge=1, le=365)):
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
def health():
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
        # PM2.5 target days are UTC: compare against UTC today, not IST, or
        # staleness inflates by one day between 00:00–05:30 IST.
        stale_days = (local_time.today("UTC") - latest).days
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
def electricity_health():
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
        # Electricity target days are Asia/Kolkata: explicit IST today.
        stale_days = (local_time.today("Asia/Kolkata") - latest).days
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
def get_electricity_forecast(model: str = "lightgbm", source: str = "latest"):
    """Latest live demand prediction; `latest` is newest daily/nowcast issuance, not backtest."""
    provenance_clause = source_filter(source, allow_latest=True)
    if model not in ELEC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT forecast_date, predicted_demand_mw, actual_demand_mw,
                           model, created_at, source
                    FROM electricity_predictions
                    WHERE state = %s AND model = %s AND {provenance_clause}
                    ORDER BY forecast_date DESC, created_at DESC
                    LIMIT 1
                    """,
                    (STATE, model),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No {model} forecast found")

        forecast_date, predicted, actual, model_name, created_at, source = row

        # Defensive: schema says NOT NULL but a corrupt write must 404, not 500.
        if predicted is None:
            raise HTTPException(status_code=404, detail=f"No {model} forecast found")

        return {
            "state": STATE,
            "forecast_date": forecast_date.isoformat(),
            "forecast_demand_mw": float(predicted),
            "model": model_name,
            "actual_demand_mw": float(actual) if actual is not None else None,
            "status": "verified" if actual is not None else "pending",
            "created_at": created_at.isoformat() if created_at else None,
            "source": PROVENANCE.get(source, source),
            **timing_payload(forecast_date, created_at, source, "Asia/Kolkata"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise db_error(exc)


@app.get("/electricity/predictions", response_model=PredictionPage)
def get_electricity_predictions(
    model: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    scored_only: bool = False,
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Individual prediction rows from electricity_predictions.

    - model: filter to a single model. Omit for all models.
    - limit: max rows returned (1-500), most recent forecast_date first.
    - offset: rows to skip for pagination (default: 0).
    - scored_only: if true, only return rows where actual_demand_mw is known.
    - source: filter to one provenance, 'daily' or 'backtest'. Omit for both.
    - start_date / end_date: filter forecast_date range (YYYY-MM-DD).

    `error_pct` is the per-row absolute percentage error, which is what the
    dashboard's status badges are banded on.

    `source` filters one provenance in SQL before the LIMIT, as in /predictions.
    It matters more here: the elec backtest tail is ~2 weeks behind the daily
    record rather than a year, so a DESC window of any size already mixes the
    two and post-filtering a payload would keep only a fraction of it.

    No rows is 200 with an empty list, for the reason /predictions gives.
    `count` is this page, `total` is all rows matching the filters.
    """
    if model is not None and model not in ELEC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if start_date:
        try:
            date.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Expected YYYY-MM-DD.")
    if end_date:
        try:
            date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Expected YYYY-MM-DD.")
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date.")

    if source is not None and source not in PROVENANCE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source: {source}. Expected one of {sorted(PROVENANCE)}.",
        )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                clauses = ["state = %s"]
                params = [STATE]

                if model:
                    clauses.append("model = %s")
                    params.append(model)

                if source:
                    clauses.append("source = %s")
                    params.append(source)

                if scored_only:
                    clauses.append("actual_demand_mw IS NOT NULL")

                if start_date:
                    clauses.append("forecast_date >= %s")
                    params.append(start_date)

                if end_date:
                    clauses.append("forecast_date <= %s")
                    params.append(end_date)

                # Safe parameterized query with allowlisted clauses
                where_clause = " AND ".join(clauses)
                count_params = list(params)
                params.extend([limit, offset])

                cur.execute(f"""
                    SELECT COUNT(*) FROM electricity_predictions WHERE {where_clause}
                """, count_params)
                total_row = cur.fetchone()
                total = int(total_row[0]) if total_row else 0

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_demand_mw,
                           actual_demand_mw, created_at, source
                    FROM electricity_predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model, source
                    LIMIT %s OFFSET %s
                """, params)

                rows = cur.fetchall()

        # Serialization outside the `with`, as in /predictions above.
        predictions = []
        # row_source for the reason /predictions gives: `source` is the query
        # parameter here.
        for forecast_date, model_name, predicted, actual, created_at, row_source in rows:
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
                # Labelled, not filtered by default - see /predictions.
                "source": PROVENANCE.get(row_source, row_source),
                **timing_payload(forecast_date, created_at, row_source, "Asia/Kolkata"),
            })

        return {"predictions": predictions, "count": len(predictions), "total": total}
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/evaluation")
def get_electricity_evaluation(days: Optional[int] = Query(None, ge=0)):
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
                # (GMT on Neon). Aggregated in SQL for the reason /evaluation gives,
                # and `pending` requires a prediction for the same reason: only a row
                # with predicted_demand_mw set can ever be scored by
                # vericast/elec/score.py. MAPE's FILTER drops actual = 0 rather than
                # dividing by it.
                metrics = """
                    SELECT model, source,
                           COUNT(*) FILTER (WHERE predicted_demand_mw IS NOT NULL
                                              AND actual_demand_mw IS NOT NULL) AS scored,
                           COUNT(*) FILTER (WHERE predicted_demand_mw IS NOT NULL
                                              AND actual_demand_mw IS NULL) AS pending,
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
            "note": ("`verified` covers advance forecasts issued before target midnight. "
                     "`nowcast` covers delayed estimates issued at or after cutoff. "
                     "`backtest` is the walk-forward launch record. Quote them separately."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/leaderboard")
def get_electricity_leaderboard(source: str = "daily"):
    """Latest scored demand day per model in exactly one provenance."""
    provenance_clause = source_filter(source)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # DISTINCT ON as in /leaderboard, plus a STATE filter this table can
                # apply and model_performance cannot. source = 'daily' keeps a
                # backtest re-run out of the published leaderboard.
                cur.execute(f"""
                    SELECT DISTINCT ON (model)
                           model, mae, rmse, mape, sample_size, score_date
                    FROM electricity_model_performance
                    WHERE state = %s AND {provenance_clause}
                    ORDER BY model, score_date DESC
                """, (STATE,))
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        return {
            "state": STATE,
            "source": PROVENANCE[source],
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
def get_electricity_history(days: int = Query(30, ge=1, le=365)):
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