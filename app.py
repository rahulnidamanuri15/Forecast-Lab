import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv
from typing import Optional

from vericast import local_time

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

CITY = os.getenv("CITY", "Nagpur")
STATE = os.getenv("STATE", "Maharashtra")  # target #2: regional electricity demand
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")

app = FastAPI(
    title="VeriCast API",
    description=(
        f"Read-only public record of next-day forecasts published before the "
        f"actual was knowable: {CITY} PM2.5 (ug/m3) and {STATE} peak demand met (MW)."
    ),
    version="0.1.0"
)

# Configure CORS. Read-only public GET API: no cookies or auth headers, so
# allow_credentials stays off and only GET is permitted. FRONTEND_ORIGIN must
# name the real deployed origin in production - empty means no browser origin
# is allowed at all, which is the safe default rather than "*".
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

def get_db_connection():
    """Create a new database connection"""
    return psycopg.connect(DATABASE_URL)


def db_error(exc: Exception) -> HTTPException:
    """500 without leaking the raw database exception to the client."""
    print(f"[error] {type(exc).__name__}: {exc}")  # goes to server logs only
    return HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Forecasting API",
        "docs": "/docs",
        "endpoints": {
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
    """Return the latest stored forecast for the requested model."""

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
                        created_at
                    FROM predictions
                    WHERE city = %s
                      AND model = %s
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

        forecast_date, predicted, actual, model_name, created_at = row

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
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise db_error(exc)

MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's PM2.5 as today's PM2.5",
    "lightgbm": "LightGBM with lagged and rolling features",
}

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
                # For each model, grab the row with the most recent score_date.
                # DISTINCT ON is Postgres-specific and exactly fits "latest per group".
                cur.execute("""
                    SELECT DISTINCT ON (model) model, mae, rmse, sample_size, score_date
                    FROM model_performance
                    ORDER BY model, score_date DESC
                """)
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        leaderboard = []
        for model, mae, rmse, sample_size, score_date in rows:
            leaderboard.append({
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "sample_size": sample_size,
                "as_of": score_date.isoformat(),
                "description": MODEL_DESCRIPTIONS.get(model, ""),
            })

        # Sort by MAE (ascending) - lower is better. model_performance.mae is
        # nullable, so unscored models sort last via inf rather than a bare key:
        # a None mae raises TypeError comparing float < None. Same key as
        # /evaluation and /electricity/evaluation below.
        leaderboard.sort(key=lambda x: float("inf") if x["mae"] is None else x["mae"])

        return {
            "leaderboard": leaderboard,
            "note": "Lower MAE and RMSE indicate better performance. Each model shows its most recently scored metrics."
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

                # ponytail: f-string WHERE, not a query builder. Safe because
                # every fragment joined here is a literal in this file and the
                # only user value (`model`) is allowlisted above; all real
                # values still go through %s. Revisit if a caller-supplied
                # column or operator ever needs to reach this string.
                where_clause = " AND ".join(clauses)
                params.append(limit)

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_pm2_5, actual_pm2_5, created_at
                    FROM predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model
                    LIMIT %s
                """, params)

                rows = cur.fetchall()

                predictions = []
                for forecast_date, model_name, predicted, actual, created_at in rows:
                    error = abs(actual - predicted) if actual is not None and predicted is not None else None
                    predictions.append({
                        "forecast_date": forecast_date.isoformat(),
                        "model": model_name,
                        "predicted_pm2_5": predicted,
                        "actual_pm2_5": actual,
                        "error": error,
                        "created_at": created_at.isoformat() if created_at else None,
                    })

                return {
                    "predictions": predictions,
                    "count": len(predictions),
                }
    except Exception as e:
        raise db_error(e)


@app.get("/evaluation")
async def get_evaluation(days: Optional[int] = None):
    """
    Accuracy over published predictions, grouped by model.

    Default (no `days`) is the full record - every prediction ever published.
    That is the headline MAE the README quotes, and the only figure that should
    be used for accuracy claims. Pass `days=N` for a rolling window instead
    (`days=0` is also the full record).
    """
    if days is not None and days < 0:
        raise HTTPException(status_code=400, detail="days must be >= 0")

    full_record = not days  # None or 0

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Window is anchored to the app timezone (see vericast/local_time.py),
                # not Postgres's CURRENT_DATE, which is GMT on Neon.
                #
                # MAE/RMSE are aggregated in SQL rather than fetched row-by-row:
                # the full-record branch has no window to bound it, so pulling
                # every prediction ever published grew unbounded with the record
                # itself. COUNT/AVG return one row per model regardless of size.
                # AVG(...) FILTER skips the pending rows without a second query.
                metrics = """
                    SELECT model,
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
                    cur.execute(metrics + " GROUP BY model", (CITY,))
                else:
                    cur.execute(
                        metrics + """
                          AND forecast_date >= %s - %s * INTERVAL '1 day'
                        GROUP BY model
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

        evaluation = [
            {
                "model": model_name,
                "window_days": None if full_record else days,
                "scored_count": scored,
                "pending_count": pending,
                "description": MODEL_DESCRIPTIONS.get(model_name, ""),
                "mae": float(mae) if mae is not None else None,
                "rmse": float(rmse) if rmse is not None else None,
            }
            for model_name, scored, pending, mae, rmse in rows
        ]

        # Unscored models sort last. inf rather than a (is_none, mae) tuple:
        # two None maes would make that tuple compare None < None -> TypeError.
        evaluation.sort(key=lambda x: float("inf") if x["mae"] is None else x["mae"])

        return {"evaluation": evaluation}
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/history")
async def get_history(days: int = Query(30, ge=1, le=365)):
    """
    Get historical observations for the last N days available.
    Defaults to last 30 days of available data.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get observations - order by date descending and limit to 'days'
                # This gives us the most recent 'days' days of available data
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

        # Convert to list of dictionaries
        history = []
        for row in rows:
            history.append({
                "date": row[0].isoformat(),
                "pm2_5": row[1],
                "pm10": row[2],
                "temperature_2m_mean": row[3],
                "wind_speed_10m_max": row[4],
                "precipitation_sum": row[5]
            })

        # Return in chronological order (oldest first) for easier charting
        history.reverse()

        return {
            "historical_data": history,
            "days_returned": len(history),
            "city": CITY
        }
    except Exception as e:
        raise db_error(e)


@app.get("/health")
async def health():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(as_of) FROM observations WHERE city = %s", (CITY,))
                latest = cur.fetchone()[0]
        # An empty table is not an outage. Without this the arithmetic below
        # raises TypeError, the blanket handler turns it into 503, and the one
        # endpoint whose job is saying what is wrong reports the wrong thing.
        if latest is None:
            return {"status": "no_data", "latest_observation": None, "stale_days": None,
                    "detail": f"no observations for {CITY}"}
        stale_days = (local_time.today() - latest).days
        return {"status": "ok", "latest_observation": latest.isoformat(), "stale_days": stale_days}
    except Exception as e:
        print(f"[error] health check failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# Target #2: Maharashtra regional electricity peak demand (MW)
#
# Separate routes rather than a `target=` parameter on the existing ones: the
# column names, units and metrics differ (MW vs ug/m3, and MAPE only makes sense
# here), so one set of routes serving both would be a branch in every handler.
# ---------------------------------------------------------------------------

# seasonal_naive is electricity-only - a power grid's same-weekday-last-week
# value is a real baseline; PM2.5 has no equivalent published model, so this
# allowlist stays separate from /forecast's rather than being merged into it.
ELEC_MODELS = {"lightgbm", "naive_baseline", "seasonal_naive"}

ELEC_MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's peak demand as today's peak demand",
    "seasonal_naive": "Predict tomorrow's peak demand as the same weekday last week",
    "lightgbm": "LightGBM with lagged demand, rolling aggregates, thermal and calendar features",
}

# The demand mirror publishes a few days behind real time, so 2-4 stale days is
# normal here and only past this is it a stalled source (see vericast/elec/ingest.py).
ELEC_STALE_LIMIT_DAYS = 5


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
        print(f"[error] electricity health check failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/electricity/forecast")
async def get_electricity_forecast(model: str = "lightgbm"):
    """Latest stored peak-demand forecast (MW) for the requested model."""
    if model not in ELEC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT forecast_date, predicted_demand_mw, actual_demand_mw,
                           model, created_at
                    FROM electricity_predictions
                    WHERE state = %s AND model = %s
                    ORDER BY forecast_date DESC
                    LIMIT 1
                    """,
                    (STATE, model),
                )
                row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No {model} forecast found")

        forecast_date, predicted, actual, model_name, created_at = row

        return {
            "state": STATE,
            "forecast_date": forecast_date.isoformat(),
            "forecast_demand_mw": float(predicted),
            "model": model_name,
            "actual_demand_mw": float(actual) if actual is not None else None,
            "status": "verified" if actual is not None else "pending",
            "created_at": created_at.isoformat(),
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

                # ponytail: f-string WHERE, not a query builder. Safe because
                # every fragment joined here is a literal in this file and the
                # only user value (`model`) is allowlisted above; all real
                # values still go through %s. Revisit if a caller-supplied
                # column or operator ever needs to reach this string.
                where_clause = " AND ".join(clauses)
                params.append(limit)

                cur.execute(f"""
                    SELECT forecast_date, model, predicted_demand_mw,
                           actual_demand_mw, created_at
                    FROM electricity_predictions
                    WHERE {where_clause}
                    ORDER BY forecast_date DESC, model
                    LIMIT %s
                """, params)

                rows = cur.fetchall()

                predictions = []
                for forecast_date, model_name, predicted, actual, created_at in rows:
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
                    })

                return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/evaluation")
async def get_electricity_evaluation(days: Optional[int] = None):
    """Accuracy over published electricity predictions, grouped by model.

    Default (no `days`) is the full record - every prediction ever published, and
    the only figure to quote for accuracy claims. Adds MAPE alongside MAE/RMSE,
    since a fixed MW error means different things at 20 GW and 32 GW.
    """
    if days is not None and days < 0:
        raise HTTPException(status_code=400, detail="days must be >= 0")

    full_record = not days  # None or 0

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Window anchored to the app timezone, not Postgres CURRENT_DATE
                # (which is GMT on Neon). Aggregated in SQL for the same reason
                # as /evaluation: the full-record branch has no window to bound
                # it, so the row count grew with the published record. MAPE's
                # FILTER drops actual = 0 rather than dividing by it.
                metrics = """
                    SELECT model,
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
                    cur.execute(metrics + " GROUP BY model", (STATE,))
                else:
                    cur.execute(
                        metrics + """
                          AND forecast_date >= %s - %s * INTERVAL '1 day'
                        GROUP BY model
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

        evaluation = [
            {
                "model": model_name,
                "window_days": None if full_record else days,
                "scored_count": scored,
                "pending_count": pending,
                "description": ELEC_MODEL_DESCRIPTIONS.get(model_name, ""),
                "mae": float(mae) if mae is not None else None,
                "rmse": float(rmse) if rmse is not None else None,
                "mape": float(mape) if mape is not None else None,
            }
            for model_name, scored, pending, mae, rmse, mape in rows
        ]

        # Unscored models sort last; inf rather than a tuple, since two None maes
        # would compare None < None -> TypeError.
        evaluation.sort(key=lambda x: float("inf") if x["mae"] is None else x["mae"])

        return {"state": STATE, "evaluation": evaluation}
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)


@app.get("/electricity/leaderboard")
async def get_electricity_leaderboard():
    """Most recent scored MAE/RMSE/MAPE per electricity model.

    The counterpart of /leaderboard, reading electricity_model_performance -
    which vericast/elec/score.py writes one row per scored day into. Same
    caveat as the PM2.5 version: this is the latest scored *day*, so
    sample_size is normally 1. /electricity/evaluation is the accuracy claim.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # DISTINCT ON is Postgres-specific and exactly fits "latest per
                # group", as in /leaderboard. Unlike model_performance this table
                # has a state column, so it filters on STATE.
                cur.execute("""
                    SELECT DISTINCT ON (model)
                           model, mae, rmse, mape, sample_size, score_date
                    FROM electricity_model_performance
                    WHERE state = %s
                    ORDER BY model, score_date DESC
                """, (STATE,))
                rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No model performance data found")

        leaderboard = [
            {
                "model": model,
                "mae": float(mae) if mae is not None else None,
                "rmse": float(rmse) if rmse is not None else None,
                "mape": float(mape) if mape is not None else None,
                "sample_size": sample_size,
                "as_of": score_date.isoformat(),
                "description": ELEC_MODEL_DESCRIPTIONS.get(model, ""),
            }
            for model, mae, rmse, mape, sample_size, score_date in rows
        ]

        leaderboard.sort(key=lambda x: float("inf") if x["mae"] is None else x["mae"])

        return {
            "state": STATE,
            "leaderboard": leaderboard,
            "note": ("Lower MAE, RMSE and MAPE indicate better performance. Each model "
                     "shows its most recently scored metrics; use /electricity/evaluation "
                     "for the full record."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise db_error(e)

@app.get("/electricity/history")
async def get_electricity_history(days: int = Query(30, ge=1, le=365)):
    """Historical peak demand (MW), energy (MU) and temperature for the last N
    days of available data."""
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

        history = [{
            "date": row[0].isoformat(),
            "peak_demand_mw": row[1],
            "energy_met_mu": row[2],
            "temperature_2m_mean": row[3],
            "temperature_2m_max": row[4],
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
    # This is for development - in production, use uvicorn app:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)