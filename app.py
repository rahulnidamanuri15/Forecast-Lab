import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Optional

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CITY = "Nagpur"

app = FastAPI(
    title="ML Forecasting API",
    description="API for air quality forecasting in Nagpur",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests (for the HTML page)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Create a new database connection"""
    return psycopg.connect(DATABASE_URL)

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
            "evaluation": "/evaluation?days=30"
        }
    }

@app.get("/forecast")
async def get_forecast(model: str = "lightgbm"):
    """Return the latest stored forecast for the requested model."""

    allowed_models = {"lightgbm", "naive_baseline"}

    if model not in allowed_models:
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
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

MODEL_DESCRIPTIONS = {
    "naive_baseline": "Predict tomorrow's PM2.5 as today's PM2.5",
    "lightgbm": "LightGBM with lagged and rolling features",
    "sarima": "SARIMA(1,1,0) with fallback to SES",
}

@app.get("/leaderboard")
async def get_leaderboard():
    """
    Get the leaderboard of model performance, read live from model_performance.
    For each model, returns its most recent scored MAE/RMSE (i.e. the latest
    score_date on record for that model), not a fixed backtest snapshot.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # For each model, grab the row with the most recent score_date.
        # DISTINCT ON is Postgres-specific and exactly fits "latest per group".
        cur.execute("""
            SELECT DISTINCT ON (model) model, mae, rmse, sample_size, score_date
            FROM model_performance
            ORDER BY model, score_date DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

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

        # Sort by MAE (ascending) - lower is better
        leaderboard.sort(key=lambda x: x["mae"])

        return {
            "leaderboard": leaderboard,
            "note": "Lower MAE and RMSE indicate better performance. Each model shows its most recently scored metrics."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def get_predictions(
    model: Optional[str] = None,
    limit: int = 50,
    scored_only: bool = False,
):
    """
    Get individual prediction rows from the predictions table.

    - model: filter to a single model (e.g. 'lightgbm'). Omit for all models.
    - limit: max rows returned, most recent forecast_date first.
    - scored_only: if true, only return rows where actual_pm2_5 is known.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        clauses = ["city = %s"]
        params = ["Nagpur"]

        if model:
            clauses.append("model = %s")
            params.append(model)

        if scored_only:
            clauses.append("actual_pm2_5 IS NOT NULL")

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
        cur.close()
        conn.close()

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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation")
async def get_evaluation(days: int = 30):
    """
    Rolling evaluation: for each model, MAE/RMSE computed over predictions
    scored within the last `days` days (based on forecast_date), plus how
    many of those predictions are still pending an actual.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT model, predicted_pm2_5, actual_pm2_5
            FROM predictions
            WHERE city = %s
              AND forecast_date >= CURRENT_DATE - %s * INTERVAL '1 day'
        """, ("Nagpur", days))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No predictions found in that window")

        by_model = {}
        for model_name, predicted, actual in rows:
            by_model.setdefault(model_name, {"scored": [], "pending": 0})
            if actual is not None and predicted is not None:
                by_model[model_name]["scored"].append((predicted, actual))
            else:
                by_model[model_name]["pending"] += 1

        evaluation = []
        for model_name, data in by_model.items():
            scored = data["scored"]
            entry = {
                "model": model_name,
                "window_days": days,
                "scored_count": len(scored),
                "pending_count": data["pending"],
            }
            if scored:
                errors = [abs(a - p) for p, a in scored]
                squared_errors = [(a - p) ** 2 for p, a in scored]
                entry["mae"] = sum(errors) / len(errors)
                entry["rmse"] = (sum(squared_errors) / len(squared_errors)) ** 0.5
            else:
                entry["mae"] = None
                entry["rmse"] = None
            evaluation.append(entry)

        evaluation.sort(key=lambda x: (x["mae"] is None, x["mae"]))

        return {"evaluation": evaluation}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(days: int = 30):
    """
    Get historical observations for the last N days available.
    Defaults to last 30 days of available data.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get observations - order by date descending and limit to 'days'
        # This gives us the most recent 'days' days of available data
        cur.execute("""
            SELECT as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum
            FROM observations
            WHERE city = 'Nagpur'
            ORDER BY as_of DESC
            LIMIT %s
        """, (days,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

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
            "city": "Nagpur"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This is for development - in production, use uvicorn app:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)