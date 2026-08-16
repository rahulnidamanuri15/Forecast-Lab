import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Optional

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

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
            "history": "/history"
        }
    }

@app.get("/forecast")
async def get_forecast():
    """
    Get the forecast for the next day using the naive baseline (yesterday's PM2.5).
    Returns the forecasted PM2.5 for tomorrow based on today's observation.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get the most recent observation
        cur.execute("""
            SELECT as_of, pm2_5
            FROM observations
            WHERE city = 'Nagpur'
            ORDER BY as_of DESC
            LIMIT 1
        """)

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="No observations found")

        as_of, pm2_5 = row
        # The forecast for tomorrow (as_of + 1 day) is today's PM2_5 (naive baseline)
        forecast_date = as_of + timedelta(days=1)

        return {
            "forecast_date": forecast_date.isoformat(),
            "forecast_pm2_5": pm2_5,
            "model": "naive_baseline",
            "based_on_observation_date": as_of.isoformat(),
            "observation_pm2_5": pm2_5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def get_leaderboard():
    """
    Get the leaderboard of model performance from our backtests.
    Returns MAE and RMSE for each model we tested.
    """
    # These values are from our backtest runs
    leaderboard = [
        {
            "model": "naive_baseline",
            "mae": 7.3724,
            "rmse": 9.8950,
            "description": "Predict tomorrow's PM2.5 as today's PM2.5"
        },
        {
            "model": "lightgbm",
            "mae": 7.5459,
            "rmse": 10.0399,
            "description": "LightGBM with lagged and rolling features"
        },
        {
            "model": "sarima",
            "mae": 7.5602,
            "rmse": 10.0756,
            "description": "SARIMA(1,1,0) with fallback to SES"
        }
    ]

    # Sort by MAE (ascending) - lower is better
    leaderboard.sort(key=lambda x: x["mae"])

    return {
        "leaderboard": leaderboard,
 "note": "Lower MAE and RMSE indicate better performance"
    }

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