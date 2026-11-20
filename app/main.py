"""
FastAPI Backend (Step 11)
Connects the web dashboard (index.html) to the saved ML model (predictor.py)
and fetches live weather data via Open-Meteo.
"""

from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import predictor

# 1. Initialize FastAPI App
app = FastAPI(title="Road Accident Severity Checker")
HTML_FILE = Path(__file__).parent / "static" / "index.html"


# 2. Define Expected Input Format from the Frontend Form
class PredictRequest(BaseModel):
    weather: str
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    lighting: str = "Day"
    visibility: float = Field(default=10, ge=0, le=100)
    temperature: float = Field(default=65, ge=-60, le=140)
    wind_speed: float = Field(default=5, ge=0, le=200)
    junction: bool = False
    traffic_signal: bool = False
    crossing: bool = False


# 3. Helper to Convert Weather API Codes into Our 6 Model Categories
def map_weather_code(code: int) -> str:
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "Rain"
    return "Other"


# 4. Web & Data Endpoints
@app.get("/")
def serve_home():
    return FileResponse(HTML_FILE)  # Loads the frontend UI


@app.get("/health")
def health_check():
    return {"status": "ok", "model": predictor.MODEL_NAME}


@app.get("/options")
def get_options():
    return {"weather": predictor.WEATHER_OPTIONS}  # Populates dropdowns


@app.get("/stats")
def get_dashboard_stats():
    return predictor.STATS  # Sends historical chart data


# 5. ML Prediction Endpoint
@app.post("/predict")
def predict_risk(req: PredictRequest):
    try:
        return predictor.predict_severity(
            weather=req.weather,
            lighting=req.lighting,
            hour=req.hour,
            weekday=req.weekday,
            month=req.month,
            visibility=req.visibility,
            temperature=req.temperature,
            wind_speed=req.wind_speed,
            junction=req.junction,
            traffic_signal=req.traffic_signal,
            crossing=req.crossing,
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))


# 6. Live City Weather Auto-Fill Endpoint
@app.get("/live-weather")
async def get_live_weather(city: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Step A: Convert city name into latitude & longitude
        geo_res = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "format": "json"},
        )
        results = geo_res.json().get("results")
        if not results:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")

        place = results[0]

        # Step B: Fetch current temperature (°F), wind (mph), visibility, and day/night
        weather_res = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,visibility,weather_code,is_day",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
            },
        )
        curr = weather_res.json()["current"]
        local_time = datetime.fromisoformat(curr["time"])

        # Convert visibility from meters to miles (capped at 10 mi like training data)
        raw_vis = curr.get("visibility")
        vis_miles = min(round((raw_vis if raw_vis is not None else 16093) / 1609.34, 1), 10.0)

        return {
            "resolved_location": f"{place['name']}, {place.get('admin1', '')} ({place.get('country_code', '')})",
            "weather": map_weather_code(curr["weather_code"]),
            "temperature": round(curr["temperature_2m"], 1),
            "wind_speed": round(curr["wind_speed_10m"], 1),
            "visibility": vis_miles,
            "hour": local_time.hour,
            "weekday": local_time.weekday(),
            "month": local_time.month,
            "lighting": "Day" if curr["is_day"] == 1 else "Night",
        }