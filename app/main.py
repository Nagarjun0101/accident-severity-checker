"""
FastAPI Backend (Step 11)
Connects the web dashboard (index.html) to the saved ML model (predictor.py)
and fetches live weather data via Open-Meteo with automatic fallbacks.
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
    clean_city = city.split(",")[0].strip()
    if not clean_city:
        raise HTTPException(status_code=400, detail="Please enter a city name")

    headers = {"User-Agent": "RoadAccidentSeverityChecker/1.0 (student-ml-project)"}

    async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as client:
        # Step A: Geocode city -> latitude & longitude (Open-Meteo with OpenStreetMap fallback)
        lat = lon = None
        resolved_loc = clean_city

        try:
            geo_res = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": clean_city, "count": 1, "language": "en", "format": "json"},
            )
            if geo_res.status_code == 200:
                results = geo_res.json().get("results")
                if results:
                    place = results[0]
                    lat, lon = place["latitude"], place["longitude"]
                    resolved_loc = f"{place['name']}, {place.get('admin1', '')} ({place.get('country_code', '')})"
        except Exception:
            pass

        if lat is None:
            try:
                osm_res = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": clean_city, "format": "json", "limit": 1, "addressdetails": 1},
                )
                if osm_res.status_code == 200 and osm_res.json():
                    osm_place = osm_res.json()[0]
                    lat, lon = float(osm_place["lat"]), float(osm_place["lon"])
                    addr = osm_place.get("address", {})
                    name = addr.get("city") or addr.get("town") or addr.get("state") or clean_city
                    state = addr.get("state", "")
                    cc = addr.get("country_code", "").upper()
                    resolved_loc = f"{name}, {state} ({cc})"
            except Exception:
                pass

        if lat is None or lon is None:
            raise HTTPException(status_code=404, detail=f"City '{clean_city}' not found")

        # Step B: Fetch current weather (Open-Meteo with wttr.in fallback if Render IP is rate-limited)
        try:
            weather_res = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,wind_speed_10m,visibility,weather_code,is_day",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone": "auto",
                },
            )
            w_data = weather_res.json()
            if weather_res.status_code == 200 and "current" in w_data:
                curr = w_data["current"]
                local_time = datetime.fromisoformat(curr["time"])
                raw_vis = curr.get("visibility")
                vis_miles = min(round((raw_vis if raw_vis is not None else 16093) / 1609.34, 1), 10.0)

                return {
                    "resolved_location": resolved_loc,
                    "weather": map_weather_code(curr["weather_code"]),
                    "temperature": round(curr["temperature_2m"], 1),
                    "wind_speed": round(curr["wind_speed_10m"], 1),
                    "visibility": vis_miles,
                    "hour": local_time.hour,
                    "weekday": local_time.weekday(),
                    "month": local_time.month,
                    "lighting": "Day" if curr["is_day"] == 1 else "Night",
                }
        except Exception:
            pass

        # Fallback weather source (wttr.in) if Open-Meteo rate-limits Render's shared IP
        try:
            wttr_res = await client.get(f"https://wttr.in/{lat},{lon}", params={"format": "j1"})
            if wttr_res.status_code == 200:
                cond = wttr_res.json()["current_condition"][0]
                now = datetime.now()
                desc = cond["weatherDesc"][0]["value"].lower()
                if "snow" in desc or "ice" in desc or "sleet" in desc:
                    w_group = "Snow"
                elif "rain" in desc or "drizzle" in desc or "shower" in desc or "thunder" in desc:
                    w_group = "Rain"
                elif "fog" in desc or "mist" in desc or "haze" in desc:
                    w_group = "Fog"
                elif "cloud" in desc or "overcast" in desc:
                    w_group = "Cloudy"
                elif "clear" in desc or "sunny" in desc:
                    w_group = "Clear"
                else:
                    w_group = "Other"

                return {
                    "resolved_location": resolved_loc,
                    "weather": w_group,
                    "temperature": round(float(cond["temp_F"]), 1),
                    "wind_speed": round(float(cond["windspeedMiles"]), 1),
                    "visibility": min(round(float(cond["visibilityMiles"]), 1), 10.0),
                    "hour": now.hour,
                    "weekday": now.weekday(),
                    "month": now.month,
                    "lighting": "Day" if 6 <= now.hour < 18 else "Night",
                }
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"Weather provider unavailable: {err}")

    raise HTTPException(status_code=502, detail="Unable to fetch live weather right now")