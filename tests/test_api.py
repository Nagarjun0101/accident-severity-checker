"""Tests for the FastAPI application (Step 14). Needs: pip install pytest httpx"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD = {"weather": "Clear", "hour": 17, "weekday": 2, "month": 6}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Road Accident Severity Checker" in response.text


def test_options():
    assert "Fog" in client.get("/options").json()["weather"]


def test_stats_have_three_tables():
    stats = client.get("/stats").json()
    assert set(stats.keys()) == {"by_hour", "by_weather", "by_state"}


def test_predict_works():
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert abs(sum(data["class_scores"].values()) - 1) < 0.01


def test_hour_out_of_range_is_rejected():
    assert client.post("/predict", json={**GOOD, "hour": 25}).status_code == 422


def test_missing_field_is_rejected():
    bad = {key: value for key, value in GOOD.items() if key != "weather"}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_weather_is_rejected():
    assert client.post("/predict", json={**GOOD, "weather": "Tornado"}).status_code == 422


def test_bad_lighting_is_rejected():
    assert client.post("/predict", json={**GOOD, "lighting": "Dusk"}).status_code == 422
