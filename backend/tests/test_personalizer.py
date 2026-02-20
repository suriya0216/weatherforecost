from app.schemas import Coordinates, RoutineProfile, UserPreferences
from app.services.personalizer import build_personalized_response


def test_build_personalized_response_shapes_output() -> None:
    weather_payload = {
        "timezone": "America/New_York",
        "current": {
            "time": "2026-02-19T09:00",
            "temperature_2m": 24.0,
            "relative_humidity_2m": 64,
            "weather_code": 2,
            "wind_speed_10m": 12.0,
            "is_day": 1,
        },
        "hourly": {
            "time": [
                "2026-02-19T09:00",
                "2026-02-19T10:00",
                "2026-02-19T11:00",
                "2026-02-19T12:00",
            ],
            "temperature_2m": [24.0, 25.0, 27.0, 26.0],
            "precipitation_probability": [12, 20, 42, 18],
            "wind_speed_10m": [12.0, 14.0, 18.0, 15.0],
            "uv_index": [2.0, 4.0, 6.0, 5.0],
        },
        "daily": {
            "time": ["2026-02-19"],
            "weather_code": [2],
            "temperature_2m_max": [28.0],
            "temperature_2m_min": [21.0],
            "precipitation_probability_max": [42],
        },
    }
    aqi_payload = {
        "hourly": {
            "time": [
                "2026-02-19T09:00",
                "2026-02-19T10:00",
                "2026-02-19T11:00",
                "2026-02-19T12:00",
            ],
            "us_aqi": [35, 40, 52, 48],
        }
    }

    response = build_personalized_response(
        location=Coordinates(name="New York", latitude=40.71, longitude=-74.0, timezone="America/New_York"),
        weather_payload=weather_payload,
        aqi_payload=aqi_payload,
        routine=RoutineProfile(),
        preferences=UserPreferences(),
    )

    assert response["location"]["name"] == "New York"
    assert response["current"]["weather"] == "Partly cloudy"
    assert len(response["hourly"]) == 4
    assert len(response["insights"]) == 3
    assert response["metrics"]["rain_peak_probability"] == 42
    assert response["activity_windows"][0]["activity"] == "Outdoor workout"


def test_build_personalized_response_adds_quality_and_alert_actions() -> None:
    weather_payload = {
        "timezone": "America/New_York",
        "current": {
            "time": "2026-02-19T09:00",
            "temperature_2m": 24.0,
            "relative_humidity_2m": 64,
            "weather_code": 2,
            "wind_speed_10m": 12.0,
            "is_day": 1,
        },
        "hourly": {
            "time": [
                "2026-02-19T09:00",
                "2026-02-19T10:00",
                "2026-02-19T11:00",
                "2026-02-19T12:00",
            ],
            "temperature_2m": [24.0, 25.0, 27.0, 26.0],
            "precipitation_probability": [12, 20, 42, 18],
            "wind_speed_10m": [12.0, 14.0, 18.0, 15.0],
            "uv_index": [2.0, 4.0, 6.0, 5.0],
        },
        "daily": {
            "time": ["2026-02-19"],
            "weather_code": [2],
            "temperature_2m_max": [28.0],
            "temperature_2m_min": [21.0],
            "precipitation_probability_max": [42],
        },
    }
    aqi_payload = {
        "hourly": {
            "time": [
                "2026-02-19T09:00",
                "2026-02-19T10:00",
                "2026-02-19T11:00",
                "2026-02-19T12:00",
            ],
            "us_aqi": [35, 40, 52, 48],
        }
    }
    diagnostics = {"confidence_score": 78, "model_count": 4, "models_used": ["best_match", "gfs_seamless"]}
    alerts = [{"event": "Heat Advisory", "severity": "Severe"}]
    satellite = {"data_age_hours": 40, "cloud_cover_percent": 82}

    response = build_personalized_response(
        location=Coordinates(name="New York", latitude=40.71, longitude=-74.0, timezone="America/New_York"),
        weather_payload=weather_payload,
        aqi_payload=aqi_payload,
        routine=RoutineProfile(),
        preferences=UserPreferences(),
        satellite_payload=satellite,
        forecast_diagnostics=diagnostics,
        weather_alerts=alerts,
    )

    assert response["forecast_quality"]["confidence_level"] in {"high", "moderate", "low"}
    assert response["forecast_quality"]["high_severity_alerts"] == 1
    assert response["alerts"][0]["event"] == "Heat Advisory"
    assert any("Heat Advisory" in item for item in response["recommended_actions"])
