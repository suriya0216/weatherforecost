from app.services.weather_client import _build_ensemble_diagnostics, _build_ensemble_payload


def test_build_ensemble_payload_averages_hourly_values() -> None:
    payload_a = {
        "timezone": "Asia/Kolkata",
        "current": {
            "time": "2026-02-19T09:00",
            "temperature_2m": 26.0,
            "relative_humidity_2m": 62,
            "weather_code": 2,
            "wind_speed_10m": 10.0,
            "is_day": 1,
        },
        "hourly": {
            "time": ["2026-02-19T09:00", "2026-02-19T10:00"],
            "temperature_2m": [26.0, 27.0],
            "precipitation_probability": [10, 20],
            "wind_speed_10m": [10.0, 11.0],
            "uv_index": [4.0, 5.0],
        },
        "daily": {
            "time": ["2026-02-19"],
            "weather_code": [2],
            "temperature_2m_max": [31.0],
            "temperature_2m_min": [24.0],
            "precipitation_probability_max": [36],
        },
    }
    payload_b = {
        "timezone": "Asia/Kolkata",
        "current": {
            "time": "2026-02-19T09:00",
            "temperature_2m": 28.0,
            "relative_humidity_2m": 58,
            "weather_code": 3,
            "wind_speed_10m": 14.0,
            "is_day": 1,
        },
        "hourly": {
            "time": ["2026-02-19T09:00", "2026-02-19T10:00"],
            "temperature_2m": [28.0, 29.0],
            "precipitation_probability": [30, 40],
            "wind_speed_10m": [14.0, 15.0],
            "uv_index": [6.0, 7.0],
        },
        "daily": {
            "time": ["2026-02-19"],
            "weather_code": [3],
            "temperature_2m_max": [33.0],
            "temperature_2m_min": [25.0],
            "precipitation_probability_max": [52],
        },
    }

    ensemble_payload, spread_metrics = _build_ensemble_payload(
        model_payloads=[("best_match", payload_a), ("gfs", payload_b)],
        fallback_payload=payload_a,
    )

    assert ensemble_payload["hourly"]["temperature_2m"][0] == 27.0
    assert ensemble_payload["hourly"]["precipitation_probability"][0] == 20.0
    assert ensemble_payload["daily"]["temperature_2m_max"][0] == 32.0
    assert spread_metrics["temperature_c_mean"] is not None


def test_build_ensemble_diagnostics_contains_confidence() -> None:
    diagnostics = _build_ensemble_diagnostics(
        model_names=["best_match", "gfs_seamless", "icon_seamless"],
        failed_models=["ecmwf_ifs04"],
        spread_metrics={
            "temperature_c_mean": 0.7,
            "temperature_c_max": 1.5,
            "rain_pct_mean": 8.2,
            "wind_kph_mean": 5.1,
            "hourly_points_compared": 48,
        },
    )

    assert diagnostics["provider"] == "Open-Meteo Multi-Model Ensemble"
    assert diagnostics["confidence_score"] >= 20
    assert diagnostics["confidence_level"] in {"high", "moderate", "low"}
    assert diagnostics["models_failed"] == ["ecmwf_ifs04"]
