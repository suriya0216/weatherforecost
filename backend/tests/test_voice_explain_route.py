from fastapi.testclient import TestClient

from app import main as main_module


class _FakeVoiceWeatherClient:
    async def close(self) -> None:
        return None

    async def geocode(self, query: str) -> list[dict]:
        return [
            {
                "name": "London",
                "country": "United Kingdom",
                "admin1": "England",
                "latitude": 51.5072,
                "longitude": -0.1276,
                "timezone": "Europe/London",
            }
        ]

    async def fetch_weather_ensemble(self, **kwargs):  # noqa: ANN003
        return (
            {
                "timezone": "Europe/London",
                "current": {
                    "time": "2026-02-20T09:00",
                    "temperature_2m": 6.0,
                    "relative_humidity_2m": 76,
                    "weather_code": 3,
                    "wind_speed_10m": 18.0,
                    "is_day": 1,
                },
                "hourly": {
                    "time": [
                        "2026-02-20T09:00",
                        "2026-02-20T10:00",
                        "2026-02-20T11:00",
                        "2026-02-20T12:00",
                    ],
                    "temperature_2m": [6.0, 7.0, 8.0, 8.0],
                    "relative_humidity_2m": [76, 74, 73, 72],
                    "pressure_msl": [1006.0, 1007.0, 1008.0, 1009.0],
                    "cloud_cover": [82, 80, 78, 75],
                    "precipitation_probability": [65, 58, 44, 40],
                    "wind_speed_10m": [18.0, 21.0, 19.0, 17.0],
                    "uv_index": [1.0, 2.0, 2.5, 3.0],
                },
                "daily": {
                    "time": [
                        "2026-02-20",
                        "2026-02-21",
                        "2026-02-22",
                        "2026-02-23",
                        "2026-02-24",
                    ],
                    "weather_code": [3, 63, 2, 61, 2],
                    "temperature_2m_max": [9.0, 8.0, 7.0, 6.0, 7.0],
                    "temperature_2m_min": [3.0, 2.0, 1.0, 2.0, 3.0],
                    "precipitation_probability_max": [72, 68, 40, 53, 32],
                },
            },
            {"confidence_score": 71},
        )

    async def fetch_weather_alerts(self, **kwargs):  # noqa: ANN003
        return [{"event": "Rain Advisory", "severity": "Moderate"}]


def test_voice_explain_route_returns_transcript(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeVoiceWeatherClient())
    client = TestClient(main_module.app)

    response = client.get(
        "/api/voice/explain",
        params={"location_query": "London", "language": "es", "horizon_days": 4},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["location"]["name"] == "London"
    assert payload["language"] == "es"
    assert isinstance(payload["sections"], list)
    assert len(payload["sections"]) >= 4
    assert isinstance(payload["transcript"], str)
    assert payload["transcript"]
    assert "reasoning_prompt_template" in payload


def test_voice_explain_route_falls_back_to_english_for_unknown_language(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeVoiceWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/voice/explain", params={"location_query": "London", "language": "xx"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "en"


def test_voice_explain_route_returns_native_tamil_transcript(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeVoiceWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/voice/explain", params={"location_query": "London", "language": "ta"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "ta"
    assert "இப்போது" in payload["transcript"]
