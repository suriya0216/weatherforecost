from fastapi.testclient import TestClient

from app import main as main_module


class _FakeNotificationWeatherClient:
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
                    "relative_humidity_2m": 70,
                    "weather_code": 2,
                    "wind_speed_10m": 33.0,
                    "is_day": 1,
                },
                "hourly": {
                    "time": [
                        "2026-02-20T09:00",
                        "2026-02-20T10:00",
                        "2026-02-20T11:00",
                    ],
                    "temperature_2m": [6.0, 7.0, 8.0],
                    "precipitation_probability": [45, 20, 15],
                    "wind_speed_10m": [33.0, 22.0, 18.0],
                    "uv_index": [1.0, 2.0, 2.5],
                },
                "daily": {
                    "time": ["2026-02-20", "2026-02-21"],
                    "weather_code": [2, 63],
                    "temperature_2m_max": [9.0, 8.0],
                    "temperature_2m_min": [3.0, 2.0],
                    "precipitation_probability_max": [32, 61],
                },
            },
            {"confidence_score": 72},
        )

    async def fetch_weather_alerts(self, **kwargs):  # noqa: ANN003
        return [
            {
                "event": "Rain Advisory",
                "severity": "Moderate",
                "headline": "Moderate rain likely in the next few hours.",
                "effective": "2026-02-20T08:00:00+00:00",
                "expires": "2026-02-20T18:00:00+00:00",
            }
        ]


def test_notifications_route_returns_items(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeNotificationWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/notifications", params={"location_query": "London"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["location"]["name"] == "London"
    assert isinstance(payload["items"], list)
    assert len(payload["items"]) >= 1
    assert {"id", "title", "message", "severity"}.issubset(payload["items"][0].keys())


def test_notifications_route_applies_limit(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeNotificationWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/notifications", params={"location_query": "London", "limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
