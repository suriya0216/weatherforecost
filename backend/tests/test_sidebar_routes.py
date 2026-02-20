from fastapi.testclient import TestClient

from app import main as main_module


class _FakeSidebarWeatherClient:
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
                    "wind_speed_10m": 13.0,
                    "is_day": 1,
                },
                "hourly": {
                    "time": [
                        "2026-02-20T09:00",
                        "2026-02-20T10:00",
                        "2026-02-20T11:00",
                    ],
                    "temperature_2m": [6.0, 7.0, 8.0],
                    "precipitation_probability": [15, 20, 25],
                    "wind_speed_10m": [13.0, 14.0, 12.0],
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

    async def fetch_air_quality(self, **kwargs):  # noqa: ANN003
        return {
            "hourly": {
                "time": ["2026-02-20T09:00", "2026-02-20T10:00", "2026-02-20T11:00"],
                "us_aqi": [40, 42, 44],
            }
        }

    async def fetch_weather_alerts(self, **kwargs):  # noqa: ANN003
        return [{"event": "Rain Advisory", "severity": "Moderate"}]


def test_sidebar_statistics_route_returns_expected_payload(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeSidebarWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/sidebar/statistics", params={"location_query": "London"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["section"] == "statistics"
    assert payload["overview"]["average_temp_c"] is not None
    assert len(payload["trend"]) > 0


def test_sidebar_map_route_returns_points(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "weather_client", _FakeSidebarWeatherClient())
    client = TestClient(main_module.app)

    response = client.get("/api/sidebar/map", params={"location_query": "London"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["section"] == "map"
    assert len(payload["points"]) >= 1


def test_sidebar_route_rejects_invalid_section() -> None:
    client = TestClient(main_module.app)
    response = client.get("/api/sidebar/not-a-section", params={"location_query": "London"})
    assert response.status_code == 400
    assert "Unsupported sidebar section" in response.json()["detail"]
