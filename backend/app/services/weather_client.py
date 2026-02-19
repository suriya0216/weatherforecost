from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings


WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Heavy thunderstorm with hail",
}


@dataclass
class WeatherClient:
    settings: Settings

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def geocode(self, query: str) -> list[dict]:
        response = await self._client.get(
            self.settings.open_meteo_geo_url,
            params={"name": query, "count": 5, "language": "en", "format": "json"},
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])

    async def fetch_weather(self, latitude: float, longitude: float, timezone: str = "auto") -> dict:
        response = await self._client.get(
            f"{self.settings.open_meteo_base_url}/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,is_day",
                "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,uv_index",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 3,
            },
        )
        response.raise_for_status()
        return response.json()

    async def fetch_air_quality(self, latitude: float, longitude: float, timezone: str = "auto") -> dict:
        response = await self._client.get(
            self.settings.open_meteo_air_quality_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "hourly": "us_aqi",
                "forecast_days": 2,
            },
        )
        response.raise_for_status()
        return response.json()


def weather_code_to_label(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WEATHER_CODE_LABELS.get(code, "Unknown")

