from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from time import monotonic
from typing import Any

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

RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
NOMINATIM_GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


@dataclass
class WeatherClient:
    settings: Settings
    _cache: dict[str, tuple[float, Any]] = field(default_factory=dict, init=False)
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def geocode(self, query: str) -> list[dict]:
        query = query.strip()
        if not query:
            return []

        payload = await self._get_json(
            url=self.settings.open_meteo_geo_url,
            params={"name": query, "count": 5, "language": "en", "format": "json"},
            cache_key=f"geo:{query.lower()}",
            cache_ttl_seconds=3600,
        )
        results = payload.get("results", [])
        if results:
            return results
        return await self._geocode_fallback(query)

    async def reverse_geocode(self, latitude: float, longitude: float) -> dict | None:
        try:
            payload = await self._get_json(
                url=self.settings.open_meteo_reverse_geo_url,
                params={
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
                cache_key=f"reverse-geo:{round(latitude, 5)}:{round(longitude, 5)}",
                cache_ttl_seconds=3600,
                retry_attempts=1,
            )
            results = payload.get("results", [])
            if results:
                first = results[0]
                return {
                    "name": first.get("name"),
                    "country": first.get("country"),
                    "admin1": first.get("admin1"),
                    "latitude": _as_float(first.get("latitude")) or latitude,
                    "longitude": _as_float(first.get("longitude")) or longitude,
                    "timezone": first.get("timezone", "auto"),
                }
        except httpx.HTTPError:
            pass

        return await self._reverse_geocode_fallback(latitude=latitude, longitude=longitude)

    async def fetch_weather(self, latitude: float, longitude: float, timezone: str = "auto") -> dict:
        return await self._fetch_forecast(latitude=latitude, longitude=longitude, timezone=timezone, model=None)

    async def fetch_weather_model(self, latitude: float, longitude: float, timezone: str, model: str) -> dict:
        return await self._fetch_forecast(latitude=latitude, longitude=longitude, timezone=timezone, model=model)

    async def fetch_weather_ensemble(
        self, latitude: float, longitude: float, timezone: str = "auto"
    ) -> tuple[dict, dict]:
        base_payload = await self.fetch_weather(latitude=latitude, longitude=longitude, timezone=timezone)
        model_payloads: list[tuple[str, dict]] = [("best_match", base_payload)]
        failed_models: list[str] = []

        if self.settings.ensemble_models:
            tasks = [
                self.fetch_weather_model(latitude=latitude, longitude=longitude, timezone=timezone, model=model)
                for model in self.settings.ensemble_models
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for model, result in zip(self.settings.ensemble_models, results):
                if isinstance(result, Exception):
                    failed_models.append(model)
                    continue
                model_payloads.append((model, result))

        ensemble_payload, spread_metrics = _build_ensemble_payload(
            model_payloads=model_payloads,
            fallback_payload=base_payload,
        )
        diagnostics = _build_ensemble_diagnostics(
            model_names=[name for name, _ in model_payloads],
            failed_models=failed_models,
            spread_metrics=spread_metrics,
        )
        return ensemble_payload, diagnostics

    async def fetch_air_quality(self, latitude: float, longitude: float, timezone: str = "auto") -> dict:
        return await self._get_json(
            url=self.settings.open_meteo_air_quality_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "hourly": "us_aqi",
                "forecast_days": 2,
            },
            cache_key=f"aqi:{round(latitude, 4)}:{round(longitude, 4)}:{timezone}",
            cache_ttl_seconds=self.settings.api_cache_ttl_seconds,
        )

    async def fetch_satellite_observation(self, latitude: float, longitude: float) -> dict | None:
        """
        Fetch latest available satellite-derived atmospheric observation from NASA POWER.
        Data is not real-time; provider generally has a delay, so we scan progressively older windows.
        """
        utc_now = datetime.now(tz=timezone.utc)
        for delay_days in (2, 10, 30, 90, 180, 365):
            end_date = (utc_now - timedelta(days=delay_days)).date()
            start_date = end_date - timedelta(days=13)
            cache_key = (
                f"sat:{round(latitude, 4)}:{round(longitude, 4)}:"
                f"{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
            )

            payload = await self._get_json(
                url=NASA_POWER_HOURLY_URL,
                params={
                    "latitude": round(latitude, 5),
                    "longitude": round(longitude, 5),
                    "community": "RE",
                    "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,PRECTOTCORR,CLOUD_AMT",
                    "start": start_date.strftime("%Y%m%d"),
                    "end": end_date.strftime("%Y%m%d"),
                    "format": "JSON",
                    "time-standard": "UTC",
                },
                cache_key=cache_key,
                cache_ttl_seconds=3600,
            )

            parsed = _extract_latest_satellite_observation(payload=payload, utc_now=utc_now)
            if parsed is not None:
                return parsed

        return None

    async def fetch_weather_alerts(self, latitude: float, longitude: float) -> list[dict]:
        try:
            payload = await self._get_json(
                url=self.settings.nws_alerts_url,
                params={"point": f"{latitude:.4f},{longitude:.4f}"},
                headers={
                    "User-Agent": f"{self.settings.app_name}/{self.settings.app_version}",
                    "Accept": "application/geo+json",
                },
                cache_key=f"alerts:{round(latitude, 3)}:{round(longitude, 3)}",
                cache_ttl_seconds=300,
                retry_attempts=1,
            )
        except httpx.HTTPError:
            return []

        features = payload.get("features", [])
        if not isinstance(features, list):
            return []

        alerts: list[dict] = []
        for feature in features[:8]:
            if not isinstance(feature, dict):
                continue
            props = feature.get("properties", {})
            if not isinstance(props, dict):
                continue
            alerts.append(
                {
                    "event": props.get("event"),
                    "severity": props.get("severity"),
                    "headline": props.get("headline"),
                    "effective": props.get("effective"),
                    "expires": props.get("expires"),
                    "sender": props.get("senderName"),
                    "instruction": props.get("instruction"),
                }
            )
        return alerts

    async def _fetch_forecast(
        self, *, latitude: float, longitude: float, timezone: str, model: str | None
    ) -> dict:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,is_day",
            "hourly": (
                "temperature_2m,relative_humidity_2m,pressure_msl,"
                "cloud_cover,precipitation_probability,wind_speed_10m,uv_index"
            ),
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 7,
        }
        if model:
            params["models"] = model

        model_key = model or "best_match"
        return await self._get_json(
            url=f"{self.settings.open_meteo_base_url}/forecast",
            params=params,
            cache_key=f"forecast:{round(latitude, 4)}:{round(longitude, 4)}:{timezone}:{model_key}",
            cache_ttl_seconds=self.settings.api_cache_ttl_seconds,
        )

    async def _geocode_fallback(self, query: str) -> list[dict]:
        try:
            payload = await self._get_json(
                url=NOMINATIM_GEOCODE_URL,
                params={"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1},
                headers={"User-Agent": f"{self.settings.app_name}/{self.settings.app_version}"},
                cache_key=f"geo-fallback:{query.lower()}",
                cache_ttl_seconds=3600,
                retry_attempts=1,
            )
        except httpx.HTTPError:
            return []

        if not isinstance(payload, list):
            return []

        mapped_results: list[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            address = item.get("address", {}) if isinstance(item.get("address"), dict) else {}
            lat = item.get("lat")
            lon = item.get("lon")

            try:
                latitude = float(lat)
                longitude = float(lon)
            except (TypeError, ValueError):
                continue

            name = (
                item.get("name")
                or address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("state")
                or address.get("region")
                or item.get("display_name")
            )

            mapped_results.append(
                {
                    "name": name,
                    "country": address.get("country"),
                    "admin1": address.get("state") or address.get("region"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone": "auto",
                }
            )
        return mapped_results

    async def _reverse_geocode_fallback(self, *, latitude: float, longitude: float) -> dict | None:
        try:
            payload = await self._get_json(
                url=NOMINATIM_REVERSE_URL,
                params={
                    "lat": round(latitude, 6),
                    "lon": round(longitude, 6),
                    "format": "jsonv2",
                    "addressdetails": 1,
                },
                headers={"User-Agent": f"{self.settings.app_name}/{self.settings.app_version}"},
                cache_key=f"reverse-geo-fallback:{round(latitude, 5)}:{round(longitude, 5)}",
                cache_ttl_seconds=3600,
                retry_attempts=1,
            )
        except httpx.HTTPError:
            return None

        if not isinstance(payload, dict):
            return None

        address = payload.get("address", {}) if isinstance(payload.get("address"), dict) else {}
        name = (
            payload.get("name")
            or address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state")
            or payload.get("display_name")
        )
        return {
            "name": name,
            "country": address.get("country"),
            "admin1": address.get("state") or address.get("region"),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
        }

    async def _get_json(
        self,
        *,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_key: str | None = None,
        cache_ttl_seconds: int = 0,
        retry_attempts: int | None = None,
    ) -> Any:
        if cache_key and cache_ttl_seconds > 0:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        attempts = self.settings.api_retry_attempts if retry_attempts is None else max(0, retry_attempts)
        for attempt in range(attempts + 1):
            try:
                response = await self._client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
                if cache_key and cache_ttl_seconds > 0:
                    self._cache_set(cache_key, payload, ttl_seconds=cache_ttl_seconds)
                return payload
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in RETRYABLE_HTTP_STATUS or attempt >= attempts:
                    raise
            except httpx.RequestError:
                if attempt >= attempts:
                    raise
            await asyncio.sleep(0.35 * (attempt + 1))

        raise RuntimeError("Failed to fetch upstream JSON payload.")

    def _cache_get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        return payload

    def _cache_set(self, key: str, payload: Any, *, ttl_seconds: int) -> None:
        self._cache[key] = (monotonic() + max(1, ttl_seconds), payload)


def weather_code_to_label(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WEATHER_CODE_LABELS.get(code, "Unknown")


def _build_ensemble_payload(
    *, model_payloads: list[tuple[str, dict]], fallback_payload: dict
) -> tuple[dict, dict]:
    if not model_payloads:
        return fallback_payload, _empty_spread_metrics()

    model_lookups = [_build_model_lookup(payload) for _, payload in model_payloads]
    fallback_hourly = fallback_payload.get("hourly", {})
    fallback_daily = fallback_payload.get("daily", {})
    hourly_stamps = fallback_hourly.get("time", [])
    daily_stamps = fallback_daily.get("time", [])

    hourly_temp: list[float | None] = []
    hourly_humidity: list[float | None] = []
    hourly_pressure: list[float | None] = []
    hourly_cloud_cover: list[float | None] = []
    hourly_rain: list[float | None] = []
    hourly_wind: list[float | None] = []
    hourly_uv: list[float | None] = []
    temp_spreads: list[float] = []
    rain_spreads: list[float] = []
    wind_spreads: list[float] = []

    for stamp in hourly_stamps:
        temp_values = _collect_hourly_values(model_lookups, "temperature_2m", stamp)
        humidity_values = _collect_hourly_values(model_lookups, "relative_humidity_2m", stamp)
        pressure_values = _collect_hourly_values(model_lookups, "pressure_msl", stamp)
        cloud_values = _collect_hourly_values(model_lookups, "cloud_cover", stamp)
        rain_values = _collect_hourly_values(model_lookups, "precipitation_probability", stamp)
        wind_values = _collect_hourly_values(model_lookups, "wind_speed_10m", stamp)
        uv_values = _collect_hourly_values(model_lookups, "uv_index", stamp)

        hourly_temp.append(_round_or_none(_safe_mean(temp_values), 1))
        hourly_humidity.append(_round_or_none(_safe_mean(humidity_values), 0))
        hourly_pressure.append(_round_or_none(_safe_mean(pressure_values), 1))
        hourly_cloud_cover.append(_round_or_none(_safe_mean(cloud_values), 1))
        hourly_rain.append(_round_or_none(_safe_mean(rain_values), 1))
        hourly_wind.append(_round_or_none(_safe_mean(wind_values), 1))
        hourly_uv.append(_round_or_none(_safe_mean(uv_values), 1))

        if len(temp_values) >= 2:
            temp_spreads.append(max(temp_values) - min(temp_values))
        if len(rain_values) >= 2:
            rain_spreads.append(max(rain_values) - min(rain_values))
        if len(wind_values) >= 2:
            wind_spreads.append(max(wind_values) - min(wind_values))

    daily_weather_code: list[int | None] = []
    daily_temp_max: list[float | None] = []
    daily_temp_min: list[float | None] = []
    daily_rain_max: list[float | None] = []

    for stamp in daily_stamps:
        weather_codes = _collect_daily_codes(model_lookups, "weather_code", stamp)
        temp_max_values = _collect_daily_values(model_lookups, "temperature_2m_max", stamp)
        temp_min_values = _collect_daily_values(model_lookups, "temperature_2m_min", stamp)
        rain_max_values = _collect_daily_values(model_lookups, "precipitation_probability_max", stamp)

        daily_weather_code.append(_mode_or_none(weather_codes))
        daily_temp_max.append(_round_or_none(_safe_mean(temp_max_values), 1))
        daily_temp_min.append(_round_or_none(_safe_mean(temp_min_values), 1))
        daily_rain_max.append(_round_or_none(_safe_mean(rain_max_values), 1))

    current_entries = [lookup.get("current", {}) for lookup in model_lookups]
    current_temp = _safe_mean([_as_float(entry.get("temperature_2m")) for entry in current_entries])
    current_humidity = _safe_mean([_as_float(entry.get("relative_humidity_2m")) for entry in current_entries])
    current_wind = _safe_mean([_as_float(entry.get("wind_speed_10m")) for entry in current_entries])
    current_weather = _mode_or_none([_as_int(entry.get("weather_code")) for entry in current_entries])
    current_is_day = _mode_or_none([_as_int(entry.get("is_day")) for entry in current_entries])

    current_fallback = fallback_payload.get("current", {})
    ensemble_current = {
        "time": current_fallback.get("time"),
        "temperature_2m": _round_or_none(current_temp, 1),
        "relative_humidity_2m": _round_or_none(current_humidity, 0),
        "weather_code": current_weather,
        "wind_speed_10m": _round_or_none(current_wind, 1),
        "is_day": current_is_day if current_is_day is not None else current_fallback.get("is_day"),
    }

    ensemble_payload: dict[str, Any] = {}
    passthrough_keys = (
        "latitude",
        "longitude",
        "generationtime_ms",
        "utc_offset_seconds",
        "timezone",
        "timezone_abbreviation",
        "elevation",
        "current_units",
        "hourly_units",
        "daily_units",
    )
    for key in passthrough_keys:
        if key in fallback_payload:
            ensemble_payload[key] = fallback_payload[key]

    ensemble_payload["current"] = ensemble_current
    ensemble_payload["hourly"] = {
        "time": hourly_stamps,
        "temperature_2m": hourly_temp,
        "relative_humidity_2m": hourly_humidity,
        "pressure_msl": hourly_pressure,
        "cloud_cover": hourly_cloud_cover,
        "precipitation_probability": hourly_rain,
        "wind_speed_10m": hourly_wind,
        "uv_index": hourly_uv,
    }
    ensemble_payload["daily"] = {
        "time": daily_stamps,
        "weather_code": daily_weather_code,
        "temperature_2m_max": daily_temp_max,
        "temperature_2m_min": daily_temp_min,
        "precipitation_probability_max": daily_rain_max,
    }

    spread_metrics = {
        "temperature_c_mean": _round_or_none(_safe_mean(temp_spreads), 2),
        "temperature_c_max": _round_or_none(max(temp_spreads) if temp_spreads else None, 2),
        "rain_pct_mean": _round_or_none(_safe_mean(rain_spreads), 2),
        "wind_kph_mean": _round_or_none(_safe_mean(wind_spreads), 2),
        "hourly_points_compared": len(hourly_stamps),
    }
    return ensemble_payload, spread_metrics


def _build_ensemble_diagnostics(
    *, model_names: list[str], failed_models: list[str], spread_metrics: dict
) -> dict:
    model_count = len(model_names)
    temp_spread = _as_float(spread_metrics.get("temperature_c_mean")) or 0.0
    rain_spread = _as_float(spread_metrics.get("rain_pct_mean")) or 0.0
    wind_spread = _as_float(spread_metrics.get("wind_kph_mean")) or 0.0

    score = 55.0 + min(30.0, model_count * 7.0)
    score -= min(34.0, temp_spread * 5.0 + rain_spread * 0.25 + wind_spread * 1.5)
    score -= min(12.0, len(failed_models) * 3.0)
    final_score = int(max(20.0, min(98.0, round(score))))

    return {
        "provider": "Open-Meteo Multi-Model Ensemble",
        "model_count": model_count,
        "models_used": model_names,
        "models_failed": failed_models,
        "confidence_score": final_score,
        "confidence_level": _confidence_level(final_score),
        "spread": spread_metrics,
        "note": (
            "Higher confidence means models are in closer agreement. "
            "Treat low confidence as higher weather uncertainty."
        ),
    }


def _build_model_lookup(payload: dict) -> dict[str, Any]:
    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})
    return {
        "current": payload.get("current", {}),
        "hourly": hourly,
        "hourly_index": {stamp: idx for idx, stamp in enumerate(hourly.get("time", []))},
        "daily": daily,
        "daily_index": {stamp: idx for idx, stamp in enumerate(daily.get("time", []))},
    }


def _collect_hourly_values(model_lookups: list[dict[str, Any]], key: str, stamp: str) -> list[float]:
    values: list[float] = []
    for lookup in model_lookups:
        idx = lookup["hourly_index"].get(stamp)
        if idx is None:
            continue
        series = lookup["hourly"].get(key, [])
        if idx >= len(series):
            continue
        parsed = _as_float(series[idx])
        if parsed is not None:
            values.append(parsed)
    return values


def _collect_daily_values(model_lookups: list[dict[str, Any]], key: str, stamp: str) -> list[float]:
    values: list[float] = []
    for lookup in model_lookups:
        idx = lookup["daily_index"].get(stamp)
        if idx is None:
            continue
        series = lookup["daily"].get(key, [])
        if idx >= len(series):
            continue
        parsed = _as_float(series[idx])
        if parsed is not None:
            values.append(parsed)
    return values


def _collect_daily_codes(model_lookups: list[dict[str, Any]], key: str, stamp: str) -> list[int]:
    values: list[int] = []
    for lookup in model_lookups:
        idx = lookup["daily_index"].get(stamp)
        if idx is None:
            continue
        series = lookup["daily"].get(key, [])
        if idx >= len(series):
            continue
        parsed = _as_int(series[idx])
        if parsed is not None:
            values.append(parsed)
    return values


def _safe_mean(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return float(mean(numeric_values))


def _mode_or_none(values: list[int | None]) -> int | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return Counter(numeric_values).most_common(1)[0][0]


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_spread_metrics() -> dict:
    return {
        "temperature_c_mean": None,
        "temperature_c_max": None,
        "rain_pct_mean": None,
        "wind_kph_mean": None,
        "hourly_points_compared": 0,
    }


def _confidence_level(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "moderate"
    return "low"


def _valid_nasa_value(value: object) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    if parsed <= -998.0:
        return None
    return parsed


def _parse_nasa_stamp(stamp: object) -> datetime | None:
    if not isinstance(stamp, str) or len(stamp) != 10 or not stamp.isdigit():
        return None
    try:
        parsed = datetime.strptime(stamp, "%Y%m%d%H")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _extract_latest_satellite_observation(payload: dict, *, utc_now: datetime) -> dict | None:
    parameter_block = payload.get("properties", {}).get("parameter", {})
    all_sky = parameter_block.get("ALLSKY_SFC_SW_DWN", {})
    clear_sky = parameter_block.get("CLRSKY_SFC_SW_DWN", {})
    precip = parameter_block.get("PRECTOTCORR", {})
    cloud_amt = parameter_block.get("CLOUD_AMT", {})

    if not all_sky and not clear_sky and not precip and not cloud_amt:
        return None

    available_stamps = {
        key
        for source in (all_sky, clear_sky, precip, cloud_amt)
        if isinstance(source, dict)
        for key in source.keys()
    }
    if not available_stamps:
        return None

    for stamp in sorted(available_stamps, reverse=True):
        stamp_dt = _parse_nasa_stamp(stamp)
        if stamp_dt is None:
            continue

        all_sky_value = _valid_nasa_value(all_sky.get(stamp))
        clear_sky_value = _valid_nasa_value(clear_sky.get(stamp))
        precip_value = _valid_nasa_value(precip.get(stamp))
        cloud_amt_value = _valid_nasa_value(cloud_amt.get(stamp))

        if (
            all_sky_value is None
            and clear_sky_value is None
            and precip_value is None
            and cloud_amt_value is None
        ):
            continue

        cloud_cover_percent = cloud_amt_value
        if cloud_cover_percent is None and all_sky_value is not None and clear_sky_value and clear_sky_value > 0:
            estimated = max(0.0, min(1.0, 1 - (all_sky_value / clear_sky_value))) * 100
            cloud_cover_percent = round(estimated, 1)

        age_hours = round((utc_now - stamp_dt).total_seconds() / 3600, 1)
        return {
            "provider": "NASA POWER",
            "observation_time_utc": stamp_dt.isoformat().replace("+00:00", "Z"),
            "data_age_hours": age_hours,
            "cloud_cover_percent": round(cloud_cover_percent, 1) if cloud_cover_percent is not None else None,
            "precipitation_mm_h": round(precip_value, 2) if precip_value is not None else None,
            "solar_radiation_wm2": round(all_sky_value, 1) if all_sky_value is not None else None,
        }

    return None
