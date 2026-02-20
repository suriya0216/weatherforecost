from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import Coordinates, MLPredictRequest, MLTrainRequest, PersonalizedForecastRequest
from app.services.ml_engine import WeatherMLEngine
from app.services.personalizer import build_personalized_response
from app.services.weather_client import WeatherClient, weather_code_to_label


settings = get_settings()
weather_client = WeatherClient(settings=settings)
ml_engine = WeatherMLEngine(settings=settings)

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_VOICE_LANGUAGES = {
    "en": {"name": "English", "speech_language": "en-US"},
    "ta": {"name": "Tamil", "speech_language": "ta-IN"},
    "hi": {"name": "Hindi", "speech_language": "hi-IN"},
    "es": {"name": "Spanish", "speech_language": "es-ES"},
    "fr": {"name": "French", "speech_language": "fr-FR"},
}

WEATHER_REASON_PROMPT_TEMPLATE = (
    "You are an advanced meteorological AI assistant.\n\n"
    "Based on the following weather data:\n"
    "Temperature: {temp}\n"
    "Humidity: {humidity}\n"
    "Pressure: {pressure}\n"
    "Wind Speed: {wind}\n"
    "Cloud Coverage: {cloud}\n"
    "Rain Probability: {rain_prob}\n\n"
    "Explain:\n"
    "1. Current weather condition.\n"
    "2. Why it is happening (scientific atmospheric reason).\n"
    "3. What may happen tomorrow.\n"
    "4. Trend for next 3 days.\n"
    "5. Use simple human-friendly language.\n\n"
    "Keep it under 150 words.\n"
    "Sound like a professional weather expert."
)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await weather_client.close()
    await ml_engine.close()


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/geocode")
async def geocode(query: str = Query(min_length=2, max_length=80)) -> dict:
    try:
        results = await weather_client.geocode(query=query)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding provider error: {exc}") from exc

    mapped = [
        {
            "name": item.get("name"),
            "country": item.get("country"),
            "admin1": item.get("admin1"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "timezone": item.get("timezone", "auto"),
        }
        for item in results
    ]
    return {"results": mapped}


@app.get("/api/geocode/reverse")
async def reverse_geocode(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
) -> dict:
    try:
        result = await weather_client.reverse_geocode(latitude=latitude, longitude=longitude)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Reverse geocoding provider error: {exc}") from exc
    return {"result": result}


@app.post("/api/forecast/personalized")
async def personalized_forecast(payload: PersonalizedForecastRequest) -> dict:
    location = await _resolve_location(payload.location, payload.location_query)

    forecast_diagnostics = None
    weather_alerts: list[dict] = []
    try:
        (weather_payload, forecast_diagnostics), aqi_payload, weather_alerts = await asyncio.gather(
            weather_client.fetch_weather_ensemble(
                latitude=location.latitude,
                longitude=location.longitude,
                timezone=location.timezone,
            ),
            weather_client.fetch_air_quality(
                latitude=location.latitude,
                longitude=location.longitude,
                timezone=location.timezone,
            ),
            weather_client.fetch_weather_alerts(
                latitude=location.latitude,
                longitude=location.longitude,
            ),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Weather provider error: {exc}") from exc

    satellite_payload = None
    try:
        satellite_payload = await weather_client.fetch_satellite_observation(
            latitude=location.latitude,
            longitude=location.longitude,
        )
    except httpx.HTTPError:
        # Satellite feed is optional; forecast should still be returned when this provider is unavailable.
        satellite_payload = None

    return build_personalized_response(
        location=location,
        weather_payload=weather_payload,
        aqi_payload=aqi_payload,
        routine=payload.routine,
        preferences=payload.preferences,
        satellite_payload=satellite_payload,
        forecast_diagnostics=forecast_diagnostics,
        weather_alerts=weather_alerts,
    )


@app.post("/api/ml/train")
async def train_ml_model(payload: MLTrainRequest) -> dict:
    location = await _resolve_location(payload.location, payload.location_query)

    try:
        return await ml_engine.train(
            location=location,
            history_days=payload.history_days or settings.ml_history_days,
            epochs=payload.epochs or settings.ml_default_epochs,
            force_retrain=payload.force_retrain,
            min_retrain_hours=payload.min_retrain_hours or settings.ml_min_retrain_hours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"ML training upstream provider error: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ml/predict")
async def predict_with_ml(payload: MLPredictRequest) -> dict:
    location = await _resolve_location(payload.location, payload.location_query)

    try:
        return await ml_engine.predict(
            location=location,
            horizon_hours=payload.horizon_hours or settings.ml_default_horizon_hours,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"ML prediction upstream provider error: {exc}") from exc


@app.get("/api/ml/metrics")
async def ml_metrics() -> dict:
    try:
        return await ml_engine.get_metrics()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sidebar/{section}")
async def sidebar_section(
    section: str,
    location_query: str | None = Query(default=None, max_length=80),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    timezone: str = Query(default="auto"),
) -> dict:
    normalized_section = section.strip().lower()
    if normalized_section == "settings":
        normalized_section = "setting"

    supported_sections = {"dashboard", "statistics", "map", "calendar", "setting"}
    if normalized_section not in supported_sections:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported sidebar section '{section}'. Use one of: {sorted(supported_sections)}.",
        )

    if latitude is not None and longitude is not None:
        resolved_location = Coordinates(
            name=location_query or "Selected location",
            latitude=latitude,
            longitude=longitude,
            timezone=timezone or "auto",
        )
    else:
        query_to_resolve = (location_query or "").strip() or "London"
        resolved_location = await _resolve_location(None, query_to_resolve)

    if normalized_section == "dashboard":
        return {
            "section": "dashboard",
            "title": "Dashboard",
            "location": _serialize_location(resolved_location),
            "message": "Dashboard data is available via /api/forecast/personalized.",
        }

    if normalized_section == "statistics":
        try:
            (weather_payload, _), aqi_payload, weather_alerts = await asyncio.gather(
                weather_client.fetch_weather_ensemble(
                    latitude=resolved_location.latitude,
                    longitude=resolved_location.longitude,
                    timezone=resolved_location.timezone,
                ),
                weather_client.fetch_air_quality(
                    latitude=resolved_location.latitude,
                    longitude=resolved_location.longitude,
                    timezone=resolved_location.timezone,
                ),
                weather_client.fetch_weather_alerts(
                    latitude=resolved_location.latitude,
                    longitude=resolved_location.longitude,
                ),
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Sidebar statistics provider error: {exc}") from exc
        return _build_statistics_slide(
            location=resolved_location,
            weather_payload=weather_payload,
            aqi_payload=aqi_payload,
            weather_alerts=weather_alerts,
        )

    if normalized_section == "map":
        try:
            weather_payload, _ = await weather_client.fetch_weather_ensemble(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
                timezone=resolved_location.timezone,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Sidebar map provider error: {exc}") from exc

        nearby_points: list[dict] = []
        if resolved_location.name:
            try:
                nearby_points = await weather_client.geocode(query=resolved_location.name)
            except httpx.HTTPError:
                nearby_points = []

        return _build_map_slide(
            location=resolved_location,
            weather_payload=weather_payload,
            nearby_points=nearby_points,
        )

    if normalized_section == "calendar":
        try:
            weather_payload, _ = await weather_client.fetch_weather_ensemble(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
                timezone=resolved_location.timezone,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Sidebar calendar provider error: {exc}") from exc
        return _build_calendar_slide(location=resolved_location, weather_payload=weather_payload)

    try:
        weather_payload, _ = await weather_client.fetch_weather_ensemble(
            latitude=resolved_location.latitude,
            longitude=resolved_location.longitude,
            timezone=resolved_location.timezone,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Sidebar setting provider error: {exc}") from exc

    return _build_setting_slide(location=resolved_location, weather_payload=weather_payload)


@app.get("/api/notifications")
async def notifications(
    location_query: str | None = Query(default=None, max_length=80),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    timezone_name: str = Query(default="auto", alias="timezone"),
    limit: int = Query(default=8, ge=1, le=20),
) -> dict:
    if latitude is not None and longitude is not None:
        resolved_location = Coordinates(
            name=location_query or "Selected location",
            latitude=latitude,
            longitude=longitude,
            timezone=timezone_name or "auto",
        )
    else:
        query_to_resolve = (location_query or "").strip() or "London"
        resolved_location = await _resolve_location(None, query_to_resolve)

    try:
        (weather_payload, forecast_diagnostics), weather_alerts = await asyncio.gather(
            weather_client.fetch_weather_ensemble(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
                timezone=resolved_location.timezone,
            ),
            weather_client.fetch_weather_alerts(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
            ),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Notification provider error: {exc}") from exc

    notification_items = _build_notification_items(
        location=resolved_location,
        weather_payload=weather_payload,
        weather_alerts=weather_alerts,
        forecast_diagnostics=forecast_diagnostics,
    )[:limit]

    return {
        "location": _serialize_location(resolved_location),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "unread_count": sum(1 for item in notification_items if item.get("severity") in {"high", "medium"}),
        "items": notification_items,
    }


@app.get("/api/voice/explain")
async def voice_explain(
    location_query: str | None = Query(default=None, max_length=80),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    timezone_name: str = Query(default="auto", alias="timezone"),
    language: str = Query(default="en", max_length=10),
    horizon_days: int = Query(default=5, ge=3, le=7),
) -> dict:
    normalized_language = _normalize_voice_language(language)

    if latitude is not None and longitude is not None:
        resolved_location = Coordinates(
            name=location_query or "Selected location",
            latitude=latitude,
            longitude=longitude,
            timezone=timezone_name or "auto",
        )
    else:
        query_to_resolve = (location_query or "").strip() or "London"
        resolved_location = await _resolve_location(None, query_to_resolve)

    try:
        (weather_payload, forecast_diagnostics), weather_alerts = await asyncio.gather(
            weather_client.fetch_weather_ensemble(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
                timezone=resolved_location.timezone,
            ),
            weather_client.fetch_weather_alerts(
                latitude=resolved_location.latitude,
                longitude=resolved_location.longitude,
            ),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Voice explanation provider error: {exc}") from exc

    voice_payload = _build_voice_explanation(
        location=resolved_location,
        weather_payload=weather_payload,
        weather_alerts=weather_alerts,
        forecast_diagnostics=forecast_diagnostics,
        language=normalized_language,
        horizon_days=horizon_days,
    )

    return {
        "location": _serialize_location(resolved_location),
        "language": normalized_language,
        "language_name": SUPPORTED_VOICE_LANGUAGES[normalized_language]["name"],
        "speech_language": SUPPORTED_VOICE_LANGUAGES[normalized_language]["speech_language"],
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "reasoning_prompt_template": voice_payload["reasoning_prompt_template"],
        "factors": voice_payload["factors"],
        "sections": voice_payload["sections"],
        "transcript": voice_payload["transcript"],
        "alerts_count": len(weather_alerts),
        "model_confidence": forecast_diagnostics.get("confidence_score"),
        "available_languages": [
            {"code": code, "name": data["name"], "speech_language": data["speech_language"]}
            for code, data in SUPPORTED_VOICE_LANGUAGES.items()
        ],
    }


async def _resolve_location(location: Coordinates | None, location_query: str | None) -> Coordinates:
    if location is not None:
        return location

    try:
        geo_results = await weather_client.geocode(location_query or "")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding provider error: {exc}") from exc

    if not geo_results:
        raise HTTPException(status_code=404, detail="Location not found.")

    first = geo_results[0]
    return Coordinates(
        name=first.get("name"),
        latitude=first.get("latitude"),
        longitude=first.get("longitude"),
        timezone=first.get("timezone", "auto"),
    )


def _serialize_location(location: Coordinates) -> dict:
    return {
        "name": location.name,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": location.timezone,
    }


def _number_at(values: list | tuple, idx: int) -> float | None:
    if idx >= len(values):
        return None
    value = values[idx]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _hour_label(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).strftime("%H:%M")
    except ValueError:
        return stamp[-5:]


def _safe_average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _max_number_with_index(values: list | tuple) -> tuple[float | None, int]:
    max_value: float | None = None
    max_index = -1
    for idx, value in enumerate(values):
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if max_value is None or numeric > max_value:
            max_value = numeric
            max_index = idx
    return max_value, max_index


def _normalize_severity(value: str | None) -> str:
    normalized = str(value or "").lower().strip()
    if normalized in {"extreme", "severe", "high"}:
        return "high"
    if normalized in {"moderate", "medium"}:
        return "medium"
    return "low"


def _build_notification_items(
    *,
    location: Coordinates,
    weather_payload: dict,
    weather_alerts: list[dict],
    forecast_diagnostics: dict,
) -> list[dict]:
    items: list[dict] = []
    now_utc = datetime.now(tz=timezone.utc).isoformat()

    for idx, alert in enumerate(weather_alerts[:4]):
        event = alert.get("event") or "Weather Alert"
        severity = _normalize_severity(alert.get("severity"))
        headline = str(alert.get("headline") or "").strip()
        if not headline:
            headline = f"{event} is active near {location.name or 'your selected location'}."
        expires = alert.get("expires")
        expiry_note = f" Expires: {expires}." if expires else ""
        items.append(
            {
                "id": f"alert-{idx}",
                "type": "alert",
                "severity": severity,
                "title": event,
                "message": f"{headline}{expiry_note}",
                "timestamp_utc": alert.get("effective") or now_utc,
            }
        )

    hourly = weather_payload.get("hourly", {})
    current = weather_payload.get("current", {})
    hourly_times = hourly.get("time", [])
    rain_values = hourly.get("precipitation_probability", [])[:12]
    wind_values = hourly.get("wind_speed_10m", [])[:12]
    uv_values = hourly.get("uv_index", [])[:12]

    rain_peak, rain_idx = _max_number_with_index(rain_values)
    if rain_peak is not None and rain_peak >= 45:
        rain_time = _hour_label(hourly_times[rain_idx]) if 0 <= rain_idx < len(hourly_times) else "--:--"
        items.append(
            {
                "id": "rain-peak",
                "type": "rain",
                "severity": "high" if rain_peak >= 70 else "medium",
                "title": "Rain Risk",
                "message": f"Rain probability may reach {round(rain_peak)}% around {rain_time}.",
                "timestamp_utc": now_utc,
            }
        )

    wind_peak, wind_idx = _max_number_with_index(wind_values)
    if wind_peak is not None and wind_peak >= 28:
        wind_time = _hour_label(hourly_times[wind_idx]) if 0 <= wind_idx < len(hourly_times) else "--:--"
        items.append(
            {
                "id": "wind-peak",
                "type": "wind",
                "severity": "high" if wind_peak >= 45 else "medium",
                "title": "Wind Speed Advisory",
                "message": f"Wind may peak near {round(wind_peak)} km/h around {wind_time}.",
                "timestamp_utc": now_utc,
            }
        )

    uv_peak, uv_idx = _max_number_with_index(uv_values)
    if uv_peak is not None and uv_peak >= 7:
        uv_time = _hour_label(hourly_times[uv_idx]) if 0 <= uv_idx < len(hourly_times) else "--:--"
        items.append(
            {
                "id": "uv-peak",
                "type": "uv",
                "severity": "high" if uv_peak >= 9 else "medium",
                "title": "UV Alert",
                "message": f"UV index may rise to {round(uv_peak, 1)} around {uv_time}.",
                "timestamp_utc": now_utc,
            }
        )

    current_temp = current.get("temperature_2m")
    if isinstance(current_temp, (int, float)):
        if current_temp >= 35:
            items.append(
                {
                    "id": "heat-alert",
                    "type": "temperature",
                    "severity": "high",
                    "title": "Heat Alert",
                    "message": f"Current temperature is {round(float(current_temp), 1)}C. Hydration reminders recommended.",
                    "timestamp_utc": now_utc,
                }
            )
        elif current_temp <= 2:
            items.append(
                {
                    "id": "cold-alert",
                    "type": "temperature",
                    "severity": "medium",
                    "title": "Cold Conditions",
                    "message": f"Current temperature is {round(float(current_temp), 1)}C. Layered clothing is recommended.",
                    "timestamp_utc": now_utc,
                }
            )

    confidence_score = forecast_diagnostics.get("confidence_score")
    if isinstance(confidence_score, (int, float)):
        confidence = int(round(float(confidence_score)))
        if confidence <= 55:
            items.append(
                {
                    "id": "confidence-low",
                    "type": "model",
                    "severity": "medium",
                    "title": "Forecast Confidence Moderate",
                    "message": f"Model confidence is {confidence}/100. Check updates more frequently today.",
                    "timestamp_utc": now_utc,
                }
            )
        elif confidence >= 80:
            items.append(
                {
                    "id": "confidence-high",
                    "type": "model",
                    "severity": "low",
                    "title": "Forecast Confidence High",
                    "message": f"Model confidence is {confidence}/100 for this location.",
                    "timestamp_utc": now_utc,
                }
            )

    if not items:
        items.append(
            {
                "id": "all-clear",
                "type": "status",
                "severity": "low",
                "title": "No Critical Weather Risk",
                "message": "No major alerts found in the next forecast window.",
                "timestamp_utc": now_utc,
            }
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: severity_rank.get(str(item.get("severity")).lower(), 2))
    return items


def _normalize_voice_language(language: str | None) -> str:
    normalized = str(language or "en").strip().lower().replace("_", "-")
    short = normalized.split("-")[0]
    if short in SUPPORTED_VOICE_LANGUAGES:
        return short
    return "en"


def _average_numeric(values: list | tuple, limit: int = 12) -> float | None:
    subset = values[:limit] if limit > 0 else values
    numbers = [float(value) for value in subset if isinstance(value, (int, float))]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 1)


def _max_numeric(values: list | tuple, limit: int = 24) -> float | None:
    subset = values[:limit] if limit > 0 else values
    numbers = [float(value) for value in subset if isinstance(value, (int, float))]
    if not numbers:
        return None
    return round(max(numbers), 1)


def _fmt_number(value: float | None, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    if abs(value - int(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def _day_label(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%a")
    except ValueError:
        return str(value)[:10]


def _pressure_bucket(pressure_msl: float | None) -> str:
    if pressure_msl is None:
        return "unknown"
    if pressure_msl <= 1008:
        return "low"
    if pressure_msl >= 1018:
        return "high"
    return "normal"


def _humidity_bucket(humidity: float | None) -> str:
    if humidity is None:
        return "unknown"
    if humidity >= 75:
        return "high"
    if humidity <= 40:
        return "low"
    return "moderate"


def _wind_bucket(wind_kph: float | None) -> str:
    if wind_kph is None:
        return "unknown"
    if wind_kph >= 28:
        return "strong"
    if wind_kph <= 10:
        return "light"
    return "moderate"


def _front_bucket(
    *,
    pressure_bucket: str,
    rain_peak: float | None,
    temp_now: float | None,
    tomorrow_mid_temp: float | None,
) -> str:
    if pressure_bucket == "low" and rain_peak is not None and rain_peak >= 55:
        return "wet_front"
    if temp_now is not None and tomorrow_mid_temp is not None:
        shift = tomorrow_mid_temp - temp_now
        if shift <= -3:
            return "cool_shift"
        if shift >= 3 and pressure_bucket == "high":
            return "warm_stable"
    return "mixed"


def _build_reasoning_prompt(
    *,
    temperature: float | None,
    humidity: float | None,
    pressure: float | None,
    wind: float | None,
    cloud: float | None,
    rain_probability: float | None,
) -> str:
    return WEATHER_REASON_PROMPT_TEMPLATE.format(
        temp=f"{_fmt_number(temperature)}C",
        humidity=f"{_fmt_number(humidity)}%",
        pressure=f"{_fmt_number(pressure)} hPa",
        wind=f"{_fmt_number(wind)} km/h",
        cloud=f"{_fmt_number(cloud)}%",
        rain_prob=f"{_fmt_number(rain_probability)}%",
    )


def _localize_weather_label(label: str, language: str) -> str:
    condition_labels = {
        "en": {},
        "es": {
            "Clear sky": "Cielo despejado",
            "Mainly clear": "Mayormente despejado",
            "Partly cloudy": "Parcialmente nublado",
            "Overcast": "Nublado",
            "Fog": "Niebla",
            "Light drizzle": "Llovizna ligera",
            "Moderate drizzle": "Llovizna moderada",
            "Dense drizzle": "Llovizna intensa",
            "Slight rain": "Lluvia ligera",
            "Moderate rain": "Lluvia moderada",
            "Heavy rain": "Lluvia intensa",
            "Slight snow": "Nieve ligera",
            "Moderate snow": "Nieve moderada",
            "Heavy snow": "Nieve intensa",
            "Rain showers": "Chubascos",
            "Moderate rain showers": "Chubascos moderados",
            "Violent rain showers": "Chubascos fuertes",
            "Thunderstorm": "Tormenta",
            "Thunderstorm with hail": "Tormenta con granizo",
            "Heavy thunderstorm with hail": "Tormenta fuerte con granizo",
            "Unknown": "Condición desconocida",
        },
        "fr": {
            "Clear sky": "Ciel dégagé",
            "Mainly clear": "Plutôt dégagé",
            "Partly cloudy": "Partiellement nuageux",
            "Overcast": "Couvert",
            "Fog": "Brouillard",
            "Light drizzle": "Bruine faible",
            "Moderate drizzle": "Bruine modérée",
            "Dense drizzle": "Bruine dense",
            "Slight rain": "Pluie faible",
            "Moderate rain": "Pluie modérée",
            "Heavy rain": "Forte pluie",
            "Slight snow": "Neige faible",
            "Moderate snow": "Neige modérée",
            "Heavy snow": "Forte neige",
            "Rain showers": "Averses",
            "Moderate rain showers": "Averses modérées",
            "Violent rain showers": "Averses fortes",
            "Thunderstorm": "Orage",
            "Thunderstorm with hail": "Orage avec grêle",
            "Heavy thunderstorm with hail": "Fort orage avec grêle",
            "Unknown": "Condition inconnue",
        },
        "ta": {
            "Clear sky": "தெளிவான வானம்",
            "Mainly clear": "பெரும்பாலும் தெளிவு",
            "Partly cloudy": "பகுதியளவு மேகமூட்டம்",
            "Overcast": "முழு மேகமூட்டம்",
            "Fog": "மூடுபனி",
            "Light drizzle": "லேசான தூறல்",
            "Moderate drizzle": "மிதமான தூறல்",
            "Dense drizzle": "அதிக தூறல்",
            "Slight rain": "லேசான மழை",
            "Moderate rain": "மிதமான மழை",
            "Heavy rain": "கனமழை",
            "Slight snow": "லேசான பனிப்பொழிவு",
            "Moderate snow": "மிதமான பனிப்பொழிவு",
            "Heavy snow": "கனமான பனிப்பொழிவு",
            "Rain showers": "சாரல் மழை",
            "Moderate rain showers": "மிதமான சாரல் மழை",
            "Violent rain showers": "பலமான சாரல் மழை",
            "Thunderstorm": "இடி மின்னல் மழை",
            "Thunderstorm with hail": "கனிகல் உடன் இடி மழை",
            "Heavy thunderstorm with hail": "கனமான கனிகல் இடி மழை",
            "Unknown": "தகவல் இல்லை",
        },
        "hi": {
            "Clear sky": "आसमान साफ",
            "Mainly clear": "ज्यादातर साफ",
            "Partly cloudy": "आंशिक बादल",
            "Overcast": "घना बादल",
            "Fog": "कोहरा",
            "Light drizzle": "हल्की बूंदाबांदी",
            "Moderate drizzle": "मध्यम बूंदाबांदी",
            "Dense drizzle": "तेज बूंदाबांदी",
            "Slight rain": "हल्की बारिश",
            "Moderate rain": "मध्यम बारिश",
            "Heavy rain": "तेज बारिश",
            "Slight snow": "हल्की बर्फबारी",
            "Moderate snow": "मध्यम बर्फबारी",
            "Heavy snow": "भारी बर्फबारी",
            "Rain showers": "बारिश की फुहारें",
            "Moderate rain showers": "मध्यम बारिश की फुहारें",
            "Violent rain showers": "तेज बारिश की फुहारें",
            "Thunderstorm": "आंधी-तूफान",
            "Thunderstorm with hail": "ओलावृष्टि के साथ तूफान",
            "Heavy thunderstorm with hail": "भारी ओलावृष्टि के साथ तूफान",
            "Unknown": "स्थिति अज्ञात",
        },
    }
    language_map = condition_labels.get(language, {})
    return language_map.get(label, label)


def _compose_cause(language: str, *, pressure_key: str, humidity_key: str, wind_key: str, front_key: str) -> str:
    cause_text = {
        "en": {
            "pressure": {
                "low": "low pressure is helping cloud growth",
                "high": "higher pressure is stabilizing the atmosphere",
                "normal": "pressure is near seasonal normal",
                "unknown": "pressure data is limited",
            },
            "humidity": {
                "high": "high humidity is supplying extra moisture",
                "moderate": "moderate humidity supports patchy cloud bands",
                "low": "lower humidity is reducing rain formation",
                "unknown": "humidity data is partial",
            },
            "wind": {
                "strong": "strong wind is transporting weather systems quickly",
                "moderate": "moderate wind is moving local cloud fields",
                "light": "light wind means local heating effects are stronger",
                "unknown": "wind signal is limited",
            },
            "front": {
                "wet_front": "a weak frontal wave is likely passing through.",
                "cool_shift": "a cooler air mass is moving in for tomorrow.",
                "warm_stable": "warmer and more stable air is building overhead.",
                "mixed": "no dominant front signature is visible right now.",
            },
        },
        "es": {
            "pressure": {
                "low": "baja presion favorece formacion de nubes",
                "high": "alta presion mantiene ambiente mas estable",
                "normal": "la presion esta cerca de lo normal",
                "unknown": "datos de presion son limitados",
            },
            "humidity": {
                "high": "humedad alta aporta vapor para lluvia",
                "moderate": "humedad moderada mantiene nubes variables",
                "low": "humedad baja reduce soporte para lluvia",
                "unknown": "datos de humedad son parciales",
            },
            "wind": {
                "strong": "viento fuerte mueve sistemas nubosos rapido",
                "moderate": "viento moderado desplaza nubes locales",
                "light": "viento suave deja mas efecto local",
                "unknown": "senal de viento limitada",
            },
            "front": {
                "wet_front": "un frente debil y humedo puede estar pasando.",
                "cool_shift": "una masa de aire mas fria llega para manana.",
                "warm_stable": "aire mas calido y estable domina la zona.",
                "mixed": "no se detecta un frente dominante ahora.",
            },
        },
        "fr": {
            "pressure": {
                "low": "la basse pression favorise la formation nuageuse",
                "high": "la haute pression stabilise l atmosphere",
                "normal": "la pression reste proche de la normale",
                "unknown": "les donnees de pression sont limitees",
            },
            "humidity": {
                "high": "une humidite elevee apporte plus de vapeur",
                "moderate": "une humidite moderee maintient des nuages variables",
                "low": "une humidite basse limite la pluie",
                "unknown": "les donnees d humidite sont partielles",
            },
            "wind": {
                "strong": "un vent fort transporte vite les systemes",
                "moderate": "un vent modere deplace les nuages locaux",
                "light": "un vent faible renforce les effets locaux",
                "unknown": "signal de vent limite",
            },
            "front": {
                "wet_front": "une faible onde frontale humide peut passer.",
                "cool_shift": "une masse d air plus froide arrive demain.",
                "warm_stable": "un air plus chaud et stable s installe.",
                "mixed": "pas de front dominant clair actuellement.",
            },
        },
        "ta": {
            "pressure": {
                "low": "குறைந்த காற்றழுத்தம் மேக உருவாக்கத்தை அதிகரிக்கிறது",
                "high": "அதிக காற்றழுத்தம் வளிமண்டலத்தை நிலைப்படுத்துகிறது",
                "normal": "காற்றழுத்தம் சாதாரண நிலைக்கு அருகில் உள்ளது",
                "unknown": "காற்றழுத்த தரவு குறைவாக உள்ளது",
            },
            "humidity": {
                "high": "அதிக ஈரப்பதம் மழைக்குத் தேவையான ஈரத்தை வழங்குகிறது",
                "moderate": "மிதமான ஈரப்பதம் மாறும் மேக அமைப்பை ஆதரிக்கிறது",
                "low": "குறைந்த ஈரப்பதம் மழை உருவாகும் வாய்ப்பை குறைக்கிறது",
                "unknown": "ஈரப்பத தரவு முழுமையாக கிடைக்கவில்லை",
            },
            "wind": {
                "strong": "வலுவான காற்று வானிலை அமைப்புகளை வேகமாக நகர்த்துகிறது",
                "moderate": "மிதமான காற்று உள்ளூர் மேகங்களை நகர்த்துகிறது",
                "light": "மெதுவான காற்றில் உள்ளூர் வெப்ப விளைவு அதிகமாக இருக்கும்",
                "unknown": "காற்று தரவு குறைவாக உள்ளது",
            },
            "front": {
                "wet_front": "பலவீனமான ஈர முனைப்பகம் கடந்து செல்லும் சாத்தியம் உள்ளது.",
                "cool_shift": "நாளைக்கு குளிர்ந்த காற்று பகுதி நுழைய வாய்ப்பு உள்ளது.",
                "warm_stable": "சற்று சூடான மற்றும் நிலையான காற்று நிலை உருவாகிறது.",
                "mixed": "தற்போது தெளிவான முனைப்பக குறியீடு இல்லை.",
            },
        },
        "hi": {
            "pressure": {
                "low": "कम दबाव बादल बनने की प्रक्रिया को बढ़ा रहा है",
                "high": "उच्च दबाव वातावरण को अधिक स्थिर बना रहा है",
                "normal": "दबाव सामान्य सीमा के आसपास है",
                "unknown": "दबाव का डेटा सीमित है",
            },
            "humidity": {
                "high": "उच्च नमी वर्षा के लिए अतिरिक्त नमी उपलब्ध करा रही है",
                "moderate": "मध्यम नमी बदलते बादलों को समर्थन दे रही है",
                "low": "कम नमी वर्षा बनने की संभावना घटाती है",
                "unknown": "नमी का डेटा आंशिक है",
            },
            "wind": {
                "strong": "तेज़ हवा मौसम तंत्र को तेजी से आगे बढ़ा रही है",
                "moderate": "मध्यम हवा स्थानीय बादलों को स्थानांतरित कर रही है",
                "light": "हल्की हवा में स्थानीय गर्मी का प्रभाव बढ़ जाता है",
                "unknown": "हवा का संकेत सीमित है",
            },
            "front": {
                "wet_front": "कमज़ोर नम फ्रंट के गुजरने की संभावना है।",
                "cool_shift": "कल ठंडी हवा का प्रभाव बढ़ सकता है।",
                "warm_stable": "गर्म और स्थिर हवा का प्रभाव बन रहा है।",
                "mixed": "अभी कोई प्रमुख फ्रंट संकेत स्पष्ट नहीं है।",
            },
        },
    }

    language_pack = cause_text.get(language) or cause_text["en"]
    pressure_text = language_pack["pressure"].get(pressure_key, language_pack["pressure"]["unknown"])
    humidity_text = language_pack["humidity"].get(humidity_key, language_pack["humidity"]["unknown"])
    wind_text = language_pack["wind"].get(wind_key, language_pack["wind"]["unknown"])
    front_text = language_pack["front"].get(front_key, language_pack["front"]["mixed"])
    if language == "es":
        return f"{pressure_text}, {humidity_text}, {wind_text} y {front_text}"
    if language == "fr":
        return f"{pressure_text}, {humidity_text}, {wind_text} et {front_text}"
    if language == "ta":
        return f"{pressure_text}, {humidity_text}, {wind_text}, மேலும் {front_text}"
    if language == "hi":
        return f"{pressure_text}, {humidity_text}, {wind_text} और {front_text}"
    return f"{pressure_text}, {humidity_text}, {wind_text}, and {front_text}"


def _build_voice_explanation(
    *,
    location: Coordinates,
    weather_payload: dict,
    weather_alerts: list[dict],
    forecast_diagnostics: dict,
    language: str,
    horizon_days: int,
) -> dict:
    current = weather_payload.get("current", {})
    hourly = weather_payload.get("hourly", {})
    daily = weather_payload.get("daily", {})

    current_condition = _localize_weather_label(weather_code_to_label(current.get("weather_code")), language)
    current_temp = _number_at([current.get("temperature_2m")], 0)
    current_humidity = _number_at([current.get("relative_humidity_2m")], 0)
    current_wind = _number_at([current.get("wind_speed_10m")], 0)

    pressure_avg = _average_numeric(hourly.get("pressure_msl", []), limit=12)
    cloud_avg = _average_numeric(hourly.get("cloud_cover", []), limit=12)
    rain_peak = _max_numeric(hourly.get("precipitation_probability", []), limit=24)
    wind_peak = _max_numeric(hourly.get("wind_speed_10m", []), limit=24)
    humidity_avg = _average_numeric(hourly.get("relative_humidity_2m", []), limit=12)
    humidity_for_reason = current_humidity if current_humidity is not None else humidity_avg

    tomorrow_index = 1 if len(daily.get("time", [])) > 1 else 0
    tomorrow_label = _localize_weather_label(
        weather_code_to_label(_number_at(daily.get("weather_code", []), tomorrow_index)),
        language,
    )
    tomorrow_max = _number_at(daily.get("temperature_2m_max", []), tomorrow_index)
    tomorrow_min = _number_at(daily.get("temperature_2m_min", []), tomorrow_index)
    tomorrow_rain = _number_at(daily.get("precipitation_probability_max", []), tomorrow_index)
    tomorrow_mid = None
    if tomorrow_max is not None and tomorrow_min is not None:
        tomorrow_mid = round((tomorrow_max + tomorrow_min) / 2, 1)

    pressure_key = _pressure_bucket(pressure_avg)
    humidity_key = _humidity_bucket(humidity_for_reason)
    wind_key = _wind_bucket(wind_peak if wind_peak is not None else current_wind)
    front_key = _front_bucket(
        pressure_bucket=pressure_key,
        rain_peak=rain_peak,
        temp_now=current_temp,
        tomorrow_mid_temp=tomorrow_mid,
    )
    cause_text = _compose_cause(
        language=language,
        pressure_key=pressure_key,
        humidity_key=humidity_key,
        wind_key=wind_key,
        front_key=front_key,
    )

    start_idx = 1 if len(daily.get("time", [])) > 1 else 0
    end_idx = min(len(daily.get("time", [])), start_idx + max(3, horizon_days))
    trend_parts: list[str] = []
    for idx in range(start_idx, end_idx):
        date_value = daily.get("time", [None])[idx]
        if date_value is None:
            continue
        trend_condition = _localize_weather_label(
            weather_code_to_label(_number_at(daily.get("weather_code", []), idx)),
            language,
        )
        trend_max = _fmt_number(_number_at(daily.get("temperature_2m_max", []), idx))
        trend_min = _fmt_number(_number_at(daily.get("temperature_2m_min", []), idx))
        trend_parts.append(f"{_day_label(str(date_value))} {trend_condition} {trend_min}/{trend_max}C")
    trend_fallback = {
        "en": "Limited multi-day trend data.",
        "es": "Datos de tendencia de varios días limitados.",
        "fr": "Données de tendance sur plusieurs jours limitées.",
        "ta": "பல நாள் போக்கு தரவு குறைவாக உள்ளது.",
        "hi": "कई दिनों के रुझान का डेटा सीमित है।",
    }
    trend_summary = "; ".join(trend_parts) if trend_parts else trend_fallback.get(language, trend_fallback["en"])

    alert_events = [str(alert.get("event") or "").strip() for alert in weather_alerts if alert.get("event")]
    alert_summary = ", ".join(alert_events[:2]) if alert_events else ""
    model_confidence = forecast_diagnostics.get("confidence_score")

    templates = {
        "en": {
            "current": "Currently in {location}, conditions are {condition} with {temp}C, humidity near {humidity}% and wind around {wind} km/h.",
            "reason": "Why this is happening: {cause}",
            "tomorrow": "Tomorrow, expect {tomorrow_condition} with temperatures from {tomorrow_min}C to {tomorrow_max}C and rain chance near {tomorrow_rain}%.",
            "trend": "Next days trend: {trend_summary}.",
            "alerts": "Active alert note: {alert_summary}.",
            "labels": {
                "current": "Current Weather",
                "reason": "Why It Happens",
                "tomorrow": "Tomorrow Forecast",
                "trend": "Future Trend",
                "alerts": "Risk Alerts",
            },
        },
        "es": {
            "current": "Ahora en {location}, el tiempo está {condition}, con {temp}°C, humedad cerca de {humidity}% y viento de {wind} km/h.",
            "reason": "Por qué ocurre: {cause}",
            "tomorrow": "Mañana se espera {tomorrow_condition}, con temperatura entre {tomorrow_min}°C y {tomorrow_max}°C, y lluvia cerca de {tomorrow_rain}%.",
            "trend": "Tendencia de los próximos días: {trend_summary}.",
            "alerts": "Alerta activa: {alert_summary}.",
            "labels": {
                "current": "Clima Actual",
                "reason": "Motivo Atmosférico",
                "tomorrow": "Pronóstico Mañana",
                "trend": "Tendencia Futura",
                "alerts": "Alertas",
            },
        },
        "fr": {
            "current": "Actuellement à {location}, le temps est {condition}, avec {temp}°C, humidité vers {humidity}% et vent autour de {wind} km/h.",
            "reason": "Pourquoi: {cause}",
            "tomorrow": "Demain, conditions {tomorrow_condition}, avec température de {tomorrow_min}°C à {tomorrow_max}°C et pluie vers {tomorrow_rain}%.",
            "trend": "Tendance des prochains jours: {trend_summary}.",
            "alerts": "Alerte active: {alert_summary}.",
            "labels": {
                "current": "Météo Actuelle",
                "reason": "Cause Atmosphérique",
                "tomorrow": "Prévision Demain",
                "trend": "Tendance Future",
                "alerts": "Alertes",
            },
        },
        "ta": {
            "current": "இப்போது {location} பகுதியில் {condition} நிலை உள்ளது. வெப்பநிலை {temp}°C, ஈரப்பதம் {humidity}% மற்றும் காற்று வேகம் {wind} km/h.",
            "reason": "இது ஏற்படும் காரணம்: {cause}",
            "tomorrow": "நாளை {tomorrow_condition} நிலை இருக்கலாம். வெப்பநிலை {tomorrow_min}°C முதல் {tomorrow_max}°C வரை, மழை வாய்ப்பு {tomorrow_rain}%.",
            "trend": "அடுத்த நாட்களின் போக்கு: {trend_summary}.",
            "alerts": "செயலில் உள்ள எச்சரிக்கை: {alert_summary}.",
            "labels": {
                "current": "தற்போதைய வானிலை",
                "reason": "அறிவியல் காரணம்",
                "tomorrow": "நாளைய முன்னறிவிப்பு",
                "trend": "அடுத்த நாட்கள் போக்கு",
                "alerts": "அபாய எச்சரிக்கை",
            },
        },
        "hi": {
            "current": "अभी {location} में मौसम {condition} है। तापमान {temp}°C, नमी {humidity}% और हवा की गति {wind} km/h है।",
            "reason": "यह क्यों हो रहा है: {cause}",
            "tomorrow": "कल {tomorrow_condition} रह सकता है। तापमान {tomorrow_min}°C से {tomorrow_max}°C तक और बारिश की संभावना {tomorrow_rain}% है।",
            "trend": "अगले दिनों का रुझान: {trend_summary}.",
            "alerts": "सक्रिय चेतावनी: {alert_summary}.",
            "labels": {
                "current": "वर्तमान मौसम",
                "reason": "वैज्ञानिक कारण",
                "tomorrow": "कल का पूर्वानुमान",
                "trend": "आगे का रुझान",
                "alerts": "जोखिम चेतावनी",
            },
        },
    }
    active_template = templates.get(language) or templates["en"]

    template_data = {
        "location": location.name or "Selected location",
        "condition": current_condition,
        "temp": _fmt_number(current_temp),
        "humidity": _fmt_number(humidity_for_reason),
        "wind": _fmt_number(current_wind),
        "cause": cause_text,
        "tomorrow_condition": tomorrow_label,
        "tomorrow_min": _fmt_number(tomorrow_min),
        "tomorrow_max": _fmt_number(tomorrow_max),
        "tomorrow_rain": _fmt_number(tomorrow_rain),
        "trend_summary": trend_summary,
        "alert_summary": alert_summary,
    }

    sections: list[dict] = [
        {
            "key": "current",
            "label": active_template["labels"]["current"],
            "text": active_template["current"].format(**template_data),
        },
        {
            "key": "reason",
            "label": active_template["labels"]["reason"],
            "text": active_template["reason"].format(**template_data),
        },
        {
            "key": "tomorrow",
            "label": active_template["labels"]["tomorrow"],
            "text": active_template["tomorrow"].format(**template_data),
        },
        {
            "key": "trend",
            "label": active_template["labels"]["trend"],
            "text": active_template["trend"].format(**template_data),
        },
    ]
    if alert_summary:
        sections.append(
            {
                "key": "alerts",
                "label": active_template["labels"]["alerts"],
                "text": active_template["alerts"].format(**template_data),
            }
        )

    transcript = " ".join(item["text"] for item in sections)
    return {
        "sections": sections,
        "transcript": transcript,
        "reasoning_prompt_template": _build_reasoning_prompt(
            temperature=current_temp,
            humidity=humidity_for_reason,
            pressure=pressure_avg,
            wind=current_wind,
            cloud=cloud_avg,
            rain_probability=rain_peak,
        ),
        "factors": {
            "temperature_c": current_temp,
            "humidity_percent": humidity_for_reason,
            "pressure_msl_hpa": pressure_avg,
            "wind_kph": current_wind,
            "cloud_cover_percent": cloud_avg,
            "rain_probability_peak_percent": rain_peak,
            "wind_peak_kph": wind_peak,
            "condition": current_condition,
            "model_confidence": model_confidence,
        },
    }


def _build_statistics_slide(
    *,
    location: Coordinates,
    weather_payload: dict,
    aqi_payload: dict,
    weather_alerts: list[dict],
) -> dict:
    hourly = weather_payload.get("hourly", {})
    hourly_times = hourly.get("time", [])

    temperatures = [float(value) for value in hourly.get("temperature_2m", []) if isinstance(value, (int, float))]
    winds = [float(value) for value in hourly.get("wind_speed_10m", []) if isinstance(value, (int, float))]
    rains = [
        float(value)
        for value in hourly.get("precipitation_probability", [])
        if isinstance(value, (int, float))
    ]

    aqi_hourly = aqi_payload.get("hourly", {})
    aqi_values = [float(value) for value in aqi_hourly.get("us_aqi", []) if isinstance(value, (int, float))]
    aqi_index = {}
    for idx, stamp in enumerate(aqi_hourly.get("time", [])):
        aqi_value = _number_at(aqi_hourly.get("us_aqi", []), idx)
        if aqi_value is not None:
            aqi_index[stamp] = round(aqi_value, 1)

    trend: list[dict] = []
    for idx, stamp in enumerate(hourly_times[:12]):
        trend.append(
            {
                "time": _hour_label(stamp),
                "temperature_c": _number_at(hourly.get("temperature_2m", []), idx),
                "wind_kph": _number_at(hourly.get("wind_speed_10m", []), idx),
                "rain_probability": _number_at(hourly.get("precipitation_probability", []), idx),
                "aqi_us": aqi_index.get(stamp),
            }
        )

    severe_alerts = sum(
        1
        for alert in weather_alerts
        if str(alert.get("severity") or "").lower() in {"severe", "extreme"}
    )

    return {
        "section": "statistics",
        "title": "Weather Statistics",
        "location": _serialize_location(location),
        "overview": {
            "average_temp_c": _safe_average(temperatures),
            "average_wind_kph": _safe_average(winds),
            "max_rain_probability": round(max(rains), 1) if rains else None,
            "average_aqi_us": _safe_average(aqi_values),
            "severe_alert_count": severe_alerts,
        },
        "trend": trend,
        "alerts": weather_alerts[:5],
    }


def _build_map_slide(*, location: Coordinates, weather_payload: dict, nearby_points: list[dict]) -> dict:
    points: list[dict] = []
    seen: set[tuple] = set()

    for point in nearby_points:
        name = point.get("name")
        latitude = point.get("latitude")
        longitude = point.get("longitude")
        if name is None or latitude is None or longitude is None:
            continue
        key = (name, latitude, longitude)
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "name": name,
                "country": point.get("country"),
                "admin1": point.get("admin1"),
                "latitude": latitude,
                "longitude": longitude,
            }
        )
        if len(points) >= 8:
            break

    current = weather_payload.get("current", {})
    return {
        "section": "map",
        "title": "Location Map",
        "location": _serialize_location(location),
        "center": {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": location.timezone,
        },
        "current_weather": {
            "temperature_c": current.get("temperature_2m"),
            "wind_kph": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
        },
        "points": points,
        "note": "Map markers represent nearby matched places for your selected location.",
    }


def _build_calendar_slide(*, location: Coordinates, weather_payload: dict) -> dict:
    daily = weather_payload.get("daily", {})

    schedule: list[dict] = []
    for idx, date_value in enumerate(daily.get("time", [])[:10]):
        temp_max = _number_at(daily.get("temperature_2m_max", []), idx)
        temp_min = _number_at(daily.get("temperature_2m_min", []), idx)
        rain = _number_at(daily.get("precipitation_probability_max", []), idx)
        weather_code = _number_at(daily.get("weather_code", []), idx)
        weather_label = weather_code_to_label(int(weather_code) if weather_code is not None else None)

        risk_score = (rain or 0) * 0.7
        if temp_max is not None:
            risk_score += max(temp_max - 32, 0) * 1.8
        if weather_label.lower().find("rain") >= 0:
            risk_score += 8

        if risk_score >= 60:
            risk = "high"
            action = "Keep indoor backup plans for this day."
        elif risk_score >= 35:
            risk = "moderate"
            action = "Plan flexible outdoor timing."
        else:
            risk = "low"
            action = "Good day for outdoor tasks."

        schedule.append(
            {
                "date": date_value,
                "weather": weather_label,
                "temp_max_c": temp_max,
                "temp_min_c": temp_min,
                "rain_probability_max": rain,
                "risk": risk,
                "action": action,
            }
        )

    return {
        "section": "calendar",
        "title": "Weather Calendar",
        "location": _serialize_location(location),
        "schedule": schedule,
    }


def _build_setting_slide(*, location: Coordinates, weather_payload: dict) -> dict:
    current = weather_payload.get("current", {})
    temperature = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")

    recommendations = [
        "Keep geolocation enabled for faster local forecasts.",
        "Use UV-sensitive mode during high daylight periods.",
        "Review severe-weather alerts before commute planning.",
    ]
    if isinstance(temperature, (int, float)) and temperature >= 33:
        recommendations.append("High temperature detected. Enable heat safety reminders.")
    if isinstance(wind, (int, float)) and wind >= 25:
        recommendations.append("High wind expected. Keep travel plans flexible.")

    return {
        "section": "setting",
        "title": "Preferences & Settings",
        "location": _serialize_location(location),
        "settings": [
            {
                "key": "timezone",
                "label": "Timezone",
                "value": location.timezone,
                "editable": False,
            },
            {
                "key": "forecast_refresh_window",
                "label": "Forecast refresh window",
                "value": "Hourly",
                "editable": True,
            },
            {
                "key": "ml_default_epochs",
                "label": "ML default epochs",
                "value": settings.ml_default_epochs,
                "editable": True,
            },
            {
                "key": "ml_history_days",
                "label": "ML history days",
                "value": settings.ml_history_days,
                "editable": True,
            },
        ],
        "recommendations": recommendations,
    }

