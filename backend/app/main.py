from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import Coordinates, PersonalizedForecastRequest
from app.services.personalizer import build_personalized_response
from app.services.weather_client import WeatherClient


settings = get_settings()
weather_client = WeatherClient(settings=settings)

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await weather_client.close()


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


@app.post("/api/forecast/personalized")
async def personalized_forecast(payload: PersonalizedForecastRequest) -> dict:
    location = payload.location

    if location is None:
        try:
            geo_results = await weather_client.geocode(payload.location_query or "")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Geocoding provider error: {exc}") from exc

        if not geo_results:
            raise HTTPException(status_code=404, detail="Location not found.")

        first = geo_results[0]
        location = Coordinates(
            name=first.get("name"),
            latitude=first.get("latitude"),
            longitude=first.get("longitude"),
            timezone=first.get("timezone", "auto"),
        )

    try:
        weather_payload = await weather_client.fetch_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
        )
        aqi_payload = await weather_client.fetch_air_quality(
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=location.timezone,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Weather provider error: {exc}") from exc

    return build_personalized_response(
        location=location,
        weather_payload=weather_payload,
        aqi_payload=aqi_payload,
        routine=payload.routine,
        preferences=payload.preferences,
    )

