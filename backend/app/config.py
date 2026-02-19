from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "NimbusIQ Weather API"
    app_version: str = "1.0.0"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"
    open_meteo_geo_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    request_timeout_seconds: float = 12.0
    frontend_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")


def get_settings() -> Settings:
    origins_raw = os.getenv("FRONTEND_ORIGINS", "").strip()
    if not origins_raw:
        return Settings()
    parsed_origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip())
    if not parsed_origins:
        return Settings()
    return Settings(frontend_origins=parsed_origins)

