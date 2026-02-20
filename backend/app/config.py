from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "NimbusIQ Weather API"
    app_version: str = "1.0.0"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"
    open_meteo_geo_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    open_meteo_reverse_geo_url: str = "https://geocoding-api.open-meteo.com/v1/reverse"
    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    nws_alerts_url: str = "https://api.weather.gov/alerts/active"
    ensemble_models: tuple[str, ...] = ("gfs_seamless", "ecmwf_ifs04", "icon_seamless")
    api_cache_ttl_seconds: int = 600
    api_retry_attempts: int = 2
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    ml_model_dir: str = str((Path(__file__).resolve().parents[1] / "models" / "skymind").as_posix())
    ml_database_path: str = str((Path(__file__).resolve().parents[1] / "data" / "skymind_ml.db").as_posix())
    ml_history_days: int = 180
    ml_default_epochs: int = 50
    ml_default_horizon_hours: int = 24
    ml_min_retrain_hours: int = 6
    request_timeout_seconds: float = 12.0
    frontend_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")


def get_settings() -> Settings:
    origins_raw = os.getenv("FRONTEND_ORIGINS", "").strip()
    models_raw = os.getenv("ENSEMBLE_MODELS", "").strip()
    cache_ttl_raw = os.getenv("API_CACHE_TTL_SECONDS", "").strip()
    retry_attempts_raw = os.getenv("API_RETRY_ATTEMPTS", "").strip()
    ml_model_dir_raw = os.getenv("ML_MODEL_DIR", "").strip()
    ml_database_path_raw = os.getenv("ML_DATABASE_PATH", "").strip()
    ml_history_days_raw = os.getenv("ML_HISTORY_DAYS", "").strip()
    ml_default_epochs_raw = os.getenv("ML_DEFAULT_EPOCHS", "").strip()
    ml_default_horizon_hours_raw = os.getenv("ML_DEFAULT_HORIZON_HOURS", "").strip()
    ml_min_retrain_hours_raw = os.getenv("ML_MIN_RETRAIN_HOURS", "").strip()

    parsed_origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip())
    parsed_models = tuple(item.strip() for item in models_raw.split(",") if item.strip())

    try:
        cache_ttl_seconds = int(cache_ttl_raw) if cache_ttl_raw else 600
    except ValueError:
        cache_ttl_seconds = 600

    try:
        retry_attempts = int(retry_attempts_raw) if retry_attempts_raw else 2
    except ValueError:
        retry_attempts = 2

    try:
        ml_history_days = int(ml_history_days_raw) if ml_history_days_raw else 180
    except ValueError:
        ml_history_days = 180

    try:
        ml_default_epochs = int(ml_default_epochs_raw) if ml_default_epochs_raw else 50
    except ValueError:
        ml_default_epochs = 50

    try:
        ml_default_horizon_hours = int(ml_default_horizon_hours_raw) if ml_default_horizon_hours_raw else 24
    except ValueError:
        ml_default_horizon_hours = 24

    try:
        ml_min_retrain_hours = int(ml_min_retrain_hours_raw) if ml_min_retrain_hours_raw else 6
    except ValueError:
        ml_min_retrain_hours = 6

    return Settings(
        frontend_origins=parsed_origins or Settings.frontend_origins,
        ensemble_models=parsed_models or Settings.ensemble_models,
        api_cache_ttl_seconds=max(60, cache_ttl_seconds),
        api_retry_attempts=max(0, retry_attempts),
        ml_model_dir=ml_model_dir_raw or Settings.ml_model_dir,
        ml_database_path=ml_database_path_raw or Settings.ml_database_path,
        ml_history_days=max(30, ml_history_days),
        ml_default_epochs=max(5, ml_default_epochs),
        ml_default_horizon_hours=max(1, ml_default_horizon_hours),
        ml_min_retrain_hours=max(1, ml_min_retrain_hours),
    )

