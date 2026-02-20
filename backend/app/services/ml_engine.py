from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from app.config import Settings
from app.schemas import Coordinates

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential, load_model

    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


TRAINING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc TEXT NOT NULL,
    location_name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    history_days INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    best_model TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);
"""

PREDICTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc TEXT NOT NULL,
    location_name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    horizon_hours INTEGER NOT NULL,
    confidence_score REAL,
    anomaly_score REAL,
    storm_alert_score REAL,
    alert_level TEXT,
    summary_json TEXT NOT NULL
);
"""


@dataclass
class WeatherMLEngine:
    settings: Settings

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=max(20.0, self.settings.request_timeout_seconds))
        self._training_lock = asyncio.Lock()
        self._model_dir = Path(self.settings.ml_model_dir)
        self._db_path = Path(self.settings.ml_database_path)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    async def close(self) -> None:
        await self._client.aclose()

    async def train(
        self,
        *,
        location: Coordinates,
        history_days: int,
        epochs: int,
        force_retrain: bool,
        min_retrain_hours: int,
    ) -> dict:
        if self._training_lock.locked():
            return {
                "status": "training_in_progress",
                "reason": "Another training job is running. Please wait and retry.",
            }

        meta = self._load_metadata()
        now_utc = datetime.now(tz=timezone.utc)
        trained_at = _parse_iso_datetime(meta.get("trained_at_utc")) if meta else None

        if (
            meta
            and trained_at
            and not force_retrain
            and (now_utc - trained_at) < timedelta(hours=min_retrain_hours)
        ):
            age_hours = round((now_utc - trained_at).total_seconds() / 3600, 2)
            return {
                "status": "skipped_recent_training",
                "reason": f"Existing model is only {age_hours} hours old.",
                "trained_at_utc": meta.get("trained_at_utc"),
                "best_model": meta.get("best_model"),
                "metrics": meta.get("metrics", {}),
                "location": meta.get("location"),
            }

        async with self._training_lock:
            historical_payload = await self._fetch_historical_data(
                latitude=location.latitude,
                longitude=location.longitude,
                timezone_name=location.timezone,
                history_days=history_days,
            )
            return await asyncio.to_thread(
                self._train_from_payload,
                location,
                history_days,
                epochs,
                historical_payload,
            )

    async def predict(self, *, location: Coordinates, horizon_hours: int) -> dict:
        metadata = self._load_metadata()
        if not metadata:
            raise RuntimeError("Model not trained yet. Run /api/ml/train first.")

        feature_cols = metadata.get("feature_columns", [])
        if not feature_cols:
            raise RuntimeError("Model metadata is missing feature columns. Retrain the model.")

        forecast_payload = await self._fetch_forecast_features(
            latitude=location.latitude,
            longitude=location.longitude,
            timezone_name=location.timezone,
            horizon_hours=max(1, horizon_hours),
        )
        return await asyncio.to_thread(
            self._predict_from_payload,
            location,
            horizon_hours,
            metadata,
            forecast_payload,
        )

    async def get_metrics(self) -> dict:
        metadata = self._load_metadata()
        db_stats = self._query_db_stats()
        return {
            "model_available": bool(metadata),
            "trained_at_utc": metadata.get("trained_at_utc") if metadata else None,
            "best_model": metadata.get("best_model") if metadata else None,
            "metrics": metadata.get("metrics") if metadata else {},
            "lstm_backend": metadata.get("lstm_backend") if metadata else None,
            "xgboost_backend": metadata.get("xgboost_backend") if metadata else None,
            "feature_count": len(metadata.get("feature_columns", [])) if metadata else 0,
            "training_runs_total": db_stats["training_runs_total"],
            "prediction_logs_total": db_stats["prediction_logs_total"],
            "last_training_run": db_stats["last_training_run"],
            "last_prediction_log": db_stats["last_prediction_log"],
        }

    async def _fetch_historical_data(
        self,
        *,
        latitude: float,
        longitude: float,
        timezone_name: str,
        history_days: int,
    ) -> dict:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=history_days - 1)
        response = await self._client.get(
            self.settings.open_meteo_archive_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timezone": timezone_name or "auto",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
            },
        )
        response.raise_for_status()
        return response.json()

    async def _fetch_forecast_features(
        self,
        *,
        latitude: float,
        longitude: float,
        timezone_name: str,
        horizon_hours: int,
    ) -> dict:
        forecast_days = max(2, math.ceil(horizon_hours / 24) + 1)
        response = await self._client.get(
            f"{self.settings.open_meteo_base_url}/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone_name or "auto",
                "forecast_days": forecast_days,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
            },
        )
        response.raise_for_status()
        return response.json()

    def _train_from_payload(
        self,
        location: Coordinates,
        history_days: int,
        epochs: int,
        historical_payload: dict,
    ) -> dict:
        now_utc = datetime.now(tz=timezone.utc)
        history_frame = _historical_payload_to_frame(historical_payload)
        feature_frame = _build_feature_frame(history_frame)
        dataset = _build_supervised_dataset(feature_frame)

        if len(dataset) < 300:
            raise ValueError(
                f"Not enough training samples ({len(dataset)}). Increase history window or change location."
            )

        split_idx = int(len(dataset) * 0.8)
        train_df = dataset.iloc[:split_idx]
        test_df = dataset.iloc[split_idx:]
        feature_cols = [col for col in dataset.columns if col != "target_temperature_c"]

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_temperature_c"].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["target_temperature_c"].to_numpy(dtype=float)

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        rf_model = RandomForestRegressor(
            n_estimators=280,
            max_depth=16,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(x_train_scaled, y_train)
        rf_pred = rf_model.predict(x_test_scaled)
        rf_metrics = _regression_metrics(y_true=y_test, y_pred=rf_pred)

        if XGBOOST_AVAILABLE:
            xgb_model = XGBRegressor(
                n_estimators=320,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
        else:
            xgb_model = XGBRegressor(
                n_estimators=320,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        xgb_model.fit(x_train_scaled, y_train)
        xgb_pred = xgb_model.predict(x_test_scaled)
        xgb_metrics = _regression_metrics(y_true=y_test, y_pred=xgb_pred)

        lstm_result = self._train_lstm_or_fallback(
            x_train_scaled=x_train_scaled,
            y_train=y_train,
            x_test_scaled=x_test_scaled,
            y_test=y_test,
            epochs=epochs,
        )

        metrics_by_model = {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
            "lstm": lstm_result["metrics"],
        }
        best_model = min(metrics_by_model.items(), key=lambda item: item[1]["rmse"])[0]

        training_summary = {
            "trained_at_utc": now_utc.isoformat(),
            "location": {
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "timezone": location.timezone,
            },
            "history_days": history_days,
            "feature_columns": feature_cols,
            "target_column": "target_temperature_c",
            "sample_count": int(len(dataset)),
            "split": {"train_samples": int(len(train_df)), "test_samples": int(len(test_df))},
            "metrics": metrics_by_model,
            "best_model": best_model,
            "target_baseline": {
                "mean": round(float(np.mean(y_train)), 3),
                "std": round(float(np.std(y_train)) or 1.0, 3),
            },
            "lstm_backend": lstm_result["backend"],
            "lstm_sequence_length": lstm_result.get("sequence_length"),
            "xgboost_backend": "xgboost" if XGBOOST_AVAILABLE else "sklearn_gradient_boosting_fallback",
        }

        self._persist_training_artifacts(
            scaler=scaler,
            rf_model=rf_model,
            xgb_model=xgb_model,
            lstm_result=lstm_result,
            metadata=training_summary,
        )
        self._insert_training_log(
            location=location,
            history_days=history_days,
            sample_count=int(len(dataset)),
            best_model=best_model,
            status="trained",
            metrics=metrics_by_model,
        )

        return {
            "status": "trained",
            "best_model": best_model,
            "metrics": metrics_by_model,
            "sample_count": int(len(dataset)),
            "trained_at_utc": training_summary["trained_at_utc"],
            "lstm_backend": lstm_result["backend"],
            "xgboost_backend": training_summary["xgboost_backend"],
        }

    def _predict_from_payload(
        self,
        location: Coordinates,
        horizon_hours: int,
        metadata: dict,
        forecast_payload: dict,
    ) -> dict:
        feature_cols = metadata.get("feature_columns", [])
        scaler = self._load_pickle_artifact("scaler.pkl")
        rf_model = self._load_pickle_artifact("rf_model.pkl")
        xgb_model = self._load_pickle_artifact("xgb_model.pkl")
        lstm_backend = metadata.get("lstm_backend", "mlp")

        forecast_frame = _forecast_payload_to_frame(forecast_payload)
        engineered_frame = _build_feature_frame(forecast_frame)
        for col in feature_cols:
            if col not in engineered_frame.columns:
                engineered_frame[col] = 0.0

        x_inference = engineered_frame[feature_cols].to_numpy(dtype=float)
        x_scaled = scaler.transform(x_inference)
        rf_pred = rf_model.predict(x_scaled)
        xgb_pred = xgb_model.predict(x_scaled)

        lstm_pred = self._predict_with_lstm_artifact(
            x_scaled=x_scaled,
            lstm_backend=lstm_backend,
            sequence_length=int(metadata.get("lstm_sequence_length") or 24),
        )

        model_stack = np.vstack([rf_pred, xgb_pred, lstm_pred])
        ensemble_pred = np.mean(model_stack, axis=0)
        per_point_std = np.std(model_stack, axis=0)
        per_point_confidence = np.clip(100 - (per_point_std * 12.5), 10, 99.9)

        horizon = min(horizon_hours, len(engineered_frame))
        target_mean = float(metadata.get("target_baseline", {}).get("mean", np.mean(ensemble_pred)))
        target_std = float(metadata.get("target_baseline", {}).get("std", np.std(ensemble_pred) or 1.0))
        target_std = max(target_std, 0.1)

        rows = []
        anomaly_scores: list[float] = []
        storm_scores: list[float] = []

        for idx in range(horizon):
            stamp = engineered_frame.iloc[idx]["time"]
            predicted_temp = float(ensemble_pred[idx])
            confidence_pct = float(per_point_confidence[idx])
            rain_mm = float(engineered_frame.iloc[idx].get("precipitation_mm", 0.0))
            wind_kph = float(engineered_frame.iloc[idx].get("wind_speed_kph", 0.0))
            pressure_hpa = float(engineered_frame.iloc[idx].get("pressure_hpa", 1012.0))
            anomaly_score = min(100.0, abs(predicted_temp - target_mean) / target_std * 18.0)
            storm_score = min(
                100.0,
                rain_mm * 20.0 + max(0.0, wind_kph - 25.0) * 2.6 + max(0.0, 1012.0 - pressure_hpa) * 3.2 + anomaly_score * 0.25,
            )

            anomaly_scores.append(anomaly_score)
            storm_scores.append(storm_score)
            rows.append(
                {
                    "time": stamp.isoformat() if isinstance(stamp, pd.Timestamp) else str(stamp),
                    "predicted_temperature_c": round(predicted_temp, 2),
                    "model_predictions": {
                        "random_forest": round(float(rf_pred[idx]), 2),
                        "xgboost": round(float(xgb_pred[idx]), 2),
                        "lstm": round(float(lstm_pred[idx]), 2),
                    },
                    "confidence_pct": round(confidence_pct, 2),
                    "anomaly_score": round(anomaly_score, 2),
                    "storm_alert_score": round(storm_score, 2),
                }
            )

        overall_confidence = round(float(mean([row["confidence_pct"] for row in rows])), 2) if rows else 0.0
        max_anomaly = round(max(anomaly_scores), 2) if anomaly_scores else 0.0
        max_storm_score = round(max(storm_scores), 2) if storm_scores else 0.0
        alert_level = _alert_level_from_score(max_storm_score)

        summary = {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "horizon_hours": horizon,
            "overall_confidence_pct": overall_confidence,
            "max_anomaly_score": max_anomaly,
            "storm_alert_score": max_storm_score,
            "alert_level": alert_level,
            "best_model": metadata.get("best_model"),
            "models_used": ["random_forest", "xgboost", "lstm"],
            "location": {
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
            },
        }

        self._insert_prediction_log(
            location=location,
            horizon_hours=horizon,
            confidence_score=overall_confidence,
            anomaly_score=max_anomaly,
            storm_alert_score=max_storm_score,
            alert_level=alert_level,
            summary=summary,
        )

        return {
            "summary": summary,
            "predictions": rows,
            "training_metrics": metadata.get("metrics", {}),
        }

    def _train_lstm_or_fallback(
        self,
        *,
        x_train_scaled: np.ndarray,
        y_train: np.ndarray,
        x_test_scaled: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
    ) -> dict:
        sequence_length = 24
        if TENSORFLOW_AVAILABLE:
            train_seq_x, train_seq_y = _to_sequence_data(x_train_scaled, y_train, sequence_length)
            test_seq_x, test_seq_y = _to_sequence_data(x_test_scaled, y_test, sequence_length)

            if len(train_seq_x) >= 48 and len(test_seq_x) >= 8:
                lstm_model = Sequential(
                    [
                        LSTM(64, return_sequences=True, input_shape=(sequence_length, x_train_scaled.shape[1])),
                        Dropout(0.25),
                        LSTM(32),
                        Dropout(0.2),
                        Dense(16, activation="relu"),
                        Dense(1),
                    ]
                )
                lstm_model.compile(optimizer="adam", loss="mse")
                early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                lstm_model.fit(
                    train_seq_x,
                    train_seq_y,
                    validation_data=(test_seq_x, test_seq_y),
                    epochs=epochs,
                    batch_size=32,
                    verbose=0,
                    callbacks=[early_stopping],
                )
                lstm_predictions = lstm_model.predict(test_seq_x, verbose=0).reshape(-1)
                metrics = _regression_metrics(y_true=test_seq_y, y_pred=lstm_predictions)
                return {
                    "backend": "tensorflow_lstm",
                    "metrics": metrics,
                    "model": lstm_model,
                    "sequence_length": sequence_length,
                }

        fallback_model = MLPRegressor(
            hidden_layer_sizes=(160, 80),
            activation="relu",
            random_state=42,
            max_iter=450,
            early_stopping=True,
            validation_fraction=0.15,
        )
        fallback_model.fit(x_train_scaled, y_train)
        fallback_predictions = fallback_model.predict(x_test_scaled)
        metrics = _regression_metrics(y_true=y_test, y_pred=fallback_predictions)
        return {
            "backend": "mlp_fallback",
            "metrics": metrics,
            "model": fallback_model,
            "sequence_length": None,
        }

    def _predict_with_lstm_artifact(
        self,
        *,
        x_scaled: np.ndarray,
        lstm_backend: str,
        sequence_length: int,
    ) -> np.ndarray:
        if lstm_backend == "tensorflow_lstm" and not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed, but model metadata requires TensorFlow LSTM.")

        if lstm_backend == "tensorflow_lstm" and TENSORFLOW_AVAILABLE:
            lstm_path = self._model_dir / "lstm_model.keras"
            if not lstm_path.exists():
                raise RuntimeError("LSTM artifact missing. Retrain the model before prediction.")
            tf_model = load_model(lstm_path)
            seq_x = _build_inference_sequences(x_scaled, sequence_length)
            seq_predictions = tf_model.predict(seq_x, verbose=0).reshape(-1)
            prefix_count = max(0, sequence_length - 1)
            if len(seq_predictions) == 0:
                return np.zeros(len(x_scaled))
            prefix = np.full(prefix_count, seq_predictions[0])
            combined = np.concatenate([prefix, seq_predictions])
            return combined[: len(x_scaled)]

        fallback = self._load_pickle_artifact("lstm_fallback_model.pkl")
        return fallback.predict(x_scaled)

    def _persist_training_artifacts(
        self,
        *,
        scaler: MinMaxScaler,
        rf_model: RandomForestRegressor,
        xgb_model: Any,
        lstm_result: dict,
        metadata: dict,
    ) -> None:
        joblib.dump(scaler, self._model_dir / "scaler.pkl")
        joblib.dump(rf_model, self._model_dir / "rf_model.pkl")
        joblib.dump(xgb_model, self._model_dir / "xgb_model.pkl")

        if lstm_result["backend"] == "tensorflow_lstm":
            lstm_model = lstm_result["model"]
            lstm_model.save(self._model_dir / "lstm_model.keras")
            fallback_path = self._model_dir / "lstm_fallback_model.pkl"
            if fallback_path.exists():
                fallback_path.unlink()
        else:
            joblib.dump(lstm_result["model"], self._model_dir / "lstm_fallback_model.pkl")
            lstm_path = self._model_dir / "lstm_model.keras"
            if lstm_path.exists():
                lstm_path.unlink()

        self._save_metadata(metadata)

    def _save_metadata(self, payload: dict) -> None:
        with (self._model_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _load_metadata(self) -> dict:
        path = self._model_dir / "metadata.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}

    def _load_pickle_artifact(self, filename: str) -> Any:
        path = self._model_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing model artifact: {filename}. Retrain via /api/ml/train.")
        return joblib.load(path)

    def _init_database(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(TRAINING_TABLE_SQL)
            conn.execute(PREDICTION_TABLE_SQL)
            conn.commit()

    def _insert_training_log(
        self,
        *,
        location: Coordinates,
        history_days: int,
        sample_count: int,
        best_model: str,
        status: str,
        metrics: dict,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    created_at_utc, location_name, latitude, longitude, history_days,
                    sample_count, best_model, status, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(tz=timezone.utc).isoformat(),
                    location.name,
                    location.latitude,
                    location.longitude,
                    history_days,
                    sample_count,
                    best_model,
                    status,
                    json.dumps(metrics),
                ),
            )
            conn.commit()

    def _insert_prediction_log(
        self,
        *,
        location: Coordinates,
        horizon_hours: int,
        confidence_score: float,
        anomaly_score: float,
        storm_alert_score: float,
        alert_level: str,
        summary: dict,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    created_at_utc, location_name, latitude, longitude, horizon_hours,
                    confidence_score, anomaly_score, storm_alert_score, alert_level, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(tz=timezone.utc).isoformat(),
                    location.name,
                    location.latitude,
                    location.longitude,
                    horizon_hours,
                    confidence_score,
                    anomaly_score,
                    storm_alert_score,
                    alert_level,
                    json.dumps(summary),
                ),
            )
            conn.commit()

    def _query_db_stats(self) -> dict:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            total_train = conn.execute("SELECT COUNT(*) AS count FROM training_runs").fetchone()
            total_predict = conn.execute("SELECT COUNT(*) AS count FROM prediction_logs").fetchone()
            last_train = conn.execute(
                "SELECT created_at_utc, location_name, best_model, status FROM training_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            last_predict = conn.execute(
                "SELECT created_at_utc, location_name, confidence_score, alert_level FROM prediction_logs ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return {
            "training_runs_total": int(total_train["count"]) if total_train else 0,
            "prediction_logs_total": int(total_predict["count"]) if total_predict else 0,
            "last_training_run": dict(last_train) if last_train else None,
            "last_prediction_log": dict(last_predict) if last_predict else None,
        }


def _historical_payload_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "temperature_c": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
            "humidity_pct": pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
            "wind_speed_kph": pd.to_numeric(hourly.get("wind_speed_10m", []), errors="coerce"),
            "pressure_hpa": pd.to_numeric(hourly.get("surface_pressure", []), errors="coerce"),
            "precipitation_mm": pd.to_numeric(hourly.get("precipitation", []), errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    frame = frame.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return frame


def _forecast_payload_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "temperature_c": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
            "humidity_pct": pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
            "wind_speed_kph": pd.to_numeric(hourly.get("wind_speed_10m", []), errors="coerce"),
            "pressure_hpa": pd.to_numeric(hourly.get("surface_pressure", []), errors="coerce"),
            "precipitation_mm": pd.to_numeric(hourly.get("precipitation", []), errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    frame = frame.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return frame


def _build_feature_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    numeric_columns = ["temperature_c", "humidity_pct", "wind_speed_kph", "pressure_hpa", "precipitation_mm"]

    for col in numeric_columns:
        frame[f"{col}_lag1"] = frame[col].shift(1)
        frame[f"{col}_lag2"] = frame[col].shift(2)
        frame[f"{col}_lag3"] = frame[col].shift(3)
        frame[f"{col}_rolling3"] = frame[col].rolling(window=3, min_periods=1).mean()

    hour = frame["time"].dt.hour.fillna(0)
    day_of_year = frame["time"].dt.dayofyear.fillna(1)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    frame["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    frame = frame.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return frame


def _build_supervised_dataset(feature_frame: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame.copy()
    frame["target_temperature_c"] = frame["temperature_c"].shift(-1)
    frame = frame.dropna(subset=["target_temperature_c"]).reset_index(drop=True)
    return frame


def _to_sequence_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_data) < sequence_length:
        return np.empty((0, sequence_length, x_data.shape[1])), np.empty((0,))
    seq_x: list[np.ndarray] = []
    seq_y: list[float] = []
    for idx in range(sequence_length - 1, len(x_data)):
        seq_x.append(x_data[idx - sequence_length + 1 : idx + 1])
        seq_y.append(float(y_data[idx]))
    return np.array(seq_x), np.array(seq_y)


def _build_inference_sequences(x_data: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(x_data) < sequence_length:
        padded = np.repeat(x_data[:1], sequence_length, axis=0)
        return np.array([padded])
    seq_x: list[np.ndarray] = []
    for idx in range(sequence_length - 1, len(x_data)):
        seq_x.append(x_data[idx - sequence_length + 1 : idx + 1])
    return np.array(seq_x)


def _regression_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    within_2c = float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100.0)
    return {
        "mae": round(float(mae), 4),
        "rmse": round(rmse, 4),
        "accuracy_within_2c_pct": round(within_2c, 2),
    }


def _alert_level_from_score(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 45:
        return "moderate"
    return "low"


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
