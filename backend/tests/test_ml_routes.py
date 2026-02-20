from fastapi.testclient import TestClient

from app import main as main_module


class _FakeMLEngine:
    async def train(self, **kwargs):  # noqa: ANN003
        return {
            "status": "trained",
            "best_model": "xgboost",
            "sample_count": 1200,
            "metrics": {"xgboost": {"mae": 1.1, "rmse": 1.7, "accuracy_within_2c_pct": 82.2}},
        }

    async def predict(self, **kwargs):  # noqa: ANN003
        return {
            "summary": {
                "overall_confidence_pct": 78.4,
                "max_anomaly_score": 22.0,
                "storm_alert_score": 41.0,
                "alert_level": "moderate",
                "best_model": "xgboost",
            },
            "predictions": [
                {"time": "2026-02-19T09:00", "predicted_temperature_c": 29.2, "confidence_pct": 76.4, "storm_alert_score": 31.0}
            ],
            "training_metrics": {"xgboost": {"mae": 1.1, "rmse": 1.7}},
        }

    async def get_metrics(self):
        return {
            "model_available": True,
            "best_model": "xgboost",
            "training_runs_total": 2,
            "prediction_logs_total": 6,
        }


class _FakeMLEngineNotTrained(_FakeMLEngine):
    async def predict(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("Model not trained yet. Run /api/ml/train first.")


def test_ml_train_route_returns_trained_payload(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ml_engine", _FakeMLEngine())
    client = TestClient(main_module.app)

    response = client.post(
        "/api/ml/train",
        json={
            "location": {
                "name": "Chennai",
                "latitude": 13.0878,
                "longitude": 80.2785,
                "timezone": "Asia/Kolkata",
            },
            "history_days": 120,
            "epochs": 10,
            "force_retrain": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "trained"
    assert payload["best_model"] == "xgboost"


def test_ml_predict_route_returns_400_when_not_trained(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ml_engine", _FakeMLEngineNotTrained())
    client = TestClient(main_module.app)

    response = client.post(
        "/api/ml/predict",
        json={
            "location": {
                "name": "Chennai",
                "latitude": 13.0878,
                "longitude": 80.2785,
                "timezone": "Asia/Kolkata",
            },
            "horizon_hours": 24,
        },
    )

    assert response.status_code == 400
    assert "Model not trained yet" in response.json()["detail"]


def test_ml_metrics_route_returns_service_metrics(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ml_engine", _FakeMLEngine())
    client = TestClient(main_module.app)

    response = client.get("/api/ml/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_available"] is True
    assert payload["best_model"] == "xgboost"
