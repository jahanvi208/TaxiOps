from __future__ import annotations

from fastapi.testclient import TestClient

from taxiops.api import app as module


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == [
            "trip_distance", "passenger_count", "pickup_zone_id", "dropoff_zone_id",
            "pickup_hour", "pickup_weekday", "pickup_month",
        ]
        return [24.569]


def test_health_before_model_is_loaded():
    module.get_model.cache_clear()
    response = TestClient(module.app).get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_prediction(monkeypatch, valid_request):
    monkeypatch.setattr(module, "get_model", lambda: FakeModel())
    response = TestClient(module.app).post("/predict", json=valid_request)
    assert response.status_code == 200
    assert response.json()["estimated_fare_usd"] == 24.57
    assert response.json()["model_alias"] == "champion"


def test_unavailable_registry_returns_503(monkeypatch, valid_request):
    def missing_model():
        raise ConnectionError("tracking server unavailable")

    monkeypatch.setattr(module, "get_model", missing_model)
    response = TestClient(module.app).post("/predict", json=valid_request)
    assert response.status_code == 503
    assert "tracking server unavailable" not in response.text


def test_bad_request_returns_422(valid_request):
    payload = {**valid_request, "pickup_hour": 44}
    response = TestClient(module.app).post("/predict", json=payload)
    assert response.status_code == 422


def test_prometheus_metrics_exposed():
    response = TestClient(module.app).get("/metrics")
    assert response.status_code == 200
    assert "taxiops_http_requests_total" in response.text


def test_unknown_endpoint_is_404():
    assert TestClient(module.app).get("/not-real").status_code == 404
