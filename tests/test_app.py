from fastapi.testclient import TestClient

from app.main import app
from app.ml.pipeline import train_sample_model
from app.services.model_service import service


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    service.model = train_sample_model(n_samples=50)
    client = TestClient(app)

    payload = {
        "temperature_c": 22.0,
        "cloud_cover_pct": 20.0,
        "humidity_pct": 40.0,
        "wind_speed_mps": 3.5,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_power" in body
    assert isinstance(body["predicted_power"], (int, float))
