from pathlib import Path
from typing import Any

from app.core.config import settings
from app.ml.pipeline import load_model, predict_with_model
from app.schemas.prediction import PredictionRequest


class ModelService:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model: Any | None = None

    @property
    def is_model_ready(self) -> bool:
        if self.model is None:
            self._try_load()
        return self.model is not None

    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        self.model = load_model(self.model_path)

    def predict(self, payload: PredictionRequest) -> float:
        if self.model is None:
            self._try_load()
        if self.model is None:
            raise RuntimeError("Model artifact not found; run training first.")

        features = {
            "temperature_c": payload.temperature_c,
            "cloud_cover_pct": payload.cloud_cover_pct,
            "humidity_pct": payload.humidity_pct,
            "wind_speed_mps": payload.wind_speed_mps,
        }
        return predict_with_model(self.model, features)


service = ModelService(settings.model_path)
