from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import ModelService, service


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not service.is_model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prediction = service.predict(payload)
    return PredictionResponse(predicted_power=prediction)
