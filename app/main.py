from fastapi import FastAPI, HTTPException
from schemas.schemas import PredictRequest, PredictResponse
from services.model_service import model_service
import numpy as np
import csv
from datetime import datetime

LOG_PATH = "prediction_logs.csv"

app = FastAPI(
    title="Solar Predictor API",
    version="1.0.0",
    description="Predicts plant-level AC power using a trained RandomForest model."
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    API endpoint for producing one solar power forecast.
    """

    try:
        # Order of features must match the RF training script
        features = np.array([
            [
                payload.hour,
                payload.dayofyear,
                payload.dayofweek,
                payload.lag_1,
                payload.lag_2,
                payload.lag_3,
                payload.lag_4
            ]
        ])

        y_pred = model_service.predict(features)

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                payload.hour,
                payload.dayofyear,
                payload.dayofweek,
                payload.lag_1,
                payload.lag_2,
                payload.lag_3,
                payload.lag_4,
                y_pred
            ])

        return PredictResponse(predicted_ac_power=y_pred)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
