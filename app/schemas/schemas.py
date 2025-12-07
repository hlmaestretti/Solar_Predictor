from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Input schema that matches the feature vector used in training.
    """
    hour: int = Field(..., ge=0, le=23)
    dayofyear: int = Field(..., ge=1, le=366)
    dayofweek: int = Field(..., ge=0, le=6)

    # Lag features (15-minute intervals)
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float


class PredictResponse(BaseModel):
    """
    Output schema for a prediction response.
    """
    predicted_ac_power: float = Field(
        ..., description="Predicted plant AC power for the next 15-minute interval"
    )
