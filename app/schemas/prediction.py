from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    temperature_c: float = Field(..., description="Ambient temperature in Celsius")
    cloud_cover_pct: float = Field(..., ge=0, le=100, description="Cloud cover percentage")
    humidity_pct: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed_mps: float = Field(..., ge=0, description="Wind speed in meters per second")


class PredictionResponse(BaseModel):
    predicted_power: float = Field(..., description="Predicted solar power output in kW")
