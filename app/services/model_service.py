import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

class ModelService:
    """
    Loads the trained RandomForest model and exposes predict() for inference.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                "Place your model.pkl in the app/ directory before running."
            )
        self.model = joblib.load(MODEL_PATH)

    def predict(self, features: np.ndarray) -> float:
        """
        Predict a single row of features.
        Expects a 2D numpy array of shape (1, n_features).
        """
        pred = self.model.predict(features)[0]
        return float(pred)


# Singleton service instance
model_service = ModelService()
