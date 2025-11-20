from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = [
    "temperature_c",
    "cloud_cover_pct",
    "humidity_pct",
    "wind_speed_mps",
]


def train_sample_model(n_samples: int = 500, random_state: int = 42) -> Any:
    rng = np.random.default_rng(random_state)
    data = pd.DataFrame(
        {
            "temperature_c": rng.normal(18, 8, size=n_samples),
            "cloud_cover_pct": rng.uniform(0, 100, size=n_samples),
            "humidity_pct": rng.uniform(10, 95, size=n_samples),
            "wind_speed_mps": rng.uniform(0, 12, size=n_samples),
        }
    )
    # Toy target: higher temp + lower clouds predict higher output with some noise
    noise = rng.normal(0, 2, size=n_samples)
    target = (
        0.4 * data["temperature_c"]
        - 0.25 * (data["cloud_cover_pct"] / 10)
        - 0.1 * (data["humidity_pct"] / 10)
        + 0.3 * data["wind_speed_mps"]
        + 15
        + noise
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=random_state, n_estimators=120)),
        ]
    )
    pipeline.fit(data[FEATURE_ORDER], target)
    return pipeline


def save_model(model: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    return joblib.load(path)


def predict_with_model(model: Any, features: dict[str, float]) -> float:
    row = pd.DataFrame([features], columns=FEATURE_ORDER)
    prediction = model.predict(row)[0]
    return float(prediction)
