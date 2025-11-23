# Solar Predictor – FastAPI + ML Template

This folder is prepared as a starter for a FastAPI service that wraps a small ML pipeline. It includes:
- FastAPI app with health and prediction routes
- Pydantic schemas for request/response validation
- A minimal ML pipeline using scikit-learn
- Config handling and a training script stub

## Quick start
1) Create and activate a virtual env  
`python -m venv .venv && .venv\\Scripts\\activate`

2) Install dependencies  
`pip install -r requirements.txt`

3) Run the API (reload for dev)  
`uvicorn app.main:app --reload`

4) Hit the docs: `http://127.0.0.1:8000/docs`

## Project layout
- `app/main.py` – FastAPI entrypoint, includes router wiring
- `app/api/routes.py` – API routes (health and `/predict`)
- `app/core/config.py` – settings via `pydantic-settings`
- `app/schemas/prediction.py` – request/response schemas
- `app/services/model_service.py` – model loading + prediction interface
- `app/ml/pipeline.py` – tiny example pipeline (train + predict)
- `scripts/train.py` – example training run that saves an artifact
- `tests/` – starter tests
- `data/` - Contains the raw data along with any augmented data.

## Training stub
`scripts/train.py` shows how to train a simple scikit-learn pipeline and persist it to `app/ml/artifacts/model.joblib`. Replace the synthetic data with your own and extend the pipeline as needed.

## Environment variables
Copy `.env.example` to `.env` if you want to override defaults:
- `MODEL_PATH` – filesystem path to the serialized model artifact
- `APP_NAME` – override service name
