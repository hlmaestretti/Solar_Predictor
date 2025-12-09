# Solar Predictor - FastAPI + Kubeflow + MLflow

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Kubeflow](https://img.shields.io/badge/Kubeflow-Pipelines-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![WhyLogs](https://img.shields.io/badge/WhyLogs-Monitoring-yellow)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![GCP](https://img.shields.io/badge/GCP-Cloud_Run-red)

Short-horizon solar AC power forecasting built as a training exercise. The dataset in `data/raw/solar_generation_data.csv` is small/trivial and is only used for learning; no real operational insights are implied.

## Problem statement
Given recent plant-level AC power readings and timestamps, produce a 15-minute-ahead forecast so that downstream systems could plan dispatch. Short-term PV forecasting is widely used in grid operations, battery dispatch scheduling, and energy markets where 15-minute intervals matter for balancing supply and demand. This repository demonstrates how to stitch together data prep, training, experiment tracking, deployment, and monitoring around that toy problem.

## Solution overview
- The project walks through the full model lifecycle: data ingestion → feature engineering → training → experiment tracking → deployment → monitoring.
- Data prep aggregates inverter-level readings into plant-level AC power.
- Feature engineering adds hour/day features and 15-minute lag values.
- A RandomForestRegressor is trained (chronological split) and tracked with MLflow.
- The best `model.pkl` is served by a FastAPI endpoint at `/predict`, with each call logged to `app/prediction_logs.csv`.
- Drift is checked with whylogs profiles comparing training features vs. live prediction logs.

## Architecture diagram
```mermaid
flowchart LR
    A[Raw CSV\n data/raw/solar_generation_data.csv] --> B[data_prep_component\ncomponents/data_prep.py]
    B --> C[feature_engineering_component\ncomponents/feature_engineering.py]
    C --> D[train_model_component\ncomponents/train_model.py\nMLflow metrics]
    D --> E[Model artifact\nmodel.pkl on GCS or app/model.pkl]
    E --> F[FastAPI inference\napp/main.py + services/model_service.py]
    F --> G[Prediction log\napp/prediction_logs.csv]
    C -.-> H[pipelines/solar_training_pipeline.py\nCompiled to YAML for Kubeflow]
    G --> I[Monitoring\nmonitoring/*.py whylogs drift report]
    D --> J[Experiment tracking\nmlruns/ (MLflow UI)]
```

## Project layout (key files)
- `app/main.py` - FastAPI app with `/health` and `/predict`.
- `app/services/model_service.py` - loads `app/model.pkl` and runs inference.
- `app/schemas/schemas.py` - request/response schemas for prediction.
- `components/` - Kubeflow pipeline components for prep, feature engineering, training, and upload.
- `pipelines/solar_training_pipeline.py` - assembles the Kubeflow pipeline and compiles `solar_training_pipeline.yaml`.
- `scripts/train_rf_models.py` - local training script that logs to MLflow.
- `monitoring/` - whylogs reference/current profile generators and drift report.

## How to run locally
1) Python environment  
`python -m venv .venv && .venv\\Scripts\\activate`

2) Install dependencies  
`pip install -r requirements.txt`

3) Train (optional refresh)  
`python scripts/train_rf_models.py`  
This logs to `mlruns/` and writes processed features to `data/processed/features.parquet`. Download or copy your chosen `model.pkl` into `app/model.pkl` for inference (one is already checked in).

4) Start the API  
`uvicorn app.main:app --reload`

5) Try a prediction  
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{ \"hour\":10, \"dayofyear\":120, \"dayofweek\":2, \"lag_1\":1000, \"lag_2\":980, \"lag_3\":950, \"lag_4\":940 }"
```
Responses are appended to `app/prediction_logs.csv` for monitoring.

## Using Kubeflow
1) Compile the pipeline to YAML:  
`python pipelines/solar_training_pipeline.py` (produces `pipelines/solar_training_pipeline.yaml`).

2) Upload the YAML to your Kubeflow Pipelines instance (UI or `kfp` CLI) and supply parameters:  
   - `raw_csv_path`: path to the raw training CSV.  
   - `lag_hours`: number of hours to build 15-minute lag features (default 1).  
   - `n_estimators`, `min_samples_leaf`: RandomForest hyperparameters.  
   - `gcs_model_path`: GCS URI where the trained `model.pkl` will be uploaded.

3) After the run finishes, download the model artifact (from the run output or the GCS URI) into `app/model.pkl` before starting the FastAPI service.

## Experiment tracking
- MLflow is the tracker; runs are stored under `mlruns/`.  
- Launch the UI locally with:  
  `mlflow ui --backend-store-uri mlruns --port 5000`
- Example latest run (`mlruns/582202843907016186/e1f4e4c81088454d805fc829d258df0c`):  
  - Params: `n_estimators=200`, `min_samples_leaf=2`, `lag_hours=1`.  
  - Metrics on the trivial hold-out: `MAE ~1063.00`, `RMSE ~2082.01`.

## Monitoring and drift checks
- Drift is evaluated using the Kolmogorov–Smirnov (KS) test, comparing training feature distributions against live production inputs. WhyLogs returns per-column drift scores and categories (NO_DRIFT, POSSIBLE_DRIFT, DRIFT).
- Build reference profile from training features:  
  `python monitoring/create_reference_profile.py`
- Build current profile from live predictions:  
  `python monitoring/create_current_profile.py`
- Compare for drift (writes `monitoring/drift_report.json`):  
  `python monitoring/compare_profiles.py`  
  The checked-in report shows `NO_DRIFT` for all features on the toy data.


## Notes and limitations
- This is a training project; the dataset is small and simplistic, and no real operational conclusions should be drawn.
- Replace the sample data and retrain before using anywhere beyond experimentation.

## Future Improvements 

This project is structured as a learning exercise, but it has a clear path toward production-grade maturity. Potential enhancements include:

#### MLOps Enhancements

- Automate container builds, tests, and deployments so that new model versions or API updates are rolled out safely and reproducibly.

- Track multiple model versions, promote models to staging/production, and support rollback workflows.

- Add Katib experiments to the Kubeflow pipeline to automatically search for better model configurations.

- Use Kubeflow recurring runs or Cloud Scheduler to retrain the model periodically or based on drift metrics.

#### Data & Feature Improvements

- Incorporate irradiance, cloud cover, and temperature forecasts for more realistic PV production predictions.

- Add a daily/weekly batch forecasting job (Cloud Run Job, Kubeflow batch pipeline) to produce multi-hour production forecasts.

#### Monitoring & Observability

- Add API latency, throughput, and health metrics to a Grafana dashboard via Cloud Monitoring.

- Model performance monitoring: Store actual production outputs and compare them to model predictions to measure real-world MAE/RMSE.

#### Scalability & Reliability

- Load testing (Locust, k6): Benchmark API performance under realistic or extreme load conditions.

- Improve inference container: Slim down Docker image, add gunicorn workers, or migrate to a multi-stage build for faster startup.