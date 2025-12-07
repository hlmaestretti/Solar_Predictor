from kfp import dsl
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

@dsl.component
def train_model_component(
    features: dsl.Input[dsl.Artifact],
    n_estimators: int,
    min_samples_leaf: int,
    model_output: dsl.Output[dsl.Artifact]
):
    """
    Loads engineered features, trains RF model,
    logs to MLflow, saves model.pkl.
    """

    mlflow.set_experiment("solar-predictor-kubeflow")

    df = pd.read_parquet(features.path)

    # Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=["AC_POWER_PLANT"])
    y_train = train_df["AC_POWER_PLANT"]

    X_test = test_df.drop(columns=["AC_POWER_PLANT"])
    y_test = test_df["AC_POWER_PLANT"]

    with mlflow.start_run(run_name="kubeflow-rf-training"):

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params + metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save model
        joblib.dump(model, model_output.path)

