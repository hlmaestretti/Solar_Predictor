import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# 1. Load + preprocess the raw data
# ------------------------------------------------------------
def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Parse timestamp
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)

    # Sort in chronological order
    df = df.sort_values("DATE_TIME")

    return df


# ------------------------------------------------------------
# 2. Aggregate AC power for plant-level target
# ------------------------------------------------------------
def aggregate_plant_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with one row per timestamp
    and aggregated AC_POWER across all inverters.
    """
    agg = (
        df.groupby("DATE_TIME")["AC_POWER"]
          .sum()
          .sort_index()
          .to_frame(name="AC_POWER_PLANT")
    )

    return agg


# ------------------------------------------------------------
# 3. Feature engineering (lags + time features)
# ------------------------------------------------------------
def add_features(df: pd.DataFrame, lag_hours=1):
    """
    df is the aggregated plant-level dataframe.
    Creates:
      - time features
      - lag features for AC_POWER_PLANT
    """
    df = df.copy()

    # Time features
    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear
    df["dayofweek"] = df.index.dayofweek

    # Lag features (15-minute intervals)
    lag_steps = lag_hours * 4  # 4 intervals per hour
    for i in range(1, lag_steps + 1):
        df[f"lag_{i}"] = df["AC_POWER_PLANT"].shift(i)

    # Drop rows that do not have full lag context
    df = df.dropna()

    return df


# ------------------------------------------------------------
# 4. Train / test split (chronological)
# ------------------------------------------------------------
def split_train_test(df: pd.DataFrame, test_ratio=0.2):
    n_total = len(df)
    n_test = int(n_total * test_ratio)

    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    return train, test


# ------------------------------------------------------------
# 5. Train Random Forest regressor
# ------------------------------------------------------------
def train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame):
    features = [col for col in train_df.columns if col != "AC_POWER_PLANT"]

    X_train = train_df[features]
    y_train = train_df["AC_POWER_PLANT"]

    X_test = test_df[features]
    y_test = test_df["AC_POWER_PLANT"]

    # RF defaults (can tune later)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, mae, rmse


# ------------------------------------------------------------
# 6. MLflow logging wrapper
# ------------------------------------------------------------
def run_training(path: str):
    mlflow.set_experiment("solar-predictor-models")

    with mlflow.start_run(run_name="RF-plant-level"):

        df_raw = load_and_prepare(path)
        df_agg = aggregate_plant_power(df_raw)
        df_feat = add_features(df_agg, lag_hours=1)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True, parents=True)        
        df_feat.to_parquet(processed_dir / "features.parquet")
        print(f"Saved processed training features to {processed_dir / 'features.parquet'}")

        train_df, test_df = split_train_test(df_feat)

        model, mae, rmse = train_random_forest(train_df, test_df)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("lag_hours", 1)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Log model artifact
        mlflow.sklearn.log_model(model, "rf_model")

        print("\n=== Random Forest Training Complete ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        return model, mae, rmse


if __name__ == "__main__":
    data_path = Path("./data/raw/solar_generation_data.csv")  
    run_training(data_path)
