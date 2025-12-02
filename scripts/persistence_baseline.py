import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Parse timestamp
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)

    # Sort to ensure chronological order
    df = df.sort_values("DATE_TIME")

    return df


# ---------------------------
#  Aggregate (plant-level)
# ---------------------------
def persistence_aggregate(df: pd.DataFrame):
    # Sum AC power across inverters for each timestamp
    agg = (
        df.groupby("DATE_TIME")["AC_POWER"]
          .sum()
          .sort_index()
    )

    actual = agg.iloc[1:]
    pred = agg.shift(1).iloc[1:]

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    return mae, rmse


# ---------------------------
#  Per-inverter mode
# ---------------------------
def persistence_per_inverter(df: pd.DataFrame, run):
    """
    Logs per-inverter MAE/RMSE individually into MLflow.
    Returns overall mean MAE and RMSE.
    """

    mae_list = []
    rmse_list = []

    # Iterate over each inverter
    for key, group in df.groupby("SOURCE_KEY"):
        group = group.sort_values("DATE_TIME")

        actual = group["AC_POWER"].iloc[1:]
        pred = group["AC_POWER"].shift(1).iloc[1:]

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        mae_list.append(mae)
        rmse_list.append(rmse)

        # Log INDIVIDUAL inverter metrics to MLflow
        mlflow.log_metric(f"mae_inverter_{key}", mae)
        mlflow.log_metric(f"rmse_inverter_{key}", rmse)

    # Return average results for display
    return float(np.mean(mae_list)), float(np.mean(rmse_list))


def run_persistence(path: str, mode: str = "aggregate"):
    """
    mode = "aggregate" or "inverter"
    """

    mlflow.set_experiment("solar-predictor-baselines")

    with mlflow.start_run(run_name=f"persistence-{mode}"):

        df = load_and_prepare(path)

        # Log mode as parameter
        mlflow.log_param("mode", mode)

        # Aggregate mode
        if mode == "aggregate":
            mae, rmse = persistence_aggregate(df)

        # Per-inverter mode
        elif mode == "inverter":
            mae, rmse = persistence_per_inverter(df, mlflow.active_run())

        else:
            raise ValueError("Mode must be 'aggregate' or 'inverter'")

        # Log summary metrics
        mlflow.log_metric("mean_mae", mae)
        mlflow.log_metric("mean_rmse", rmse)

        print(f"\n== Persistence ({mode}) ==")
        print(f"Mean MAE:  {mae:.4f}")
        print(f"Mean RMSE: {rmse:.4f}")

        return mae, rmse


if __name__ == "__main__":
    data_path = Path("./data/raw/solar_generation_data.csv")
    run_persistence(data_path, mode="aggregate")
    run_persistence(data_path, mode="inverter")
