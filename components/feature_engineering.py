from kfp import dsl
import pandas as pd

@dsl.component
def feature_engineering_component(
    cleaned_data: dsl.Input[dsl.Artifact],
    lag_hours: int,
    output_path: dsl.Output[dsl.Artifact]
):
    """
    Creates features:
      - hour, dayofyear, dayofweek
      - lag features for AC_POWER_PLANT
    """
    df = pd.read_parquet(cleaned_data.path)

    # Time features
    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear
    df["dayofweek"] = df.index.dayofweek

    # Lag features (4 intervals per hour for 15min data)
    lag_steps = lag_hours * 4
    for i in range(1, lag_steps + 1):
        df[f"lag_{i}"] = df["AC_POWER_PLANT"].shift(i)

    df = df.dropna()
    df.to_parquet(output_path.path)
