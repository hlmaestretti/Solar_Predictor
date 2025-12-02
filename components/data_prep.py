from kfp import dsl
import pandas as pd

@dsl.component
def data_prep_component(raw_csv_path: str, output_path: dsl.Output[dsl.Artifact]):
    """
    Loads raw solar dataset, parses timestamp, aggregates AC_POWER
    across inverters, and creates one row per timestamp.
    """
    df = pd.read_csv(raw_csv_path)

    # Parse timestamp
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)
    df = df.sort_values("DATE_TIME")

    # Aggregate AC power across inverters
    agg = (
        df.groupby("DATE_TIME")["AC_POWER"]
          .sum()
          .sort_index()
          .to_frame(name="AC_POWER_PLANT")
    )

    agg.to_parquet(output_path.path)
