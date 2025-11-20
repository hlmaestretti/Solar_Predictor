"""
This script is used to ingest the data and perform exploratory data analysis.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_PATH = Path("data/raw/solar_generation_data.csv")

def load_data() -> pd.DataFrame:
    """
    Load the data from the CSV file.
    Args:
        None
    Returns:
        pd.DataFrame: The loaded data.
    """
    print(f"Loading dataset from: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)

    print("\n===== Data Info =====")
    print(df.info())

    print("\n===== Head =====")
    print(df.head())

    print("\n===== Missing Values =====")
    print(df.isna().sum())

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.
    Args:
        df: pd.DataFrame: The loaded data.
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # 1. Parse timestamp column
    timestamp_col = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    if timestamp_col:
        ts_col = timestamp_col[0]
        df[ts_col] = pd.to_datetime(df[ts_col])

    # 2. Sort chronologically
    if timestamp_col:
        df = df.sort_values(by=ts_col)

    # 3. Basic missing value handling
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


def quick_eda(df: pd.DataFrame) -> None:
    """
    Perform quick exploratory data analysis.
    Args:
        df: pd.DataFrame: The preprocessed data.
    Returns:
        None
    """
    # Identify likely target column
    target_candidates = [c for c in df.columns if "power" in c.lower() or "energy" in c.lower()]
    target = target_candidates[0] if target_candidates else df.columns[-1]

    # Time column
    time_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    ts = time_candidates[0] if time_candidates else None

    print(f"\nUsing target column: {target}")
    print(f"Using timestamp column: {ts}")

    # Plot target over time
    if ts:
        plt.figure(figsize=(12, 5))
        plt.plot(df[ts], df[target])
        plt.title("Solar Output Over Time")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time features.
    Args:
        df: pd.DataFrame: The preprocessed data.
    Returns:
        pd.DataFrame: The data with time features.
    """
    time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not time_cols:
        return df

    ts = time_cols[0]
    df["hour"] = df[ts].dt.hour
    df["day"] = df[ts].dt.day
    df["month"] = df[ts].dt.month
    df["dayofweek"] = df[ts].dt.dayofweek
    df["year"] = df[ts].dt.year

    return df


def train_val_test_split(df: pd.DataFrame, test_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train, validation, and test sets.
    Args:
        df: pd.DataFrame: The preprocessed data.
        test_days: int: The number of days to use for the test set.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The train, validation, and test sets.
    """
    time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not time_cols:
        raise ValueError("No timestamp column found for time-based split.")

    ts = time_cols[0]

    df = df.sort_values(by=ts)
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]

    return train, test


if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    quick_eda(df)
    df = create_time_features(df)

    train, test = train_val_test_split(df)
    print(f"\nTrain size: {len(train)}  Test size: {len(test)}")

    print("\n=== Step 1 Complete ===")