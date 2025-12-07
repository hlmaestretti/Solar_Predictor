"""
This script loads the processed training feature dataset and
creates a WhyLogs reference profile that will be used to detect drift later.

Run this once after training your model.
"""

import whylogs as why
import pandas as pd
from pathlib import Path


FEATURES_PATH = Path("data/processed/features.parquet")
OUTPUT_PROFILE = Path("monitoring/reference_profile.bin")

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Training features not found at: {FEATURES_PATH}")

    print("Loading training feature dataset...")
    df = pd.read_parquet(FEATURES_PATH)

    print("Generating WhyLogs reference profile...")
    results = why.log(df)
    profile = results.profile()  

    OUTPUT_PROFILE.parent.mkdir(exist_ok=True, parents=True)
    profile.write(str(OUTPUT_PROFILE))

    print(f"Reference profile saved to: {OUTPUT_PROFILE}")

if __name__ == "__main__":
    main()
