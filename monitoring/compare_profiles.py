"""
Compares the reference profile (training data)
with the current production profile and writes a drift report.
"""

from whylogs.core import DatasetProfileView
from pathlib import Path
import json



REFERENCE_PATH = Path("monitoring/reference_profile.bin")
CURRENT_PATH = Path("monitoring/current_profile.bin")
OUTPUT_REPORT = Path("monitoring/drift_report.json")

def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError("Reference profile missing. Run create_reference_profile.py first.")
    if not CURRENT_PATH.exists():
        raise FileNotFoundError("Current profile missing. Run create_current_profile.py first.")

    print("Loading profiles...")
    ref = DatasetProfileView.read(str(REFERENCE_PATH))
    curr = DatasetProfileView.read(str(CURRENT_PATH))

    print("Comparing profiles...")
    comparison = ref.compare(curr)
    summary = comparison.to_summary().to_dict()

    OUTPUT_REPORT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Drift report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
