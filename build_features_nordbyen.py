"""
Layer 1 pipeline: from raw heat + weather data to feature-engineered CSV.

This script orchestrates your existing scripts:
- align_data.py
- feature_engineering.py

It produces: nordbyen_features_engineered.csv
"""

from align_data import align_data
from feature_engineering import engineer_features


def build_nordbyen_features(run_align: bool = True, run_feature_engineering: bool = True) -> None:
    """
    Run the full Layer 1 preprocessing pipeline.

    Parameters
    ----------
    run_align : bool
        If True, re-run the heat+weather alignment step.
        Set to False if you already have an up-to-date aligned file.
    run_feature_engineering : bool
        If True, run feature engineering (including holidays) on the aligned file.
    """
    if run_align:
        print("=== Step 1: Align heat and weather data ===")
        align_data()

    if run_feature_engineering:
        print("=== Step 2: Engineer features (lags, holidays, etc.) ===")
        engineer_features()

    print("\nLayer 1 complete. You should now have 'nordbyen_features_engineered.csv'.\n")


if __name__ == "__main__":
    build_nordbyen_features()
