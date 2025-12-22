"""
Layer 1 pipeline: from raw water + weather data to feature-engineered CSV for Centrum.

This script orchestrates your existing scripts:
- align_data_centrum.py
- feature_engineering_centrum.py

It produces: centrum_features_engineered.csv
"""

from align_data_centrum import align_data_centrum
from feature_engineering_centrum import engineer_features_centrum


def build_centrum_features(run_align: bool = True, run_feature_engineering: bool = True) -> None:
    """
    Run the full Layer 1 preprocessing pipeline for Centrum water data.

    Parameters
    ----------
    run_align : bool
        If True, re-run the water+weather alignment step.
        Set to False if you already have an up-to-date aligned file.
    run_feature_engineering : bool
        If True, run feature engineering (including holidays) on the aligned file.
    """
    if run_align:
        print("=== Step 1: Align water and weather data ===")
        align_data_centrum()

    if run_feature_engineering:
        print("=== Step 2: Engineer features (lags, holidays, etc.) ===")
        engineer_features_centrum()

    print("\nLayer 1 complete. You should now have 'centrum_features_engineered.csv'.\n")


if __name__ == "__main__":
    build_centrum_features()
