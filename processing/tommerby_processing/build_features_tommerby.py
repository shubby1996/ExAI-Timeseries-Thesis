"""
Layer 1 pipeline: from raw water + weather data to feature-engineered CSV for Tommerby.

This script orchestrates your existing scripts:
- align_data_tommerby.py
- feature_engineering_tommerby.py

It produces: tommerby_features_engineered.csv
"""

from align_data_tommerby import align_data_tommerby
from feature_engineering_tommerby import engineer_features_tommerby


def build_tommerby_features(run_align: bool = True, run_feature_engineering: bool = True) -> None:
    """
    Run the full Layer 1 preprocessing pipeline for Tommerby water data.

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
        align_data_tommerby()

    if run_feature_engineering:
        print("=== Step 2: Engineer features (lags, holidays, etc.) ===")
        engineer_features_tommerby()

    print("\nLayer 1 complete. You should now have 'tommerby_features_engineered.csv'.\n")


if __name__ == "__main__":
    build_tommerby_features()
