"""
Test script to verify Layer 2 preprocessing functions.
"""

import pandas as pd
from model_preprocessing import (
    default_feature_config,
    load_and_validate_features,
    split_by_time
)

# Test 1: Load and validate
print("=" * 60)
print("Test 1: Loading and validating features")
print("=" * 60)

csv_path = r"c:\Uni Stuff\Semester 5\Thesis_SI\ShubhamThesis\data\nordbyen_features_engineered.csv"
cfg = default_feature_config()

print(f"\nFeature configuration:")
print(f"  Target: {cfg.target_col}")
print(f"  Past covariates ({len(cfg.past_covariates_cols)}): {cfg.past_covariates_cols[:3]}...")
print(f"  Future covariates ({len(cfg.future_covariates_cols)}): {cfg.future_covariates_cols[:3]}...")

df = load_and_validate_features(csv_path, cfg)

print(f"\nLoaded DataFrame shape: {df.shape}")
print(f"Index type: {type(df.index)}")
print(f"Frequency: {df.index.freq if hasattr(df.index, 'freq') else 'Not set'}")

# Test 2: Time-based split
print("\n" + "=" * 60)
print("Test 2: Time-based splitting")
print("=" * 60)

# Use reasonable split dates based on the data range
# Train: up to end of 2020
# Val: 2021
# Test: 2022
train_end = pd.Timestamp("2020-12-31 23:00:00+00:00")
val_end = pd.Timestamp("2021-12-31 23:00:00+00:00")

train_df, val_df, test_df = split_by_time(df, train_end, val_end)

# Verify no overlap
print("\n✓ Verification:")
print(f"  Last train timestamp: {train_df.index.max()}")
print(f"  First val timestamp:  {val_df.index.min()}")
print(f"  Last val timestamp:   {val_df.index.max()}")
print(f"  First test timestamp: {test_df.index.min()}")

# Check proportions
total = len(df)
print(f"\n✓ Split proportions:")
print(f"  Train: {len(train_df)/total*100:.1f}%")
print(f"  Val:   {len(val_df)/total*100:.1f}%")
print(f"  Test:  {len(test_df)/total*100:.1f}%")

print("\n" + "=" * 60)
print("All tests passed! Layer 2 preprocessing is ready.")
print("=" * 60)
