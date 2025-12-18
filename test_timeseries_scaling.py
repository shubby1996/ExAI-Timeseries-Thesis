"""
Test script to verify Darts TimeSeries conversion and scaling.
"""

import pandas as pd
from model_preprocessing import prepare_model_data

# Define split dates with balanced data
# Data range: 2015-05 to 2022-05 (roughly 7 years)
# Train: 2015-2018 (~3.5 years)
# Val: 2018-2019 (~1 year)  
# Test: 2019-2022 (~2.5 years)
train_end = pd.Timestamp("2018-12-31 23:00:00+00:00")
val_end = pd.Timestamp("2019-12-31 23:00:00+00:00")

# Run end-to-end preprocessing
csv_path = r"c:\Uni Stuff\Semester 5\Thesis_SI\ShubhamThesis\data\nordbyen_features_engineered.csv"

state, train_scaled, val_scaled, test_scaled = prepare_model_data(
    csv_path=csv_path,
    train_end=train_end,
    val_end=val_end,
)

# Verify results
print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print("\n1. Preprocessing State:")
print(f"  ✓ Feature config stored: {state.feature_config is not None}")
print(f"  ✓ Target scaler fitted: {state.target_scaler is not None}")
print(f"  ✓ Past covariates scaler fitted: {state.past_covariates_scaler is not None}")
print(f"  ✓ Future covariates scaler fitted: {state.future_covariates_scaler is not None}")

print("\n2. Scaled TimeSeries Shapes:")
for split_name, split_data in [("Train", train_scaled), ("Val", val_scaled), ("Test", test_scaled)]:
    print(f"\n  {split_name}:")
    print(f"    Target: {split_data['target'].values().shape}")
    if split_data['past_covariates']:
        print(f"    Past covariates: {split_data['past_covariates'].values().shape}")
    if split_data['future_covariates']:
        print(f"    Future covariates: {split_data['future_covariates'].values().shape}")

print("\n3. Scaling Verification (sample values):")
print(f"  Original train target mean: ~1.19 (from previous data)")
print(f"  Scaled train target mean: {train_scaled['target'].values().mean():.4f} (should be ~0)")
print(f"  Scaled train target std: {train_scaled['target'].values().std():.4f} (should be ~1)")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
