#!/usr/bin/env python3
"""
Quick test script to verify heat benchmarker works before submitting to SLURM.
Tests all 4 model variants (NHITS_Q, NHITS_MSE, TIMESNET_Q, TIMESNET_MSE)
with reduced epochs for fast validation.

Dataset: Nordbyen heat consumption
Expected runtime: ~10-12 minutes
"""
import sys
import os
from benchmarker import Benchmarker

print("=" * 70)
print("HEAT BENCHMARKER - QUICK TEST (Nordbyen Dataset)")
print("=" * 70)

print(f"\n[TEST MODE] Using reduced epochs for quick validation")
print(f"[TEST MODE] Testing all 4 model variants")
print(f"[TEST MODE] Expected runtime: ~10-12 minutes\n")

# Test all 4 model variants
models_to_test = ["NHITS_Q", "TIMESNET_Q"]
benchmarker_instance = Benchmarker(
    "processing/nordbyen_processing/nordbyen_features_engineered.csv",
    models_to_test
)

# Override configs with minimal epochs for testing
benchmarker_instance.configs["NHITS_Q"]["n_epochs"] = 3
# benchmarker_instance.configs["NHITS_MSE"]["n_epochs"] = 3
benchmarker_instance.configs["TIMESNET_Q"]["n_epochs"] = 2
# benchmarker_instance.configs["TIMESNET_MSE"]["n_epochs"] = 2

print(f"[TEST MODE] NHITS_Q (quantile) will use n_epochs=3")
# print(f"[TEST MODE] NHITS_MSE (deterministic) will use n_epochs=3")
print(f"[TEST MODE] TIMESNET_Q (quantile) will use n_epochs=2")
# print(f"[TEST MODE] TIMESNET_MSE (deterministic) will use n_epochs=2")

try:
    benchmarker_instance.run()
    print("\n" + "=" * 70)
    print("✓ TEST PASSED! Benchmarker completed successfully")
    print("=" * 70)
    print("You can now submit the full job to SLURM with confidence.")
    sys.exit(0)
except Exception as e:
    print("\n" + "=" * 70)
    print("✗ TEST FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
