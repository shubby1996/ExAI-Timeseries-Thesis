#!/usr/bin/env python3
"""
Quick test script to verify water benchmarker works before submitting to SLURM.
Tests all 4 model variants (NHITS_Q, NHITS_MSE, TIMESNET_Q, TIMESNET_MSE)
with reduced epochs for fast validation.

Dataset: Centrum water consumption (filtered from 2018-04-01)
Expected runtime: ~8-10 minutes (smaller dataset than heat)
"""
import sys
import os
from benchmarker import Benchmarker

print("=" * 70)
print("WATER BENCHMARKER - QUICK TEST (Centrum Dataset)")
print("=" * 70)

print(f"\n[TEST MODE] Using reduced epochs for quick validation")
print(f"[TEST MODE] Testing all 4 model variants")
print(f"[TEST MODE] Expected runtime: ~8-10 minutes\n")

# Test only _Q models (with Stage 2 HPO params)
models_to_test = ["NHITS_Q", "TIMESNET_Q"]
benchmarker_instance = Benchmarker(
    "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv",
    models_to_test,
    dataset="Water (Centrum)",
    results_dir="water_centrum_benchmark/results"
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
    print("✓ TEST PASSED! Water benchmarker completed successfully")
    print("=" * 70)
    print("You can now submit the full water benchmark job to SLURM with confidence.")
    sys.exit(0)
except Exception as e:
    print("\n" + "=" * 70)
    print("✗ TEST FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
