#!/usr/bin/env python3
"""
Quick test script to verify Tommerby water benchmarker works before submitting to SLURM.
Tests all model variants (NHITS_Q, TIMESNET_Q, TFT_Q)
with reduced epochs for fast validation.

Dataset: Tommerby water consumption (filtered from 2018-04-01)
Expected runtime: ~8-10 minutes (smaller dataset than heat)
"""
import sys
import os
from benchmarker import Benchmarker

print("=" * 70)
print("WATER BENCHMARKER - QUICK TEST (Tommerby Dataset)")
print("=" * 70)

print(f"\n[TEST MODE] Using reduced epochs for quick validation")
print(f"[TEST MODE] Testing all 3 model variants (Quantile loss)")
print(f"[TEST MODE] Expected runtime: ~8-10 minutes\n")

# Test quantile variants (Q models)
models_to_test = ["NHITS_Q", "TIMESNET_Q", "TFT_Q"]
benchmarker_instance = Benchmarker(
    "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv",
    models_to_test,
    dataset="Water (Tommerby)"
)

# Override configs with minimal epochs for testing
benchmarker_instance.configs["NHITS_Q"]["n_epochs"] = 3
benchmarker_instance.configs["TIMESNET_Q"]["n_epochs"] = 1
benchmarker_instance.configs["TFT_Q"]["n_epochs"] = 1

print(f"[TEST MODE] NHITS_Q (quantile) will use n_epochs=3")
print(f"[TEST MODE] TIMESNET_Q (quantile) will use n_epochs=1")
print(f"[TEST MODE] TFT_Q (quantile) will use n_epochs=1")

try:
    benchmarker_instance.run()
    print("\n" + "=" * 70)
    print("✓ TEST PASSED! Tommerby water benchmarker completed successfully")
    print("=" * 70)
    print("You can now submit the full Tommerby water benchmark job to SLURM with confidence.")
    sys.exit(0)
except Exception as e:
    print("\n" + "=" * 70)
    print("✗ TEST FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
