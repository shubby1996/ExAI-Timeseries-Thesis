#!/usr/bin/env python3
"""
Quick test script to verify benchmarker works before submitting to SLURM.
This tests with reduced epochs and a shorter evaluation period.
"""
import sys
import os
from benchmarker import Benchmarker

print("=" * 70)
print("QUICK BENCHMARKER TEST (Reduced epochs for fast testing)")
print("=" * 70)

print(f"\n[TEST MODE] Using reduced epochs for quick testing")
print(f"[TEST MODE] This should complete in ~2-3 minutes\n")

# Run benchmarker with test config
models_to_test = ["NHITS"]  # Test NHITS first since it failed
benchmarker_instance = Benchmarker(
    "nordbyen_processing/nordbyen_features_engineered.csv",
    models_to_test
)

# Override configs with minimal epochs for testing
benchmarker_instance.configs["NHITS"]["n_epochs"] = 2
benchmarker_instance.configs["TIMESNET"]["n_epochs"] = 10

print(f"[TEST MODE] NHITS will use n_epochs=2")
print(f"[TEST MODE] TIMESNET will use n_epochs=10")

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
