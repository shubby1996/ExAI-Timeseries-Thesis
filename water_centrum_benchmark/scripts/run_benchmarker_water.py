#!/usr/bin/env python3
"""
Water Consumption Benchmarker - Centrum DMA

Runs NHITS and TIMESNET models with both MSE and Quantile losses on water consumption data.

DEPENDENCIES:
    This script relies on shared infrastructure in the project root:
    - ../../benchmarker.py: Core benchmarking framework (ModelAdapter, metrics)
    - ../../model_preprocessing.py: Feature engineering and data preparation
    
    Both files must be present in the root directory for this script to work.

USAGE:
    python run_benchmarker_water.py --models NHITS_Q TIMESNET_Q
"""
import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarker import Benchmarker

def main():
    parser = argparse.ArgumentParser(description='Run benchmarking for water consumption data')
    parser.add_argument('--models', nargs='+', default=["NHITS_Q", "NHITS_MSE", "TIMESNET_Q", "TIMESNET_MSE"],
                        help='Models to benchmark (e.g., --models NHITS_Q TIMESNET_Q)')
    args = parser.parse_args()
    
    # Water consumption data configuration - path relative to project root (where SLURM runs from)
    DATA_PATH = "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv"
    MODELS_TO_RUN = args.models
    
    print("="*70)
    print("WATER CONSUMPTION BENCHMARKER")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Models: {', '.join(MODELS_TO_RUN)}")
    print(f"Target: water_consumption")
    print(f"Train/Val/Test Split: 2018-2019-2020")
    print("="*70)
    
    # Initialize and run benchmarker
    # results_dir points to water_centrum_benchmark/results for dataset-specific storage
    benchmarker = Benchmarker(DATA_PATH, MODELS_TO_RUN, dataset="Water (Centrum)", results_dir="water_centrum_benchmark/results")
    benchmarker.run()
    
    print("\n" + "="*70)
    print("Water Consumption Benchmarker Complete!")
    print("Results saved to:")
    print("  - Project root: results/benchmark_results.csv")
    print("  - Dataset-specific: water_centrum_benchmark/results/benchmark_results.csv")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
