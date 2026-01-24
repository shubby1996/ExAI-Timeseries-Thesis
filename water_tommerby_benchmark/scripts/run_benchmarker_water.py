#!/usr/bin/env python3
"""
Water Consumption Benchmarker - Tommerby DMA

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
    parser.add_argument('--models', nargs='+', default=["NHITS_Q", "NHITS_MSE", "TIMESNET_Q", "TIMESNET_MSE", "TFT_Q", "TFT_MSE"],
                        help='Models to benchmark (e.g., --models NHITS_Q TIMESNET_Q TFT_Q)')
    parser.add_argument('--no-cqr', action='store_true',
                        help='Disable Conformalized Quantile Regression (CQR) calibration')
    args = parser.parse_args()
    
    # Water consumption data configuration - path relative to project root (where SLURM runs from)
    DATA_PATH = "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv"
    MODELS_TO_RUN = args.models
    
    print("="*70)
    print("WATER CONSUMPTION BENCHMARKER - TOMMERBY")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Models: {', '.join(MODELS_TO_RUN)}")
    print(f"Target: water_consumption")
    print(f"Train/Val/Test Split: 2018-2019-2020")
    print(f"CQR Calibration: {'DISABLED' if args.no_cqr else 'ENABLED'}")
    print("="*70)
    
    # Initialize and run benchmarker
    benchmarker = Benchmarker(DATA_PATH, MODELS_TO_RUN, dataset="Water (Tommerby)")
    # benchmarker.run(use_cqr=not args.no_cqr)
    benchmarker.run(use_cqr=True)

    
    print("\n" + "="*70)
    print("Water Consumption Benchmarker Complete!")
    print(f"Results saved to: water_tommerby_benchmark/results/")
    print(f"  - Timestamped results: benchmark_results_*_Water_Tommerby_{benchmarker.job_id}.csv")
    print(f"  - Predictions: *_predictions_{benchmarker.job_id}.csv")
    print(f"  - Global history: results/benchmark_history.csv")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
