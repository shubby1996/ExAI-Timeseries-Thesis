#!/usr/bin/env python3
"""
Heat Consumption Benchmarker - Nordbyen DMA

Runs NHITS and TIMESNET models for heat consumption forecasting.
Designed for SLURM job submission with results saved for later analysis.

DEPENDENCIES:
    This script relies on shared infrastructure in the project root:
    - ../../benchmarker.py: Core benchmarking framework (ModelAdapter, metrics)
    - ../../model_preprocessing.py: Feature engineering and data preparation
    
    Both files must be present in the root directory for this script to work.

USAGE:
    python run_benchmarker.py --models NHITS_Q TIMESNET_Q
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run benchmarking for specified models')
    parser.add_argument('--models', nargs='+', default=["NHITS_Q", "TIMESNET_Q", "TFT_Q"],
                        help='Models to benchmark (e.g., --models NHITS_Q TIMESNET_Q TFT_Q)')
    parser.add_argument('--data', default="processing/nordbyen_processing/nordbyen_features_engineered.csv",
                        help='Path to the data CSV file')
    parser.add_argument('--no-cqr', action='store_true',
                        help='Disable Conformalized Quantile Regression (CQR) calibration')
    args = parser.parse_args()
    
    # Configuration - path relative to project root (where SLURM runs from)
    DATA_PATH = "processing/nordbyen_processing/nordbyen_features_engineered.csv"
    MODELS_TO_RUN = args.models
    
    print("="*70)
    print("HEAT CONSUMPTION BENCHMARKER - NORDBYEN")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Models: {', '.join(MODELS_TO_RUN)}")
    print(f"CQR Calibration: {'DISABLED' if args.no_cqr else 'ENABLED'}")
    print("="*70)
    
    # Initialize and run benchmarker
    benchmarker = Benchmarker(DATA_PATH, MODELS_TO_RUN, dataset="Heat (Nordbyen)")
    # benchmarker.run(use_cqr=not args.no_cqr)
    benchmarker.run(use_cqr=False)  # Temporarily disable CQR for debugging

    
    print("\n" + "="*70)
    print("Benchmarker Complete!")
    print(f"Results saved to: nordbyen_heat_benchmark/results/")
    print(f"  - Timestamped results: benchmark_results_*_Heat_Nordbyen_{benchmarker.job_id}.csv")
    print(f"  - Predictions: *_predictions_{benchmarker.job_id}.csv")
    print(f"  - Global history: results/benchmark_history.csv")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
