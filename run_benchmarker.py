#!/usr/bin/env python3
"""
Standalone script to run benchmarker for SLURM job submission.
This script runs the benchmarker and saves results, allowing the notebook
to continue with visualization and analysis once complete.
"""
import sys
import argparse
from benchmarker import Benchmarker

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run benchmarking for specified models')
    parser.add_argument('--models', nargs='+', default=["NHITS", "TIMESNET"],
                        help='Models to benchmark (e.g., --models NHITS TIMESNET)')
    parser.add_argument('--data', default="nordbyen_processing/nordbyen_features_engineered.csv",
                        help='Path to the data CSV file')
    args = parser.parse_args()
    
    # Configuration
    DATA_PATH = args.data
    MODELS_TO_RUN = args.models
    
    print("="*70)
    print("BENCHMARKER - SLURM JOB")
    print("="*70)
    print(f"Data: {DATA_PATH}")
    print(f"Models: {', '.join(MODELS_TO_RUN)}")
    print("="*70)
    
    # Initialize and run benchmarker
    benchmarker = Benchmarker(DATA_PATH, MODELS_TO_RUN)
    benchmarker.run()
    
    print("\n" + "="*70)
    print("Benchmarker Complete!")
    print("Results saved to: results/benchmark_results.csv")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
