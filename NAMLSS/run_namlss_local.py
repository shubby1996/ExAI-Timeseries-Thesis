#!/usr/bin/env python3
"""
Local NAMLSS Training and Prediction Script

Runs NAMLSS model training and prediction on CPU, saving results in the same
format as the benchmarker framework for direct comparison with other models.

Environment:
    This script requires the 'myenv' conda environment.
    
    To activate:
        conda activate myenv
    
    Or use the wrapper script:
        ./run_namlss_myenv.sh --dataset nordbyen_heat --n_epochs 5

Usage:
    # Quick test (5 epochs, no CQR)
    conda activate myenv
    python3 run_namlss_local.py --dataset nordbyen_heat --n_epochs 5
    
    # Full training (30 epochs with CQR)
    conda activate myenv
    python3 run_namlss_local.py --dataset nordbyen_heat --n_epochs 30 --use_cqr
    
    # Water dataset
    conda activate myenv
    python3 run_namlss_local.py --dataset water_centrum --n_epochs 20

Results saved to: models/{dataset}/NAMLSS*
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path (go up one level from NAMLSS directory)
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import benchmarker (uses the existing NAMLSSAdapter)
from benchmarker import Benchmarker


def get_dataset_config(dataset_name):
    """Get CSV path and dataset identifier for benchmarker."""
    configs = {
        "nordbyen_heat": {
            "csv_path": "processing/nordbyen_processing/nordbyen_features_engineered.csv",
            "dataset": "nordbyen_heat"
        },
        "water_centrum": {
            "csv_path": "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv",
            "dataset": "Water (Centrum)"
        },
        "water_tommerby": {
            "csv_path": "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv",
            "dataset": "Water (Tommerby)"
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(configs.keys())}")
    
    return configs[dataset_name]


def main():
    parser = argparse.ArgumentParser(
        description="Run NAMLSS training and prediction locally (CPU optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 5-epoch test on heat dataset
  python3 run_namlss_local.py --dataset nordbyen_heat --n_epochs 5
  
  # Full 30-epoch training with CQR calibration
  python3 run_namlss_local.py --dataset nordbyen_heat --n_epochs 30 --use_cqr
  
  # Water dataset with 20 epochs
  python3 run_namlss_local.py --dataset water_centrum --n_epochs 20

Results:
  Models and predictions saved to: models/{dataset}/NAMLSS*
  - NAMLSS.pt                     : Model checkpoint
  - NAMLSS_predictions.csv         : Test set predictions
  - NAMLSS_predictions_cqr.csv     : CQR-calibrated predictions (if --use_cqr)
  - NAMLSS_metrics.json            : Performance metrics
  - NAMLSS_preprocessing_state.pkl : Scalers and config
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["nordbyen_heat", "water_centrum", "water_tommerby"],
        help="Dataset to train on"
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10). Use 5 for quick test, 30 for full training"
    )
    
    parser.add_argument(
        "--use_cqr",
        action="store_true",
        help="Enable Conformal Quantile Regression (CQR) calibration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu). Use cuda only if GPU is available"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size (default: 128)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    args = parser.parse_args()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    csv_path_relative = dataset_config["csv_path"]
    dataset_name = dataset_config["dataset"]
    
    # Convert to absolute path using project_root
    csv_path = os.path.join(project_root, csv_path_relative)
    
    # Verify CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        print(f"   Project root: {project_root}")
        print(f"   Relative path: {csv_path_relative}")
        sys.exit(1)
    
    # Change to project root directory (so benchmarker saves to correct locations)
    print(f"\n📁 Changing to project root: {project_root}")
    os.chdir(project_root)
    print(f"✓ Working directory: {os.getcwd()}\n")
    
    # Print configuration
    print("="*70)
    print("NAMLSS LOCAL TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"CSV path: {csv_path}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"CQR Calibration: {'Enabled' if args.use_cqr else 'Disabled'}")
    print("="*70 + "\n")
    
    # Create benchmarker instance
    benchmarker = Benchmarker(csv_path, ["NAMLSS"], dataset=dataset_name)
    
    # Verify NAMLSS is in configs
    if "NAMLSS" not in benchmarker.configs:
        print("❌ Error: NAMLSS not found in benchmarker configs")
        print("   Make sure benchmarker.py has been updated with NAMLSSAdapter")
        sys.exit(1)
    
    # Override NAMLSS config with command-line arguments
    benchmarker.configs["NAMLSS"]["n_epochs"] = args.n_epochs
    benchmarker.configs["NAMLSS"]["device"] = args.device
    benchmarker.configs["NAMLSS"]["batch_size"] = args.batch_size
    benchmarker.configs["NAMLSS"]["lr"] = args.lr
    
    # Run benchmarking
    benchmarker.run(use_cqr=args.use_cqr)
    
    print("\n" + "="*70)
    print("✅ NAMLSS TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to:")
    results_dir = os.path.join(project_root, f"models/{dataset_name}")
    benchmark_dir = os.path.join(project_root, f"{dataset_name}_benchmark/results")
    print(f"\n📁 Model directory: {results_dir}/")
    print(f"  - NAMLSS.pt (model checkpoint)")
    print(f"  - NAMLSS_predictions.csv (test predictions)")
    if args.use_cqr:
        print(f"  - NAMLSS_predictions_cqr.csv (CQR-calibrated)")
    print(f"  - NAMLSS_metrics.json (performance metrics)")
    print(f"  - NAMLSS_preprocessing_state.pkl (scalers)")
    print(f"\n📁 Benchmark directory: {benchmark_dir}/")
    print(f"  - Results CSV files with all metrics")
    
    # Show quick metrics if available
    import json
    metrics_path = os.path.join(results_dir, "NAMLSS_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\n📊 Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:20s}: {value:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
