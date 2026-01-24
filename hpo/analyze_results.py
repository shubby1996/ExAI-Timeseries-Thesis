"""
Analyze HPO results across all models and datasets.

Usage:
  python hpo/analyze_results.py
  python hpo/analyze_results.py --model NHITS_Q
  python hpo/analyze_results.py --dataset heat
"""

import os
import json
import pandas as pd
import argparse
from glob import glob


def load_all_results():
    """Load all HPO result files."""
    result_files = glob("hpo/results/*/best_params_*.json")
    
    results = []
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append({
                "model": data["model"],
                "dataset": data["dataset"],
                "job_id": data["job_id"],
                "mae": data["best_mae"],
                "picp": data["best_picp_approx"],
                "n_trials": data["n_trials"],
                "n_pareto": data["n_pareto_optimal"],
                "file": file
            })
    
    return pd.DataFrame(results)


def print_summary(df, model_filter=None, dataset_filter=None):
    """Print summary of results."""
    if model_filter:
        df = df[df["model"] == model_filter]
    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]
    
    print("\n" + "="*80)
    print("HPO RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Group by model
    print("\n📊 RESULTS BY MODEL:")
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        print(f"  Best MAE:  {model_df['mae'].min():.6f} ({model_df.loc[model_df['mae'].idxmin(), 'dataset']})")
        print(f"  Best PICP: {model_df['picp'].max():.2f}% ({model_df.loc[model_df['picp'].idxmax(), 'dataset']})")
    
    # Group by dataset
    print("\n📊 RESULTS BY DATASET:")
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        print(f"\n{dataset}:")
        print(f"  Best MAE:  {dataset_df['mae'].min():.6f} ({dataset_df.loc[dataset_df['mae'].idxmin(), 'model']})")
        print(f"  Best PICP: {dataset_df['picp'].max():.2f}% ({dataset_df.loc[dataset_df['picp'].idxmax(), 'model']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    
    df = load_all_results()
    
    if len(df) == 0:
        print("❌ No HPO results found in hpo/results/")
        exit(1)
    
    print_summary(df, args.model, args.dataset)
    print(f"\n✅ Analyzed {len(df)} HPO result files")
