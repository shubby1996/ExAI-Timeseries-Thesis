#!/usr/bin/env python3
"""
Check status of HPO experiments.
Shows tracking data, job status, and results summary.

Usage:
    python hpo/check_status.py
    python hpo/check_status.py --model NHITS_Q --dataset water
"""
import os
import sys
import argparse
import subprocess
import pandas as pd
import json
from datetime import datetime

def get_slurm_status(job_id):
    """Get SLURM job status"""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=5
        )
        status = result.stdout.strip()
        return status if status else "COMPLETED/FAILED"
    except:
        return "UNKNOWN"

def check_results(stage, model, dataset):
    """Check if results exist"""
    experiment_name = f"{dataset}_{model.lower()}"
    
    if stage == 1:
        results_file = f"hpo/results/stage1/{experiment_name}/best_params.json"
    else:
        results_file = f"hpo/results/stage2/{experiment_name}/calibrated_quantiles.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data
    return None

def main():
    parser = argparse.ArgumentParser(description='Check HPO experiment status')
    parser.add_argument('--model', type=str,
                        choices=['NHITS_Q', 'TIMESNET_Q'],
                        help='Filter by model')
    parser.add_argument('--dataset', type=str,
                        choices=['heat', 'water'],
                        help='Filter by dataset')
    parser.add_argument('--stage', type=int, choices=[1, 2],
                        help='Filter by stage')
    
    args = parser.parse_args()
    
    # Load tracking data
    tracking_file = "hpo/tracking/experiments.csv"
    if not os.path.exists(tracking_file):
        print("No experiments tracked yet.")
        print("\nSubmit experiments with:")
        print("  python hpo/submit_experiment.py --stage 1 --model NHITS_Q --dataset water")
        return
    
    df = pd.read_csv(tracking_file)
    
    # Apply filters
    if args.model:
        df = df[df['model'] == args.model]
    if args.dataset:
        df = df[df['dataset'] == args.dataset]
    if args.stage:
        df = df[df['stage'] == args.stage]
    
    if len(df) == 0:
        print("No experiments match filters.")
        return
    
    print("="*80)
    print("HPO EXPERIMENT STATUS")
    print("="*80)
    
    # Update status from SLURM
    for idx, row in df.iterrows():
        job_id = row['job_id']
        slurm_status = get_slurm_status(job_id)
        df.at[idx, 'slurm_status'] = slurm_status
        
        # Check for results
        results = check_results(row['stage'], row['model'], row['dataset'])
        if results:
            df.at[idx, 'status'] = 'completed'
            if row['stage'] == 1:
                df.at[idx, 'mae'] = results.get('best_mae')
            else:
                df.at[idx, 'picp'] = results.get('achieved_picp')
    
    # Summary by stage
    for stage in sorted(df['stage'].unique()):
        stage_df = df[df['stage'] == stage]
        print(f"\nSTAGE {stage}: {'ARCHITECTURE' if stage == 1 else 'CALIBRATION'}")
        print("-"*80)
        
        for _, row in stage_df.iterrows():
            status_emoji = {
                'RUNNING': '🏃',
                'PENDING': '⏳',
                'COMPLETED': '✓',
                'FAILED': '✗',
                'COMPLETED/FAILED': '?'
            }.get(row['slurm_status'], '?')
            
            print(f"\n{status_emoji} {row['dataset'].upper()} - {row['model']}")
            print(f"  Job ID: {row['job_id']}")
            print(f"  Status: {row['slurm_status']}")
            print(f"  Trials: {row['trials']}")
            print(f"  Submitted: {row['submit_time']}")
            
            if row['status'] == 'completed':
                if stage == 1 and pd.notna(row['mae']):
                    print(f"  MAE: {row['mae']:.6f}")
                elif stage == 2 and pd.notna(row['picp']):
                    print(f"  PICP: {row['picp']:.2f}% (Target: 80%)")
                    print(f"  Error: {abs(row['picp'] - 80):.2f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count by status
    total = len(df)
    completed = len(df[df['status'] == 'completed'])
    running = len(df[df['slurm_status'] == 'RUNNING'])
    pending = len(df[df['slurm_status'] == 'PENDING'])
    
    print(f"Total experiments: {total}")
    print(f"Completed: {completed}")
    print(f"Running: {running}")
    print(f"Pending: {pending}")
    
    # Stage 1 completion
    stage1_complete = []
    for dataset in ['heat', 'water']:
        for model in ['NHITS_Q', 'TIMESNET_Q']:
            if check_results(1, model, dataset):
                stage1_complete.append((dataset, model))
    
    print(f"\nStage 1 complete: {len(stage1_complete)}/4")
    if stage1_complete:
        for dataset, model in stage1_complete:
            print(f"  ✓ {dataset.upper()} - {model}")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    incomplete_stage1 = []
    for dataset in ['heat', 'water']:
        for model in ['NHITS_Q', 'TIMESNET_Q']:
            if not check_results(1, model, dataset):
                incomplete_stage1.append((dataset, model))
    
    if incomplete_stage1:
        print("\nComplete Stage 1 for:")
        for dataset, model in incomplete_stage1:
            print(f"  python hpo/submit_experiment.py --stage 1 --model {model} --dataset {dataset}")
    
    ready_for_stage2 = []
    for dataset, model in stage1_complete:
        if not check_results(2, model, dataset):
            ready_for_stage2.append((dataset, model))
    
    if ready_for_stage2:
        print("\nReady for Stage 2:")
        for dataset, model in ready_for_stage2:
            print(f"  python hpo/submit_experiment.py --stage 2 --model {model} --dataset {dataset}")
    
    print("="*80)

if __name__ == "__main__":
    main()
