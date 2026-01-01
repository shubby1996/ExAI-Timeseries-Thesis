#!/usr/bin/env python3
"""
Submit HPO experiments to SLURM queue.
Handles both Stage 1 (architecture) and Stage 2 (calibration) optimization.

Usage:
    python hpo/submit_experiment.py --stage 1 --model NHITS_Q --dataset water --trials 50
    python hpo/submit_experiment.py --stage 2 --model TIMESNET_Q --dataset heat --trials 20
    python hpo/submit_experiment.py --stage all --priority  # Submit all experiments in priority order
"""
import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

# Experiment priorities (from HPO_STRATEGY.md)
PRIORITY_ORDER = [
    ("water", "TIMESNET_Q"),  # Best overall performer
    ("heat", "NHITS_Q"),      # Best heat model, worst PICP
    ("water", "NHITS_Q"),     # Fast training
    ("heat", "TIMESNET_Q"),   # Architecture comparison
]

# Default trial counts
DEFAULT_STAGE1_TRIALS = 50
DEFAULT_STAGE2_TRIALS = 20

# SLURM scripts
SLURM_SCRIPTS = {
    1: "hpo/run_stage1.slurm",
    2: "hpo/run_stage2.slurm"
}

def load_tracking():
    """Load experiment tracking file"""
    tracking_file = "hpo/tracking/experiments.csv"
    if os.path.exists(tracking_file):
        import pandas as pd
        return pd.read_csv(tracking_file)
    return None

def save_tracking(data):
    """Save experiment tracking data"""
    import pandas as pd
    tracking_file = "hpo/tracking/experiments.csv"
    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
    
    if os.path.exists(tracking_file):
        df = pd.read_csv(tracking_file)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    
    df.to_csv(tracking_file, index=False)

def check_stage1_complete(model, dataset):
    """Check if Stage 1 is complete for given model/dataset"""
    stage1_file = f"hpo/results/stage1/{dataset}_{model.lower()}/best_params.json"
    return os.path.exists(stage1_file)

def submit_job(stage, model, dataset, trials):
    """Submit SLURM job"""
    slurm_script = SLURM_SCRIPTS[stage]
    
    if not os.path.exists(slurm_script):
        print(f"ERROR: SLURM script not found: {slurm_script}")
        return None
    
    # Check prerequisites
    if stage == 2 and not check_stage1_complete(model, dataset):
        print(f"ERROR: Stage 1 not complete for {dataset}_{model}. Run Stage 1 first.")
        return None
    
    # Create results directories
    stage_dir = f"hpo/results/stage{stage}"
    os.makedirs(stage_dir, exist_ok=True)
    
    # Submit job
    cmd = [
        "sbatch",
        f"--export=MODEL={model},DATASET={dataset},TRIALS={trials}",
        slurm_script
    ]
    
    print(f"\nSubmitting Stage {stage} job:")
    print(f"  Model: {model}")
    print(f"  Dataset: {dataset}")
    print(f"  Trials: {trials}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"  ✓ Submitted job: {job_id}")
        
        # Track experiment
        tracking_data = {
            "stage": stage,
            "model": model,
            "dataset": dataset,
            "trials": trials,
            "job_id": job_id,
            "status": "submitted",
            "submit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": None,
            "end_time": None,
            "mae": None,
            "picp": None,
            "notes": ""
        }
        save_tracking(tracking_data)
        
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error submitting job: {e}")
        print(f"  stderr: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Submit HPO experiments')
    parser.add_argument('--stage', type=str, required=True,
                        choices=['1', '2', 'all'],
                        help='Stage to run (1=architecture, 2=calibration, all=both in sequence)')
    parser.add_argument('--model', type=str,
                        choices=['NHITS_Q', 'TIMESNET_Q'],
                        help='Model to optimize')
    parser.add_argument('--dataset', type=str,
                        choices=['heat', 'water'],
                        help='Dataset to use')
    parser.add_argument('--trials', type=int,
                        help='Number of trials (default: 50 for stage1, 20 for stage2)')
    parser.add_argument('--priority', action='store_true',
                        help='Submit experiments in priority order (use with --stage all)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HPO EXPERIMENT SUBMISSION")
    print("="*80)
    
    # Handle "all" stage with priority order
    if args.stage == 'all':
        if args.priority:
            print("\nSubmitting experiments in priority order:")
            for i, (dataset, model) in enumerate(PRIORITY_ORDER, 1):
                print(f"\n{i}. {dataset.upper()} - {model}")
                
                # Submit Stage 1
                stage1_trials = args.trials or DEFAULT_STAGE1_TRIALS
                job_id = submit_job(1, model, dataset, stage1_trials)
                
                if job_id:
                    print(f"  Note: Submit Stage 2 after Stage 1 completes (job {job_id})")
            
            print("\n" + "="*80)
            print("Stage 1 jobs submitted. Monitor with:")
            print("  python hpo/check_status.py")
            print("\nSubmit Stage 2 jobs after Stage 1 completes:")
            print("  python hpo/submit_experiment.py --stage 2 --model <MODEL> --dataset <DATASET>")
            print("="*80)
        else:
            print("ERROR: --priority required when using --stage all")
            sys.exit(1)
    
    # Handle single stage
    elif args.stage in ['1', '2']:
        if not args.model or not args.dataset:
            print("ERROR: --model and --dataset required for single stage submission")
            sys.exit(1)
        
        stage = int(args.stage)
        trials = args.trials or (DEFAULT_STAGE1_TRIALS if stage == 1 else DEFAULT_STAGE2_TRIALS)
        
        job_id = submit_job(stage, args.model, args.dataset, trials)
        
        if job_id:
            print("\n" + "="*80)
            print("Job submitted successfully!")
            print(f"Job ID: {job_id}")
            print("\nMonitor progress:")
            print(f"  squeue -j {job_id}")
            print(f"  python hpo/check_status.py")
            print("\nView logs:")
            log_dir = f"hpo/results/stage{stage}"
            print(f"  tail -f {log_dir}/hpo_stage{stage}_{job_id}.log")
            print("="*80)

if __name__ == "__main__":
    main()
