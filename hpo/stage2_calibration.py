#!/usr/bin/env python3
"""
Stage 2: Calibration Optimization
Tunes quantile levels to achieve PICP ≈ 80% using best architecture from Stage 1.

Usage:
    python stage2_calibration.py --model NHITS_Q --dataset water --trials 20
"""
import os
import sys
import json
import argparse
import optuna
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Darts imports
from darts.models import NHiTSModel
from darts.utils.likelihood_models import QuantileRegression

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss

# Project imports
import model_preprocessing as mp

# Dataset configurations
DATASETS = {
    "heat": {
        "csv_path": "processing/nordbyen_processing/nordbyen_features_engineered.csv",
        "target_col": "heat_consumption",
        "train_end": "2018-12-31 23:00:00",
        "val_end": "2019-12-31 23:00:00",
    },
    "water": {
        "csv_path": "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv",
        "target_col": "water_consumption",
        "train_end": "2018-12-31 23:00:00",
        "val_end": "2019-12-31 23:00:00",
    }
}

def to_naive(ts_str: str):
    """Remove timezone from timestamp"""
    return pd.Timestamp(ts_str).tz_localize(None)

def get_feature_config(dataset: str):
    """Get feature config for dataset"""
    if dataset == "water":
        cfg = mp.water_feature_config()
    else:
        cfg = mp.default_feature_config()
    return cfg

def calculate_picp(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Calculate Prediction Interval Coverage Probability"""
    if len(y_true) == 0:
        return 0.0
    within_interval = (y_true >= y_low) & (y_true <= y_high)
    return np.mean(within_interval) * 100

def calibrate_nhits_q(csv_path: str, arch_params: dict, quantiles: list, dataset: str):
    """Train NHITS with given architecture and quantile levels"""
    cfg = get_feature_config(dataset)
    ds_cfg = DATASETS[dataset]
    
    # Prepare data
    state, t_sc, v_sc, _ = mp.prepare_model_data(
        csv_path,
        to_naive(ds_cfg["train_end"]),
        to_naive(ds_cfg["val_end"]),
        cfg
    )
    
    # Model parameters (architecture from Stage 1, quantiles from trial)
    model_params = {
        "input_chunk_length": 168,
        "output_chunk_length": 24,
        "batch_size": 32,
        "n_epochs": arch_params.get("n_epochs", 10),
        "random_state": 42,
        "force_reset": True,
        "likelihood": QuantileRegression(quantiles=quantiles),
        "optimizer_kwargs": {
            "lr": arch_params["lr"],
            "weight_decay": arch_params["weight_decay"]
        },
        "num_stacks": arch_params["num_stacks"],
        "num_blocks": arch_params["num_blocks"],
        "num_layers": arch_params["num_layers"],
        "layer_widths": arch_params["layer_widths"],
        "dropout": arch_params["dropout"],
    }
    
    model = NHiTSModel(**model_params)
    
    # Prepare covariates
    tp = t_sc["past_covariates"]
    vp = v_sc["past_covariates"]
    if t_sc["future_covariates"]:
        tp = tp.stack(t_sc["future_covariates"])
    if v_sc["future_covariates"]:
        vp = vp.stack(v_sc["future_covariates"])
    
    # Train
    model.fit(
        t_sc["target"],
        past_covariates=tp,
        val_series=v_sc["target"],
        val_past_covariates=vp
    )
    
    # Predict quantiles explicitly to ensure correct shapes
    q_low, q_med, q_high = quantiles
    # Predict with sampling and compute empirical quantiles
    pred = model.predict(
        n=24,
        series=t_sc["target"],
        past_covariates=tp,
        num_samples=200
    )
    vals = pred.values()
    # vals shape: (24, num_samples)
    y_low = np.quantile(vals, q_low, axis=1)
    y_high = np.quantile(vals, q_high, axis=1)
    y_true = v_sc["target"][:24].values().flatten()
    
    # Calculate PICP
    picp = calculate_picp(y_true, y_low, y_high)
    
    return picp

def calibrate_timesnet_q(csv_path: str, arch_params: dict, quantile_levels: list, dataset: str):
    """Train TimesNet with given architecture and quantile levels"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        cfg = get_feature_config(dataset)
        ds_cfg = DATASETS[dataset]
        
        # Load and split raw DataFrame (TimesNet uses pandas input)
        df_full = mp.load_and_validate_features(csv_path, cfg)
        train_df, val_df, _ = mp.split_by_time(
            df_full,
            to_naive(ds_cfg["train_end"]),
            to_naive(ds_cfg["val_end"])
        )
        
        # Prepare NeuralForecast format: reset index to get 'timestamp' as column, rename to 'ds'
        train_df = train_df.reset_index().copy()
        val_df = val_df.reset_index().copy()
        
        # Select only required columns: ds, y, and numeric features (exclude string columns with NaNs)
        feature_cols = [c for c in train_df.columns if c not in [cfg.time_col, cfg.target_col, 'public_holiday_name', 'school_holiday_name']]
        train_df = train_df[[cfg.time_col, cfg.target_col] + feature_cols].copy()
        val_df = val_df[[cfg.time_col, cfg.target_col] + feature_cols].copy()
        
        # Drop any remaining NaN rows
        train_df = train_df.dropna()
        val_df = val_df.dropna()
        
        # For Stage 2, use minimal setup - only ds, unique_id, y (no exogenous features)
        train_df.rename(columns={cfg.time_col: "ds", cfg.target_col: "y"}, inplace=True)
        val_df.rename(columns={cfg.time_col: "ds", cfg.target_col: "y"}, inplace=True)
        train_df["unique_id"] = "series_1"
        val_df["unique_id"] = "series_1"
        
        train_df_minimal = train_df[["unique_id", "ds", "y"]].copy()
        val_df_minimal = val_df[["unique_id", "ds", "y"]].copy()
        
        # Calculate max_steps based on epochs (same formula as benchmarker.py)
        batch_size = 32
        n_epochs_desired = 10  # Keep low for HPO speed
        n_samples = len(train_df_minimal)
        steps_per_epoch = max(1, n_samples // batch_size)
        max_steps_calculated = n_epochs_desired * steps_per_epoch
        
        # Model configuration (architecture from Stage 1, quantiles from trial)
        # Convert [0.1, 0.5, 0.9] to [10, 50, 90] for NeuralForecast
        levels = [int(q * 100) for q in quantile_levels]
        
        model = TimesNet(
            h=24,
            input_size=168,
            loss=MQLoss(level=levels),
            max_steps=max_steps_calculated,
            hidden_size=arch_params["hidden_size"],
            conv_hidden_size=arch_params["conv_hidden_size"],
            top_k=arch_params["top_k"],
            learning_rate=arch_params["lr"],
            dropout=arch_params["dropout"],
            random_seed=42,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        nf = NeuralForecast(models=[model], freq='1h')
        nf.fit(df=train_df_minimal)
        
        # Evaluate on validation set (first 24 hours)
        val_df_h = val_df_minimal.head(24).reset_index(drop=True)
        preds = nf.predict(df=train_df_minimal.tail(168), futr_df=None)
        
        # Extract quantile predictions
        low_col = [c for c in preds.columns if f'-{levels[0]}' in c][0]
        high_col = [c for c in preds.columns if f'-{levels[2]}' in c][0]
        
        y_low = preds[low_col].values
        y_high = preds[high_col].values
        y_true = val_df_h['y'].values
        
        # Calculate PICP
        picp = calculate_picp(y_true, y_low, y_high)
        
        return picp
        
    except Exception as e:
        print(f"TimesNet Stage 2 calibration failed: {type(e).__name__}: {str(e)[:200]}")
        raise

def objective(trial, model: str, dataset: str, csv_path: str, arch_params: dict):
    """Optuna objective function - minimize distance from PICP=80%"""
    
    # Suggest quantile levels (low and high, median always 0.5)
    # Suggest quantile offset (symmetric around 0.5)
    # q_offset is the distance from 0.5, so quantiles are [0.5-offset, 0.5, 0.5+offset]
    q_offset = trial.suggest_float("q_offset", 0.01, 0.49)
    
    q_low = 0.5 - q_offset
    q_high = 0.5 + q_offset
    
    quantiles = [q_low, 0.5, q_high]
    
    # Train and evaluate
    if model == "NHITS_Q":
        picp = calibrate_nhits_q(csv_path, arch_params, quantiles, dataset)
    elif model == "TIMESNET_Q":
        picp = calibrate_timesnet_q(csv_path, arch_params, quantiles, dataset)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Objective: minimize distance from target PICP (80%)
    calibration_error = abs(picp - 80.0)
    
    return calibration_error

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Calibration Optimization')
    parser.add_argument('--model', type=str, required=True,
                        choices=['NHITS_Q', 'TIMESNET_Q'],
                        help='Model to calibrate')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['heat', 'water'],
                        help='Dataset to use')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (1 trial)')
    
    args = parser.parse_args()
    
    # Load Stage 1 results
    experiment_name = f"{args.dataset}_{args.model.lower()}"
    stage1_dir = os.path.join(project_root, "hpo", "results", "stage1", experiment_name)
    stage1_params_file = os.path.join(stage1_dir, "best_params.json")
    
    if not os.path.exists(stage1_params_file):
        print(f"ERROR: Stage 1 results not found: {stage1_params_file}")
        print("Run Stage 1 first:")
        print(f"  python hpo/stage1_architecture.py --model {args.model} --dataset {args.dataset}")
        sys.exit(1)
    
    with open(stage1_params_file, 'r') as f:
        stage1_results = json.load(f)
    
    arch_params = stage1_results['best_params']

    # Speed up test mode by reducing epochs
    if args.test:
        arch_params['n_epochs'] = 2
    
    # Get dataset configuration
    ds_cfg = DATASETS[args.dataset]
    csv_path = os.path.join(project_root, ds_cfg["csv_path"])
    
    # Create results directory
    results_dir = os.path.join(project_root, "hpo", "results", "stage2", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup Optuna study
    study_name = f"stage2_{experiment_name}"
    storage_path = f"sqlite:///{results_dir}/study.db"
    
    print("="*80)
    print("STAGE 2: CALIBRATION OPTIMIZATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Objective: Minimize |PICP - 80|")
    print(f"Trials: {args.trials if not args.test else 1}")
    print(f"Results: {results_dir}")
    print(f"Stage 1 MAE: {stage1_results['best_mae']:.6f}")
    print("="*80)
    print("Architecture (from Stage 1):")
    for key, value in arch_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="minimize",
        load_if_exists=True
    )
    
    # Run optimization
    n_trials = 1 if args.test else args.trials
    study.optimize(
        lambda trial: objective(trial, args.model, args.dataset, csv_path, arch_params),
        n_trials=n_trials
    )
    
    # Compute best quantiles from q_offset (symmetric around 0.5)
    best_offset = study.best_trial.params.get('q_offset')
    if best_offset is None:
        # Backward compatibility: if q_low/q_high existed
        best_q_low = study.best_trial.params.get('q_low')
        best_q_high = study.best_trial.params.get('q_high')
        best_quantiles = [best_q_low, 0.5, best_q_high]
    else:
        best_q_low = 0.5 - best_offset
        best_q_high = 0.5 + best_offset
        best_quantiles = [best_q_low, 0.5, best_q_high]

    # Recompute achieved PICP using best quantiles
    if args.model == 'NHITS_Q':
        achieved_picp = calibrate_nhits_q(csv_path, arch_params, best_quantiles, args.dataset)
    else:
        achieved_picp = calibrate_timesnet_q(csv_path, arch_params, best_quantiles, args.dataset)
    best_picp_error = abs(achieved_picp - 80.0)
    
    # Save results
    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    print(f"Best PICP: {achieved_picp:.2f}% (Target: 80%)")
    print(f"Calibration Error: {best_picp_error:.2f}%")
    print(f"Best quantiles: {best_quantiles}")
    print(f"  Lower: {best_q_low:.4f}")
    print(f"  Median: 0.5000")
    print(f"  Upper: {best_q_high:.4f}")
    
    # Save calibrated quantiles
    calibration_file = os.path.join(results_dir, "calibrated_quantiles.json")
    result_data = {
        "stage": 2,
        "model": args.model,
        "dataset": args.dataset,
        "objective": "PICP_calibration",
        "target_picp": 80.0,
        "achieved_picp": achieved_picp,
        "calibration_error": best_picp_error,
        "calibrated_quantiles": best_quantiles,
        "architecture_params": arch_params,
        "stage1_mae": stage1_results['best_mae'],
        "n_trials": len(study.trials),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(calibration_file, "w") as f:
        json.dump(result_data, f, indent=4)
    
    print(f"\n✓ Results saved to: {calibration_file}")
    print("="*80)
    print("\nNext step: Run final benchmark with optimized parameters")
    print(f"  The benchmarker will use architecture from Stage 1")
    print(f"  and quantiles: {best_quantiles}")
    print("="*80)

if __name__ == "__main__":
    main()
