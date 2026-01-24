#!/usr/bin/env python3
"""
Multi-Objective HPO Runner for Time Series Forecasting Models

Optimizes both:
  1. MAE (Mean Absolute Error) - deterministic accuracy
  2. PICP (Prediction Interval Coverage Probability) - probabilistic calibration

Usage:
  python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 50
  python hpo/run_hpo.py --model TFT_Q --dataset water_centrum --trials 30
  
Environment Variables:
  SLURM_JOB_ID: Auto-detected for result naming
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import pickle

# Optuna
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_pareto_front
)

# Darts
from darts.models import NHiTSModel, TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts import TimeSeries

# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp
from hpo.hpo_config import (
    get_nhits_search_space,
    get_tft_search_space,
    get_timesnet_search_space,
    DATASET_PATHS,
    SPLIT_CONFIG,
    HPO_TRAINING_CONFIG,
    HPO_TRAINING_CONFIG_TFT
)

# Global overrides for testing (set via command line args)
N_EPOCHS_OVERRIDE = None
N_STEPS_OVERRIDE = None


def to_naive(ts_str: str) -> pd.Timestamp:
    """Convert timezone-aware timestamp to naive."""
    return pd.Timestamp(ts_str).tz_localize(None)


def calculate_picp(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Calculate Prediction Interval Coverage Probability."""
    if len(y_true) == 0:
        return 0.0
    within_interval = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(within_interval) * 100)


def evaluate_model_walk_forward(
    model, 
    val_data: Dict[str, TimeSeries],
    state: mp.PreprocessingState,
    n_steps: int = 10,
    is_tft: bool = False,
    model_type: str = "nhits",
    csv_path: str = None
) -> Tuple[float, float]:
    """
    Walk-forward evaluation on validation set.
    
    Returns:
        (mae, picp): Mean Absolute Error and Prediction Interval Coverage Probability
    """
    val_target = val_data["target"]
    val_past = val_data["past_covariates"]
    val_future = val_data.get("future_covariates", None)
    
    # Load raw dataframe for getting unscaled actuals (like benchmarker does)
    df_full = None
    if csv_path:
        df_full = mp.load_and_validate_features(csv_path)
        df_full.index = df_full.index.tz_localize(None)
    
    all_errors = []
    all_actuals = []
    all_p10 = []
    all_p90 = []
    
    # Get minimum input chunk length (NHiTS needs 168 hours)
    input_chunk = 168  # Fixed value, could be from config if needed
    
    # Walk forward through validation set
    for i in range(n_steps):
        # Each step predicts the next 24 hours
        # Start with enough history for the model (at least input_chunk hours)
        hist_end = input_chunk + i * 24  # Growing history window
        pred_start = hist_end
        pred_end = pred_start + 24
        
        # Check if we have enough data
        if pred_end > len(val_target):
            break
        
        # Historical data: from start up to hist_end
        hist_target = val_target[:hist_end]
        
        # Predict next 24 hours
        try:
            if is_tft:
                # TFT needs separate past and future covariates
                hist_past = val_past[:hist_end] if val_past else None
                
                # TFT needs future covariates from START of history to END of prediction window
                # NOT just the prediction window itself!
                if val_future is not None:
                    try:
                        # Slice from beginning to end of prediction (not just pred_start:pred_end)
                        fut_cov = val_future[:pred_end]
                        # Verify we have enough future covariates
                        if len(fut_cov) < pred_end:
                            print(f"  Warning: Not enough future covariates at step {i}: needed {pred_end}, got {len(fut_cov)}")
                            fut_cov = None
                    except Exception as slice_err:
                        print(f"  Warning: Failed to slice future_covariates at step {i}: {slice_err}")
                        fut_cov = None
                else:
                    fut_cov = None
                
                preds = model.predict(
                    n=24,
                    series=hist_target,
                    past_covariates=hist_past,
                    future_covariates=fut_cov,
                    num_samples=100
                )
            elif model_type == "nhits":
                # NHiTS: Stack past and future covariates THEN slice to history
                hist_cov = val_past[:hist_end].stack(val_future[:hist_end]) if val_future else val_past[:hist_end]
                
                preds = model.predict(
                    n=24,
                    series=hist_target,
                    past_covariates=hist_cov,
                    num_samples=100
                )
            else:
                # TimesNet or other: use pre-stacked covariates directly
                hist_cov = val_past[:hist_end]
                
                preds = model.predict(
                    n=24,
                    series=hist_target,
                    past_covariates=hist_cov,
                    num_samples=100
                )
            
            # Inverse transform predictions
            preds_original = state.target_scaler.inverse_transform(preds)
            
            # Extract quantiles DIRECTLY from TimeSeries (before calling .values())
            # This ensures quantiles are computed across samples correctly
            p10 = preds_original.quantile(0.1).values().flatten()
            p50 = preds_original.quantile(0.5).values().flatten()
            p90 = preds_original.quantile(0.9).values().flatten()
            
            # Validate quantile results
            if np.isnan(p50).any() or np.isinf(p50).any():
                print(f"  Warning: Quantiles contain NaN/Inf at step {i}")
                continue
            
            # Get actual values for the prediction window
            # Use raw dataframe if available (like benchmarker), otherwise inverse transform scaled data
            if df_full is not None:
                # Get prediction start time from validation target
                pred_time_start = val_target.time_index[pred_start]
                pred_time_end = val_target.time_index[pred_end - 1]
                actuals = df_full[state.feature_config.target_col][pred_time_start:pred_time_end].values
            else:
                # Fallback: inverse transform scaled validation data
                actuals_scaled = val_target[pred_start:pred_end]
                actuals = state.target_scaler.inverse_transform(actuals_scaled).values().flatten()
            
            # Check for NaN/Inf in actuals
            if np.isnan(actuals).any() or np.isinf(actuals).any():
                print(f"  Warning: Actuals contain NaN/Inf at step {i}")
                continue
            
            # Compute errors
            errors = np.abs(actuals - p50)
            
            # Check for NaN/Inf in errors
            if np.isnan(errors).any() or np.isinf(errors).any():
                print(f"  Warning: Errors contain NaN/Inf at step {i}")
                continue
            
            # Store results
            all_errors.extend(errors)
            all_actuals.extend(actuals)
            all_p10.extend(p10)
            all_p90.extend(p90)
            
        except Exception as e:
            print(f"  Warning: Prediction failed at step {i}: {type(e).__name__}: {e}")
            continue
    
    if len(all_errors) == 0:
        print(f"  ERROR: No valid predictions generated! Returning inf MAE.")
        return float('inf'), 0.0
    
    mae = float(np.mean(all_errors))
    picp = calculate_picp(
        np.array(all_actuals),
        np.array(all_p10),
        np.array(all_p90)
    )
    
    # Final validation of results
    if np.isnan(mae) or np.isinf(mae):
        print(f"  ERROR: Final MAE is NaN/Inf: {mae}")
        return float('inf'), 0.0
    
    return mae, picp


def train_nhits(params: Dict[str, Any], csv_path: str, dataset: str) -> Tuple[float, float]:
    """Train and evaluate NHiTS model."""
    print(f"\n  Training NHiTS with params: {params}")
    
    # Auto-detect feature config based on dataset name
    cfg = mp.water_feature_config() if 'water' in dataset.lower() else mp.default_feature_config()
    
    # Get dataset-specific split config
    split_cfg = SPLIT_CONFIG[dataset]
    
    # Prepare data
    state, t_sc, v_sc, _ = mp.prepare_model_data(
        csv_path,
        to_naive(split_cfg["train_end"]),
        to_naive(split_cfg["val_end"]),
        cfg
    )
    
    # Model configuration
    model_config = {
        **HPO_TRAINING_CONFIG,
        "random_state": 42,
        "force_reset": True,
        "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        "num_stacks": params["num_stacks"],
        "num_blocks": params["num_blocks"],
        "num_layers": params["num_layers"],
        "layer_widths": params["layer_widths"],
        "dropout": params["dropout"],
        "optimizer_kwargs": {
            "lr": params["lr"],
            "weight_decay": params["weight_decay"]
        },
        "pl_trainer_kwargs": {
            "logger": False,  # Disable logging for HPO
            "enable_checkpointing": False,
        }
    }
    
    # Prepare covariates (stack past and future)
    tp = t_sc["past_covariates"]
    if t_sc["future_covariates"]:
        tp = tp.stack(t_sc["future_covariates"])
    
    vp = v_sc["past_covariates"]
    if v_sc["future_covariates"]:
        vp = vp.stack(v_sc["future_covariates"])
    
    # Train model
    model = NHiTSModel(**model_config)
    model.fit(
        t_sc["target"],
        past_covariates=tp,
        val_series=v_sc["target"],
        val_past_covariates=vp
    )
    
    # Prepare validation data for evaluation with stacked covariates
    v_sc_eval = {
        "target": v_sc["target"],
        "past_covariates": v_sc["past_covariates"],
        "future_covariates": v_sc["future_covariates"]
    }
    
    # Evaluate (use override if set)
    n_eval_steps = N_STEPS_OVERRIDE if N_STEPS_OVERRIDE is not None else 10
    mae, picp = evaluate_model_walk_forward(model, v_sc_eval, state, n_steps=n_eval_steps, is_tft=False, model_type="nhits", csv_path=csv_path)
    
    print(f"  Results: MAE={mae:.4f}, PICP={picp:.2f}%")
    return mae, picp


def train_tft(params: Dict[str, Any], csv_path: str, dataset: str) -> Tuple[float, float]:
    """Train and evaluate TFT model (optimized for speed)."""
    print(f"\n  Training TFT with params: {params}")
    
    try:
        # Auto-detect feature config based on dataset name
        cfg = mp.water_feature_config() if 'water' in dataset.lower() else mp.default_feature_config()
        
        # Get dataset-specific split config
        split_cfg = SPLIT_CONFIG[dataset]
        
        # Prepare data
        state, t_sc, v_sc, _ = mp.prepare_model_data(
            csv_path,
            to_naive(split_cfg["train_end"]),
            to_naive(split_cfg["val_end"]),
            cfg
        )
        
        print(f"  Data shapes - Train target: {t_sc['target'].shape}, Val target: {v_sc['target'].shape}")
        
        # Model configuration - Use TFT-specific faster config (use override if set)
        n_epochs = N_EPOCHS_OVERRIDE if N_EPOCHS_OVERRIDE is not None else HPO_TRAINING_CONFIG_TFT["n_epochs"]
        
        model_config = {
            **HPO_TRAINING_CONFIG_TFT,  # Use faster config for TFT
            "n_epochs": n_epochs,  # Apply override
            "random_state": 42,
            "force_reset": True,
            "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            "hidden_size": params["hidden_size"],
            "lstm_layers": params["lstm_layers"],
            "num_attention_heads": params["num_attention_heads"],
            "dropout": params["dropout"],
            "optimizer_kwargs": {"lr": params["lr"]},
            "pl_trainer_kwargs": {
                "logger": False,
                "enable_checkpointing": False,
            }
        }
        
        # TFT uses separate past and future covariates
        tp = t_sc["past_covariates"]
        tf = t_sc["future_covariates"]
        vp = v_sc["past_covariates"]
        vf = v_sc["future_covariates"]
        
        print(f"  Covariate shapes - tp: {tp.shape if tp else None}, tf: {tf.shape if tf else None}")
        print(f"                      vp: {vp.shape if vp else None}, vf: {vf.shape if vf else None}")
        
        # Train model
        print(f"  Training TFT model...")
        model = TFTModel(**model_config)
        model.fit(
            t_sc["target"],
            past_covariates=tp,
            future_covariates=tf,
            val_series=v_sc["target"],
            val_past_covariates=vp,
            val_future_covariates=vf
        )
        print(f"  TFT training completed successfully")
        
        # Prepare validation data for evaluation
        v_sc_eval = {
            "target": v_sc["target"],
            "past_covariates": vp,
            "future_covariates": vf
        }
        
        # Evaluate with fewer steps for speed (use override if set)
        n_eval_steps = N_STEPS_OVERRIDE if N_STEPS_OVERRIDE is not None else 5
        print(f"  Evaluating TFT model on validation set...")
        mae, picp = evaluate_model_walk_forward(model, v_sc_eval, state, n_steps=n_eval_steps, is_tft=True, model_type="tft", csv_path=csv_path)
        
        print(f"  Results: MAE={mae:.6f}, PICP={picp:.2f}%")
        return mae, picp
        
    except Exception as e:
        print(f"  ERROR in train_tft: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0


def train_timesnet(params: Dict[str, Any], csv_path: str, dataset: str) -> Tuple[float, float]:
    """Train and evaluate TimesNet model."""
    print(f"\n  Training TimesNet with params: {params}")
    
    # Load and prepare data
    df_full = mp.load_and_validate_features(csv_path)
    
    # Determine target column based on dataset name
    target_col = "water_consumption" if 'water' in dataset.lower() else "heat_consumption"
    
    # Convert to NeuralForecast format
    nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", target_col: "y"})
    # Use dataset name for unique_id
    unique_id = dataset.replace('_', '-')  # water_centrum -> water-centrum
    nf_df["unique_id"] = unique_id
    
    # Get feature config based on dataset name
    cfg = mp.water_feature_config() if 'water' in dataset.lower() else mp.default_feature_config()
    
    # Prepare exogenous features (TimesNet treats all as future-known)
    futr_ex = [c for c in (cfg.past_covariates_cols + cfg.future_covariates_cols)
               if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
    
    nf_df = nf_df[["unique_id", "ds", "y"] + futr_ex].dropna()
    nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
    
    # Get dataset-specific split config
    split_cfg = SPLIT_CONFIG[dataset]
    
    # Split data
    train_df = nf_df[nf_df['ds'] <= to_naive(split_cfg["train_end"])].reset_index(drop=True)
    val_df = nf_df[(nf_df['ds'] > to_naive(split_cfg["train_end"])) & 
                   (nf_df['ds'] <= to_naive(split_cfg["val_end"]))].reset_index(drop=True)
    
    # Calculate max_steps for NeuralForecast
    n_samples = len(train_df)
    steps_per_epoch = max(1, n_samples // HPO_TRAINING_CONFIG["batch_size"])
    max_steps = HPO_TRAINING_CONFIG["n_epochs"] * steps_per_epoch
    
    # Model configuration
    model = TimesNet(
        h=HPO_TRAINING_CONFIG["output_chunk_length"],
        input_size=HPO_TRAINING_CONFIG["input_chunk_length"],
        futr_exog_list=futr_ex,
        scaler_type="robust",
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        max_steps=max_steps,
        batch_size=HPO_TRAINING_CONFIG["batch_size"],
        hidden_size=params["hidden_size"],
        conv_hidden_size=params["conv_hidden_size"],
        top_k=params["top_k"],
        learning_rate=params["lr"],
        dropout=params["dropout"],
        logger=False,
        enable_checkpointing=False
    )
    
    # Train
    nf = NeuralForecast(models=[model], freq='h')
    nf.fit(df=train_df)
    
    # Evaluate on validation set (walk-forward)
    all_errors = []
    all_actuals = []
    all_p10 = []
    all_p90 = []
    
    n_steps = min(10, len(val_df) // 24)
    for i in range(n_steps):
        start_idx = i * 24
        hist_end = len(train_df) + start_idx
        
        hist_df = nf_df.iloc[:hist_end].tail(168).reset_index(drop=True)
        fut_df = val_df.iloc[start_idx:start_idx+24].reset_index(drop=True)
        
        if len(fut_df) < 24:
            break
        
        try:
            fcst = nf.predict(df=hist_df, futr_df=fut_df[["unique_id", "ds"] + futr_ex])
            
            # Extract quantiles
            def find_col(suffixes):
                for s in suffixes:
                    for c in fcst.columns:
                        if c.endswith(s):
                            return c
                return None
            
            cm = find_col(['median', '-q-0.5'])
            cl = find_col(['-lo-80.0', '-q-0.1'])
            ch = find_col(['-hi-80.0', '-q-0.9'])
            
            p50 = fcst[cm].values if cm else fcst.iloc[:, -1].values
            p10 = fcst[cl].values if cl else p50
            p90 = fcst[ch].values if ch else p50
            
            actuals = fut_df['y'].values
            
            all_errors.extend(np.abs(actuals - p50))
            all_actuals.extend(actuals)
            all_p10.extend(p10)
            all_p90.extend(p90)
            
        except Exception as e:
            print(f"  Warning: Prediction failed at step {i}: {e}")
            continue
    
    if len(all_errors) == 0:
        return float('inf'), 0.0
    
    mae = float(np.mean(all_errors))
    picp = calculate_picp(
        np.array(all_actuals),
        np.array(all_p10),
        np.array(all_p90)
    )
    
    print(f"  Results: MAE={mae:.4f}, PICP={picp:.2f}%")
    return mae, picp


def objective(trial, model_name: str, dataset: str) -> Tuple[float, float]:
    """
    Multi-objective optimization function.
    
    Returns:
        (mae, picp_penalty): We minimize both objectives
        - Minimize MAE (lower is better)
        - Minimize picp_penalty (deviation from 80% target)
    """
    # Get CSV path
    csv_path = DATASET_PATHS[dataset]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    # Get hyperparameters based on model
    if model_name == "NHITS_Q":
        params = get_nhits_search_space(trial)
        mae, picp = train_nhits(params, csv_path, dataset)
    elif model_name == "TFT_Q":
        params = get_tft_search_space(trial)
        mae, picp = train_tft(params, csv_path, dataset)
    elif model_name == "TIMESNET_Q":
        params = get_timesnet_search_space(trial)
        mae, picp = train_timesnet(params, csv_path, dataset)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Multi-objective: minimize MAE, minimize deviation from 80% PICP
    picp_penalty = abs(picp - 80.0)  # 0 is perfect, higher is worse
    
    return mae, picp_penalty


def run_optimization(model_name: str, dataset: str, n_trials: int, job_id: str = None):
    """Run multi-objective hyperparameter optimization."""
    
    # Get job ID from environment or use provided
    if job_id is None:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # For TFT, reduce trials and use faster settings (TFT is much slower)
    if model_name == "TFT_Q" and n_trials > 20:
        print(f"\n⚠️  TFT is slow - reducing trials from {n_trials} to 20 for faster completion")
        n_trials = 20
    
    # Setup result directory
    result_dir = f"hpo/results/{model_name}_{dataset}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Create study name
    study_name = f"{model_name}_{dataset}_{job_id}"
    storage_path = f"sqlite:///{result_dir}/study_{model_name}_{dataset}_{job_id}.db"
    
    print(f"\n{'='*70}")
    print(f"STARTING HPO OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Model:       {model_name}")
    print(f"Dataset:     {dataset}")
    print(f"Trials:      {n_trials}")
    print(f"Job ID:      {job_id}")
    print(f"Study Name:  {study_name}")
    print(f"Results Dir: {result_dir}")
    print(f"{'='*70}\n")
    
    # Create multi-objective study with model-specific pruner
    if model_name == "TFT_Q":
        # More aggressive pruning for TFT to fail fast on bad hyperparams
        pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        directions=["minimize", "minimize"],  # Minimize MAE and PICP penalty
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, model_name, dataset),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get Pareto-optimal trials
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto-optimal solutions: {len(study.best_trials)}")
    
    # Find best trial (lowest MAE among Pareto front)
    best_trial = min(study.best_trials, key=lambda t: t.values[0])
    best_mae = best_trial.values[0]
    best_picp_penalty = best_trial.values[1]
    best_picp = 80.0 - best_picp_penalty if best_picp_penalty <= 80.0 else 80.0 + best_picp_penalty
    
    print(f"\nBest Solution (lowest MAE on Pareto front):")
    print(f"  MAE:  {best_mae:.6f}")
    print(f"  PICP: ~{best_picp:.2f}%")
    print(f"  Trial: {best_trial.number}")
    print(f"\nBest Parameters:")
    for param, value in best_trial.params.items():
        print(f"  {param}: {value}")
    
    # Save results
    result_file = f"{result_dir}/best_params_{model_name}_{dataset}_{job_id}.json"
    result_data = {
        "model": model_name,
        "dataset": dataset,
        "job_id": job_id,
        "optimization_date": datetime.now().isoformat(),
        "n_trials": n_trials,
        "n_completed": len(study.trials),
        "n_pareto_optimal": len(study.best_trials),
        "best_trial_number": best_trial.number,
        "best_mae": best_mae,
        "best_picp_approx": best_picp,
        "best_params": best_trial.params,
        "study_name": study_name,
        "storage_path": storage_path,
        "all_pareto_trials": [
            {
                "trial": t.number,
                "mae": t.values[0],
                "picp_penalty": t.values[1],
                "params": t.params
            }
            for t in study.best_trials
        ]
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")
    
    # Generate visualizations
    try:
        print(f"\nGenerating visualizations...")
        
        # Pareto front
        fig = plot_pareto_front(study, target_names=["MAE", "PICP Penalty"])
        fig.write_html(f"{result_dir}/pareto_front_{job_id}.html")
        
        # Optimization history
        fig = plot_optimization_history(study, target=lambda t: t.values[0], target_name="MAE")
        fig.write_html(f"{result_dir}/mae_history_{job_id}.html")
        
        # Parameter importances (for MAE objective)
        fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="MAE")
        fig.write_html(f"{result_dir}/param_importance_{job_id}.html")
        
        print(f"✅ Visualizations saved to: {result_dir}/")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"HPO COMPLETE - {model_name} on {dataset}")
    print(f"{'='*70}\n")
    
    return result_file


def main():
    parser = argparse.ArgumentParser(description="Multi-Objective HPO for Time Series Models")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["NHITS_Q", "TFT_Q", "TIMESNET_Q"],
                        help="Model to optimize")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["heat", "water_centrum", "water_tommerby"],
                        help="Dataset to use")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--job-id", type=str, default=None,
                        help="Job ID for result naming (auto-detected from SLURM if not provided)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override number of training epochs (for testing)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override number of walk-forward validation steps (for testing)")
    
    args = parser.parse_args()
    
    # Set global overrides if provided (for testing)
    if args.n_epochs is not None:
        global N_EPOCHS_OVERRIDE
        N_EPOCHS_OVERRIDE = args.n_epochs
        print(f"⚠️  TESTING MODE: Using {args.n_epochs} epochs instead of default")
    
    if args.n_steps is not None:
        global N_STEPS_OVERRIDE
        N_STEPS_OVERRIDE = args.n_steps
        print(f"⚠️  TESTING MODE: Using {args.n_steps} validation steps instead of default")
    
    # Run optimization
    result_file = run_optimization(
        model_name=args.model,
        dataset=args.dataset,
        n_trials=args.trials,
        job_id=args.job_id
    )
    
    print(f"\n✅ HPO completed successfully!")
    print(f"📁 Results: {result_file}")


if __name__ == "__main__":
    main()
