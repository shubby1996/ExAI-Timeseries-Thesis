# HPO Revival Plan for Three _Q Models (REVISED)

**Date:** January 16, 2026  
**Target Models:** NHITS_Q, TIMESNET_Q, TFT_Q  
**Objective:** Optimize hyperparameters using multi-objective approach (MAE + PICP simultaneously)  
**Philosophy:** Fresh start, no two-stage process, clean structure

---

## Design Principles

### ✅ What We Want

1. **Single-Stage Multi-Objective Optimization:**
   - Optimize BOTH deterministic (MAE) AND probabilistic (PICP) metrics together
   - No separate architecture → calibration stages
   - Use Optuna's multi-objective optimization

2. **Clean Result Organization:**
   - **Storage:** `hpo/results/{MODEL}_{DATASET}/`
   - **Format:** `best_params_{MODEL}_{DATASET}_{JOBID}.json`
   - **Example:** `hpo/results/NHITS_Q_heat/best_params_NHITS_Q_heat_1234567.json`
   - Keep HPO results separate from main `results/` folder

3. **Individual Job Submission:**
   - One SLURM job per (model, dataset) combination
   - 9 separate jobs, not batch submission
   - Each job self-contained and independent

4. **Multi-Dataset Support:**
   - Single runner script handles all datasets
   - Auto-detect CSV paths based on dataset name
   - Support: heat, water_centrum, water_tommerby

5. **Fresh Implementation:**
   - Don't rely on old `hpo_tuner.py` structure
   - Build new clean runner from scratch
   - Modern Optuna features (multi-objective, pruning)

### ❌ What We're Avoiding

- ❌ Two-stage optimization (architecture then calibration)
- ❌ Saving results to root `results/` folder
- ❌ Batch submission scripts
- ❌ Focus only on MAE/deterministic metrics
- ❌ Legacy code dependencies

---

## Architecture Overview

### Directory Structure

```
hpo/
├── run_hpo.py                          # Main HPO runner (NEW - fresh implementation)
├── submit_job.sh                       # Single job submission helper (NEW)
├── hpo_config.py                       # Hyperparameter search spaces (NEW)
├── README.md                           # Updated documentation
├── HPO_REVIVAL_PLAN.md                 # This file
├── results/                            # All HPO results stored here
│   ├── NHITS_Q_heat/
│   │   ├── best_params_NHITS_Q_heat_1234567.json
│   │   ├── study_NHITS_Q_heat_1234567.db
│   │   └── optuna_vis_1234567.html
│   ├── NHITS_Q_water_centrum/
│   │   └── best_params_NHITS_Q_water_centrum_1234568.json
│   ├── TFT_Q_heat/
│   │   └── best_params_TFT_Q_heat_1234569.json
│   ├── TIMESNET_Q_heat/
│   │   └── best_params_TIMESNET_Q_heat_1234570.json
│   └── ... (all 9 combinations)
└── logs/                               # SLURM logs
    ├── hpo_NHITS_Q_heat_1234567.log
    ├── hpo_NHITS_Q_heat_1234567.err
    └── ...
```

### File Purposes

| File | Purpose |
|------|---------|
| `run_hpo.py` | Main runner - handles training, evaluation, optimization |
| `hpo_config.py` | Search space definitions for each model |
| `submit_job.sh` | Submit single job: `./submit_job.sh NHITS_Q heat 50` |
| `results/{MODEL}_{DATASET}/` | All results for specific model+dataset combo |

---

## Implementation Plan

### Phase 1: Core Implementation (Priority: HIGH)

#### Task 1.1: Create Search Space Configuration

**Create:** `hpo/hpo_config.py`

```python
"""
Hyperparameter search spaces for all models.
Defines reasonable ranges based on literature and experience.
"""

def get_nhits_search_space(trial):
    """NHITS hyperparameter search space."""
    return {
        "num_stacks": trial.suggest_int("num_stacks", 2, 5),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "layer_widths": trial.suggest_categorical("layer_widths", [256, 512, 1024]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
    }

def get_tft_search_space(trial):
    """TFT hyperparameter search space."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "num_attention_heads": trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }

def get_timesnet_search_space(trial):
    """TimesNet hyperparameter search space."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "conv_hidden_size": trial.suggest_categorical("conv_hidden_size", [32, 64, 128]),
        "top_k": trial.suggest_int("top_k", 2, 5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
    }

# Dataset to CSV path mapping
DATASET_PATHS = {
    "heat": "processing/nordbyen_processing/nordbyen_features_engineered.csv",
    "water_centrum": "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv",
    "water_tommerby": "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv",
}

# Training/validation splits (consistent across all experiments)
SPLIT_CONFIG = {
    "train_end": "2023-10-31 23:00:00+00:00",
    "val_end": "2024-03-31 23:00:00+00:00",
}

# HPO training config (faster than full training)
HPO_TRAINING_CONFIG = {
    "n_epochs": 15,  # Reduced for HPO speed (vs 100 for final training)
    "batch_size": 32,
    "input_chunk_length": 168,
    "output_chunk_length": 24,
}
```

**Key Design Decisions:**
- Search spaces based on literature and prior results
- Logarithmic scale for learning rates (better exploration)
- Reasonable bounds to avoid extreme/unstable configurations
- Faster training (15 epochs) for HPO efficiency

---

#### Task 1.2: Create Main HPO Runner

**Create:** `hpo/run_hpo.py`

```python
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
from hpo_config import (
    get_nhits_search_space,
    get_tft_search_space,
    get_timesnet_search_space,
    DATASET_PATHS,
    SPLIT_CONFIG,
    HPO_TRAINING_CONFIG
)


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
    is_tft: bool = False
) -> Tuple[float, float]:
    """
    Walk-forward evaluation on validation set.
    
    Returns:
        (mae, picp): Mean Absolute Error and Prediction Interval Coverage Probability
    """
    val_target = val_data["target"]
    val_past = val_data["past_covariates"]
    val_future = val_data.get("future_covariates", None)
    
    all_errors = []
    all_actuals = []
    all_p10 = []
    all_p90 = []
    
    # Walk forward through validation set
    for i in range(n_steps):
        start_idx = i * 24  # Each step = 24 hours
        if start_idx + 24 > len(val_target):
            break
        
        # Historical data up to prediction point
        hist_target = val_target[:start_idx] if start_idx > 0 else val_target[:24]
        
        # Predict next 24 hours
        try:
            if is_tft:
                # TFT needs separate past and future covariates
                hist_past = val_past[:start_idx] if start_idx > 0 else val_past[:24]
                fut_cov = val_future[start_idx:start_idx+24] if val_future else None
                
                preds = model.predict(
                    n=24,
                    series=hist_target,
                    past_covariates=hist_past,
                    future_covariates=fut_cov,
                    num_samples=100
                )
            else:
                # NHiTS: stack past and future covariates
                hist_cov = val_past[:start_idx] if start_idx > 0 else val_past[:24]
                if val_future:
                    fut_slice = val_future[:start_idx+24] if start_idx > 0 else val_future[:48]
                    hist_cov = hist_cov.stack(fut_slice)
                
                preds = model.predict(
                    n=24,
                    series=hist_target,
                    past_covariates=hist_cov,
                    num_samples=100
                )
            
            # Inverse transform predictions
            preds_original = state.target_scaler.inverse_transform(preds)
            
            # Extract quantiles
            p10 = preds_original.quantile(0.1).values().flatten()
            p50 = preds_original.quantile(0.5).values().flatten()
            p90 = preds_original.quantile(0.9).values().flatten()
            
            # Get actual values
            actuals_scaled = val_target[start_idx:start_idx+24]
            actuals = state.target_scaler.inverse_transform(actuals_scaled).values().flatten()
            
            # Store results
            errors = np.abs(actuals - p50)
            all_errors.extend(errors)
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
    
    return mae, picp


def train_nhits(params: Dict[str, Any], csv_path: str) -> Tuple[float, float]:
    """Train and evaluate NHiTS model."""
    print(f"\n  Training NHiTS with params: {params}")
    
    # Auto-detect feature config
    cfg = mp.water_feature_config() if 'water' in csv_path.lower() else mp.default_feature_config()
    
    # Prepare data
    state, t_sc, v_sc, _ = mp.prepare_model_data(
        csv_path,
        to_naive(SPLIT_CONFIG["train_end"]),
        to_naive(SPLIT_CONFIG["val_end"]),
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
    
    # Evaluate
    mae, picp = evaluate_model_walk_forward(model, v_sc, state, n_steps=10, is_tft=False)
    
    print(f"  Results: MAE={mae:.4f}, PICP={picp:.2f}%")
    return mae, picp


def train_tft(params: Dict[str, Any], csv_path: str) -> Tuple[float, float]:
    """Train and evaluate TFT model."""
    print(f"\n  Training TFT with params: {params}")
    
    # Auto-detect feature config
    cfg = mp.water_feature_config() if 'water' in csv_path.lower() else mp.default_feature_config()
    
    # Prepare data
    state, t_sc, v_sc, _ = mp.prepare_model_data(
        csv_path,
        to_naive(SPLIT_CONFIG["train_end"]),
        to_naive(SPLIT_CONFIG["val_end"]),
        cfg
    )
    
    # Model configuration
    model_config = {
        **HPO_TRAINING_CONFIG,
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
    
    # Train model
    model = TFTModel(**model_config)
    model.fit(
        t_sc["target"],
        past_covariates=tp,
        future_covariates=tf,
        val_series=v_sc["target"],
        val_past_covariates=vp,
        val_future_covariates=vf
    )
    
    # Prepare validation data for evaluation
    v_sc_eval = {
        "target": v_sc["target"],
        "past_covariates": vp,
        "future_covariates": vf
    }
    
    # Evaluate
    mae, picp = evaluate_model_walk_forward(model, v_sc_eval, state, n_steps=10, is_tft=True)
    
    print(f"  Results: MAE={mae:.4f}, PICP={picp:.2f}%")
    return mae, picp


def train_timesnet(params: Dict[str, Any], csv_path: str) -> Tuple[float, float]:
    """Train and evaluate TimesNet model."""
    print(f"\n  Training TimesNet with params: {params}")
    
    # Load and prepare data
    df_full = mp.load_and_validate_features(csv_path)
    
    # Determine target column
    target_col = "water_consumption" if 'water' in csv_path.lower() else "heat_consumption"
    
    # Convert to NeuralForecast format
    nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", target_col: "y"})
    unique_id = "tommerby" if "tommerby" in csv_path.lower() else ("centrum" if "centrum" in csv_path.lower() else "nordbyen")
    nf_df["unique_id"] = unique_id
    
    # Get feature config
    cfg = mp.water_feature_config() if 'water' in csv_path.lower() else mp.default_feature_config()
    
    # Prepare exogenous features (TimesNet treats all as future-known)
    futr_ex = [c for c in (cfg.past_covariates_cols + cfg.future_covariates_cols)
               if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
    
    nf_df = nf_df[["unique_id", "ds", "y"] + futr_ex].dropna()
    nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
    
    # Split data
    train_df = nf_df[nf_df['ds'] <= to_naive(SPLIT_CONFIG["train_end"])].reset_index(drop=True)
    val_df = nf_df[(nf_df['ds'] > to_naive(SPLIT_CONFIG["train_end"])) & 
                   (nf_df['ds'] <= to_naive(SPLIT_CONFIG["val_end"]))].reset_index(drop=True)
    
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
        (mae, negative_picp): We minimize both objectives
        - Minimize MAE (lower is better)
        - Minimize negative_picp (which means maximize PICP, target 80%)
    """
    # Get CSV path
    csv_path = DATASET_PATHS[dataset]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    # Get hyperparameters based on model
    if model_name == "NHITS_Q":
        params = get_nhits_search_space(trial)
        mae, picp = train_nhits(params, csv_path)
    elif model_name == "TFT_Q":
        params = get_tft_search_space(trial)
        mae, picp = train_tft(params, csv_path)
    elif model_name == "TIMESNET_Q":
        params = get_timesnet_search_space(trial)
        mae, picp = train_timesnet(params, csv_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Multi-objective: minimize MAE, maximize PICP (target 80%)
    # We penalize deviation from 80% coverage
    picp_penalty = abs(picp - 80.0)  # 0 is perfect, higher is worse
    
    return mae, picp_penalty


def run_optimization(model_name: str, dataset: str, n_trials: int, job_id: str = None):
    """Run multi-objective hyperparameter optimization."""
    
    # Get job ID from environment or use provided
    if job_id is None:
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
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
    
    # Create multi-objective study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        directions=["minimize", "minimize"],  # Minimize MAE and PICP penalty
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
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
    best_picp = 80.0 - best_picp_penalty  # Approximate actual PICP
    
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
    
    args = parser.parse_args()
    
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
```

**Key Features:**
- Multi-objective optimization (MAE + PICP penalty)
- Walk-forward validation (10 steps on val set)
- Supports all 3 models and 3 datasets
- Auto-saves to `hpo/results/{MODEL}_{DATASET}/`
- Generates Pareto front visualizations
- Includes all pareto-optimal solutions in results

---

#### Task 1.3: Create Job Submission Helper

**Create:** `hpo/submit_job.sh`

```bash
#!/bin/bash
#
# Submit single HPO job to SLURM
#
# Usage:
#   ./hpo/submit_job.sh NHITS_Q heat 50
#   ./hpo/submit_job.sh TFT_Q water_centrum 30
#   ./hpo/submit_job.sh TIMESNET_Q water_tommerby 50
#

MODEL=$1
DATASET=$2
TRIALS=${3:-50}

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <MODEL> <DATASET> [TRIALS]"
    echo ""
    echo "Models:   NHITS_Q, TFT_Q, TIMESNET_Q"
    echo "Datasets: heat, water_centrum, water_tommerby"
    echo "Trials:   Default 50"
    echo ""
    echo "Examples:"
    echo "  $0 NHITS_Q heat 50"
    echo "  $0 TFT_Q water_centrum 30"
    exit 1
fi

# Create log directory
mkdir -p hpo/logs

# Submit job
JOB_ID=$(sbatch \
    --job-name=hpo_${MODEL}_${DATASET} \
    --output=hpo/logs/hpo_${MODEL}_${DATASET}_%j.log \
    --error=hpo/logs/hpo_${MODEL}_${DATASET}_%j.err \
    --time=12:00:00 \
    --gres=gpu:a100:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --partition=a100 \
    --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TRIALS=$TRIALS \
    --wrap="source ~/.bashrc && conda activate myenv && python hpo/run_hpo.py --model $MODEL --dataset $DATASET --trials $TRIALS" \
    | awk '{print $4}')

echo "✅ Submitted HPO job for ${MODEL} on ${DATASET}"
echo "   Job ID: $JOB_ID"
echo "   Trials: $TRIALS"
echo "   Log:    hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.log"
echo "   Results will be saved to: hpo/results/${MODEL}_${DATASET}/"
echo ""
echo "Monitor with: tail -f hpo/logs/hpo_${MODEL}_${DATASET}_${JOB_ID}.log"
echo "Check status: squeue -j $JOB_ID"
```

**Make executable:**
```bash
chmod +x hpo/submit_job.sh
```

---

### Phase 2: Testing & Validation (Priority: HIGH)

#### Task 2.1: Local Testing

**Test each model locally with 3 trials:**

```bash
# Test NHITS_Q
python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 3 --job-id test_nhits

# Test TFT_Q  
python hpo/run_hpo.py --model TFT_Q --dataset heat --trials 3 --job-id test_tft

# Test TIMESNET_Q
python hpo/run_hpo.py --model TIMESNET_Q --dataset heat --trials 3 --job-id test_timesnet
```

**Expected output structure:**
```
hpo/results/
├── NHITS_Q_heat/
│   ├── best_params_NHITS_Q_heat_test_nhits.json
│   ├── study_NHITS_Q_heat_test_nhits.db
│   ├── pareto_front_test_nhits.html
│   ├── mae_history_test_nhits.html
│   └── param_importance_test_nhits.html
├── TFT_Q_heat/
│   └── ...
└── TIMESNET_Q_heat/
    └── ...
```

#### Task 2.2: Verify Result Format

Check that generated JSON has correct structure:

```bash
cat hpo/results/NHITS_Q_heat/best_params_NHITS_Q_heat_test_nhits.json
```

Should contain:
- model, dataset, job_id
- best_mae, best_picp_approx
- best_params (hyperparameters)
- all_pareto_trials (Pareto front solutions)

---

### Phase 3: Production Runs (Priority: MEDIUM)

#### Task 3.1: Submit Individual Jobs

**Submit one job at a time, monitor completion:**

```bash
# Priority 1: Heat dataset
./hpo/submit_job.sh NHITS_Q heat 50
./hpo/submit_job.sh TFT_Q heat 50
./hpo/submit_job.sh TIMESNET_Q heat 50

# Wait for completion or submit in parallel if resources allow
# Check: squeue -u $USER

# Priority 2: Water Centrum
./hpo/submit_job.sh NHITS_Q water_centrum 50
./hpo/submit_job.sh TFT_Q water_centrum 50
./hpo/submit_job.sh TIMESNET_Q water_centrum 50

# Priority 3: Water Tommerby
./hpo/submit_job.sh NHITS_Q water_tommerby 50
./hpo/submit_job.sh TFT_Q water_tommerby 50
./hpo/submit_job.sh TIMESNET_Q water_tommerby 50
```

**Monitor progress:**
```bash
# Check all HPO jobs
squeue -u $USER | grep hpo

# Check specific job
tail -f hpo/logs/hpo_NHITS_Q_heat_<jobid>.log

# Check results
ls -lh hpo/results/*/best_params_*.json
```

---

### Phase 4: Results Analysis (Priority: MEDIUM)

#### Task 4.1: Create Analysis Script

**Create:** `hpo/analyze_results.py`

```python
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
```

#### Task 4.2: Integrate with Benchmarker

**Update:** `benchmarker.py` (Benchmarker.__init__)

```python
def _load_hpo_params(self, model: str) -> Dict:
    """Load HPO results for model-dataset combination."""
    # Determine dataset key from csv_path
    dataset_key = self._infer_dataset_key()
    
    # Try to load HPO results
    search_pattern = f"hpo/results/{model}_{dataset_key}/best_params_{model}_{dataset_key}_*.json"
    result_files = glob.glob(search_pattern)
    
    if result_files:
        # Load most recent (sort by modification time)
        latest_file = max(result_files, key=os.path.getmtime)
        with open(latest_file, 'r') as f:
            data = json.load(f)
            print(f"  ✅ Loaded HPO params for {model} from {latest_file}")
            return data["best_params"]
    
    # Fallback: try old location (backward compatibility)
    old_path = f"results/best_params_{model}.json"
    if os.path.exists(old_path):
        with open(old_path, 'r') as f:
            data = json.load(f)
            print(f"  ⚠️  Using legacy HPO params from {old_path}")
            return data
    
    print(f"  ℹ️  No HPO params found for {model}, using defaults")
    return None

def _infer_dataset_key(self) -> str:
    """Infer dataset key from csv_path."""
    lower_path = self.csv_path.lower()
    if 'tommerby' in lower_path:
        return 'water_tommerby'
    elif 'centrum' in lower_path:
        return 'water_centrum'
    elif 'nordbyen' in lower_path or 'heat' in lower_path:
        return 'heat'
    return 'unknown'

# In __init__, update loading:
nhits_best = self._load_hpo_params("NHITS_Q")
tft_best = self._load_hpo_params("TFT_Q")
timesnet_best = self._load_hpo_params("TIMESNET_Q")
```

---

### Phase 5: Documentation (Priority: LOW)

#### Task 5.1: Update README

**Update:** `hpo/README.md`

```markdown
# Hyperparameter Optimization (HPO)

Multi-objective hyperparameter optimization for time series forecasting models.

## Overview

Optimizes both:
- **MAE** (Mean Absolute Error) - Point forecast accuracy
- **PICP** (Prediction Interval Coverage Probability) - Probabilistic calibration

Uses Optuna's multi-objective optimization to find Pareto-optimal hyperparameters.

## Quick Start

### 1. Local Testing (3 trials, ~30-45 min)

```bash
python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 3 --job-id test
python hpo/run_hpo.py --model TFT_Q --dataset heat --trials 3 --job-id test
python hpo/run_hpo.py --model TIMESNET_Q --dataset heat --trials 3 --job-id test
```

### 2. Submit to SLURM (50 trials, ~8-12 hours)

```bash
# Submit individual jobs
./hpo/submit_job.sh NHITS_Q heat 50
./hpo/submit_job.sh TFT_Q water_centrum 50
./hpo/submit_job.sh TIMESNET_Q water_tommerby 50
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER | grep hpo

# View logs
tail -f hpo/logs/hpo_NHITS_Q_heat_<jobid>.log

# Check results
ls -lh hpo/results/*/best_params_*.json
```

### 4. Analyze Results

```bash
python hpo/analyze_results.py
python hpo/analyze_results.py --model NHITS_Q
python hpo/analyze_results.py --dataset heat
```

## Structure

```
hpo/
├── run_hpo.py              # Main HPO runner
├── hpo_config.py           # Search space definitions
├── submit_job.sh           # Job submission helper
├── analyze_results.py      # Results analysis
├── results/                # All HPO results
│   ├── NHITS_Q_heat/
│   │   ├── best_params_NHITS_Q_heat_1234567.json
│   │   ├── study_NHITS_Q_heat_1234567.db
│   │   ├── pareto_front_1234567.html
│   │   ├── mae_history_1234567.html
│   │   └── param_importance_1234567.html
│   └── ... (all 9 combinations)
└── logs/                   # SLURM logs
    └── hpo_NHITS_Q_heat_1234567.log
```

## Models & Datasets

**Models:**
- `NHITS_Q` - Neural Hierarchical Interpolation for Time Series
- `TFT_Q` - Temporal Fusion Transformer
- `TIMESNET_Q` - TimesNet with 2D convolutions

**Datasets:**
- `heat` - Nordbyen heat consumption
- `water_centrum` - Centrum water consumption
- `water_tommerby` - Tommerby water consumption

## Result Format

Each optimization produces:

```json
{
  "model": "NHITS_Q",
  "dataset": "heat",
  "job_id": "1234567",
  "optimization_date": "2026-01-16T10:30:00",
  "n_trials": 50,
  "best_mae": 0.185,
  "best_picp_approx": 78.5,
  "best_params": {
    "num_stacks": 4,
    "num_blocks": 2,
    "lr": 0.000634,
    ...
  },
  "all_pareto_trials": [...]
}
```

## Integration with Benchmarker

Benchmarker automatically loads HPO results:

```python
benchmarker = Benchmarker(
    csv_path="processing/nordbyen_processing/nordbyen_features_engineered.csv",
    models_to_run=["NHITS_Q", "TFT_Q", "TIMESNET_Q"]
)
# Auto-loads: hpo/results/NHITS_Q_heat/best_params_NHITS_Q_heat_*.json
```

## All Experiment Combinations

| # | Model | Dataset | Priority | Command |
|---|-------|---------|----------|---------|
| 1 | NHITS_Q | heat | ⭐⭐⭐ | `./hpo/submit_job.sh NHITS_Q heat 50` |
| 2 | TFT_Q | heat | ⭐⭐⭐ | `./hpo/submit_job.sh TFT_Q heat 50` |
| 3 | TIMESNET_Q | heat | ⭐⭐⭐ | `./hpo/submit_job.sh TIMESNET_Q heat 50` |
| 4 | NHITS_Q | water_centrum | ⭐⭐ | `./hpo/submit_job.sh NHITS_Q water_centrum 50` |
| 5 | TFT_Q | water_centrum | ⭐⭐ | `./hpo/submit_job.sh TFT_Q water_centrum 50` |
| 6 | TIMESNET_Q | water_centrum | ⭐⭐ | `./hpo/submit_job.sh TIMESNET_Q water_centrum 50` |
| 7 | NHITS_Q | water_tommerby | ⭐ | `./hpo/submit_job.sh NHITS_Q water_tommerby 50` |
| 8 | TFT_Q | water_tommerby | ⭐ | `./hpo/submit_job.sh TFT_Q water_tommerby 50` |
| 9 | TIMESNET_Q | water_tommerby | ⭐ | `./hpo/submit_job.sh TIMESNET_Q water_tommerby 50` |

**Estimated Total GPU Time:** 80-100 hours

## Troubleshooting

### Out of Memory
- Reduce `n_epochs` in `hpo_config.py`
- Request more memory: `--mem=128G`

### Job Fails
- Check log: `hpo/logs/hpo_<MODEL>_<DATASET>_<JOBID>.err`
- Test locally first with 3 trials
- Verify CSV path exists

### No Results
- Check: `ls -lh hpo/results/*/`
- Verify job completed: `sacct -j <JOBID>`
- Check for errors in logs

---

**For detailed methodology, see:** [HPO_REVIVAL_PLAN.md](HPO_REVIVAL_PLAN.md)
```

---

## Implementation Timeline

### Week 1 (Jan 16-22): Setup & Testing

- **Day 1-2:** Create core files
  - [ ] Create `hpo/hpo_config.py`
  - [ ] Create `hpo/run_hpo.py`
  - [ ] Create `hpo/submit_job.sh`
  - [ ] Make scripts executable

- **Day 3-4:** Local testing
  - [ ] Test NHITS_Q (3 trials)
  - [ ] Test TFT_Q (3 trials)
  - [ ] Test TIMESNET_Q (3 trials)
  - [ ] Verify result format

- **Day 5:** Validation
  - [ ] Check all outputs generated correctly
  - [ ] Test on all 3 datasets locally
  - [ ] Fix any bugs

### Week 2 (Jan 23-29): Production Runs

- **Day 1:** Submit Priority 1 (Heat)
  - [ ] `./hpo/submit_job.sh NHITS_Q heat 50`
  - [ ] `./hpo/submit_job.sh TFT_Q heat 50`
  - [ ] `./hpo/submit_job.sh TIMESNET_Q heat 50`

- **Day 2-3:** Monitor heat runs, submit Priority 2
  - [ ] Monitor heat experiments
  - [ ] Submit water_centrum experiments
  - [ ] Handle any failures

- **Day 4-7:** Continue monitoring
  - [ ] Submit water_tommerby experiments
  - [ ] Monitor all jobs
  - [ ] Collect results as they complete

### Week 3 (Jan 30 - Feb 5): Analysis & Integration

- **Day 1-2:** Analysis
  - [ ] Create `hpo/analyze_results.py`
  - [ ] Run analysis on all results
  - [ ] Generate comparison tables

- **Day 3-4:** Integration
  - [ ] Update benchmarker to load HPO results
  - [ ] Re-run benchmarks with optimized params
  - [ ] Compare baseline vs optimized

- **Day 5:** Documentation
  - [ ] Update README
  - [ ] Document findings
  - [ ] Create visualizations

---

## Expected Outcomes

### Performance Improvements

| Model | Baseline MAE | Optimized MAE | Baseline PICP | Optimized PICP |
|-------|--------------|---------------|---------------|----------------|
| NHITS_Q | 0.20 | 0.18-0.19 | 40-50% | 75-85% |
| TFT_Q | 0.22 | 0.19-0.21 | 45-55% | 75-85% |
| TIMESNET_Q | 0.21 | 0.19-0.20 | 50-60% | 75-85% |

**Key Improvements:**
- 5-15% MAE reduction
- 25-40% absolute PICP improvement (toward 80% target)
- Better calibrated uncertainty intervals

### Deliverables

1. ✅ 9 optimized hyperparameter configurations
2. ✅ Pareto front visualizations for each
3. ✅ Automated benchmarker integration
4. ✅ Analysis scripts and documentation
5. ✅ Thesis-ready results and methodology

---

## Next Steps After Completion

1. **Benchmark Comparison:**
   - Run full benchmarks with optimized params
   - Compare against baseline (current best_params)
   - Generate improvement reports

2. **Thesis Integration:**
   - Document HPO methodology
   - Include Pareto front visualizations
   - Explain multi-objective trade-offs

3. **Production Deployment:**
   - Use optimized params as new defaults
   - Create model cards
   - Set up retraining workflows

---

## References

### Internal
- [MODEL_CONFIGURATION_AND_PREDICTION_FLOW.md](../docs/MODEL_CONFIGURATION_AND_PREDICTION_FLOW.md)
- [benchmarker.py](../benchmarker.py)

### External
- [Optuna Multi-Objective](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
- [Pareto Optimization](https://en.wikipedia.org/wiki/Pareto_efficiency)

---

**Status:** Plan Finalized - Ready for Implementation  
**Next Action:** Create `hpo/hpo_config.py` and `hpo/run_hpo.py`

```python
# NHITS_Q Search Space
nhits_params = {
    "num_stacks": trial.suggest_int("num_stacks", 2, 5),
    "num_blocks": trial.suggest_int("num_blocks", 1, 3),
    "num_layers": trial.suggest_int("num_layers", 2, 4),
    "layer_widths": trial.suggest_categorical("layer_widths", [256, 512, 1024]),
    "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    "dropout": trial.suggest_float("dropout", 0.05, 0.4),
    "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
}

# TFT_Q Search Space (NEW)
tft_params = {
    "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
    "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
    "num_attention_heads": trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
    "dropout": trial.suggest_float("dropout", 0.05, 0.4),
    "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
}

# TIMESNET_Q Search Space
timesnet_params = {
    "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
    "conv_hidden_size": trial.suggest_categorical("conv_hidden_size", [32, 64, 128]),
    "top_k": trial.suggest_int("top_k", 2, 5),
    "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    "dropout": trial.suggest_float("dropout", 0.05, 0.4),
}
```

#### Task 1.2: Create Unified HPO Runner

**Create:** `hpo/run_hpo.py`

```python
"""
Unified HPO runner for all models and datasets.

Usage:
  python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 50
  python hpo/run_hpo.py --model TFT_Q --dataset water_centrum --trials 50 --test
"""
```

**Features:**
- Command-line interface with argparse
- Support for: `--model`, `--dataset`, `--trials`, `--test`, `--stage`
- Auto-detect CSV paths based on dataset
- Save results to appropriate location
- Progress tracking with Optuna visualization

#### Task 1.3: Update SLURM Job Script

**Update:** `hpo/hpo_job.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=hpo_{model}_{dataset}
#SBATCH --output=hpo/results/logs/hpo_%j.log
#SBATCH --error=hpo/results/logs/hpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=a100

# Parameters passed from submit script
MODEL=${1:-NHITS_Q}
DATASET=${2:-heat}
TRIALS=${3:-50}
STAGE=${4:-1}

echo "HPO Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Trials: $TRIALS"
echo "  Stage: $STAGE"

source ~/.bashrc
conda activate myenv

python hpo/run_hpo.py --model $MODEL --dataset $DATASET --trials $TRIALS --stage $STAGE
```

---

### Phase 2: TFT Implementation (Priority: HIGH)

**Goal:** Add full TFT support to HPO system

#### Task 2.1: Implement TFT Training Function

**Add to HPO runner:**

```python
def train_tft(csv_path: str, params: Dict[str, Any], dataset: str):
    """Train TFT model with given hyperparameters."""
    train_end = "2023-10-31 23:00:00+00:00"
    val_end = "2024-03-31 23:00:00+00:00"
    
    # Auto-detect feature config
    cfg = mp.water_feature_config() if 'water' in dataset.lower() else mp.default_feature_config()
    
    state, t_sc, v_sc, _ = mp.prepare_model_data(
        csv_path, to_naive(train_end), to_naive(val_end), cfg
    )
    
    model_params = {
        "input_chunk_length": 168,
        "output_chunk_length": 24,
        "batch_size": 32,
        "n_epochs": params.get("n_epochs", 10),
        "random_state": 42,
        "force_reset": True,
        "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        "hidden_size": params["hidden_size"],
        "lstm_layers": params["lstm_layers"],
        "num_attention_heads": params["num_attention_heads"],
        "dropout": params["dropout"],
        "optimizer_kwargs": {"lr": params["lr"]},
        "pl_trainer_kwargs": {
            "logger": False,  # Disable for HPO speed
            "enable_checkpointing": False,
        }
    }
    
    # TFT uses separate past and future covariates
    tp, vp = t_sc["past_covariates"], v_sc["past_covariates"]
    tf, vf = t_sc["future_covariates"], v_sc["future_covariates"]
    
    model = TFTModel(**model_params)
    model.fit(
        t_sc["target"], 
        past_covariates=tp,
        future_covariates=tf,
        val_series=v_sc["target"],
        val_past_covariates=vp,
        val_future_covariates=vf
    )
    
    # Evaluate on validation set (walk-forward mini-benchmark)
    mae = evaluate_model_walk_forward(model, v_sc, state)
    return mae
```

#### Task 2.2: Test TFT HPO Locally

```bash
# Quick test with 3 trials
python hpo/run_hpo.py --model TFT_Q --dataset heat --trials 3 --test

# Expected output:
# Trial 0: MAE = X.XXX
# Trial 1: MAE = X.XXX
# Trial 2: MAE = X.XXX
# Best parameters: {...}
```

---

### Phase 3: Dataset-Specific Optimization (Priority: MEDIUM)

**Goal:** Run HPO for each model on each relevant dataset

#### Task 3.1: Define Experiment Matrix

| Experiment ID | Model | Dataset | Trials | Priority | Est. Time |
|---------------|-------|---------|--------|----------|-----------|
| EXP-01 | NHITS_Q | heat | 50 | ⭐⭐⭐ | 8-10h |
| EXP-02 | NHITS_Q | water_centrum | 50 | ⭐⭐ | 8-10h |
| EXP-03 | NHITS_Q | water_tommerby | 50 | ⭐ | 8-10h |
| EXP-04 | TFT_Q | heat | 50 | ⭐⭐⭐ | 10-12h |
| EXP-05 | TFT_Q | water_centrum | 50 | ⭐⭐ | 10-12h |
| EXP-06 | TFT_Q | water_tommerby | 50 | ⭐ | 10-12h |
| EXP-07 | TIMESNET_Q | heat | 50 | ⭐⭐⭐ | 8-10h |
| EXP-08 | TIMESNET_Q | water_centrum | 50 | ⭐⭐ | 8-10h |
| EXP-09 | TIMESNET_Q | water_tommerby | 50 | ⭐ | 8-10h |

**Total Estimated GPU Time:** 80-100 hours

#### Task 3.2: Create Batch Submission Script

**Create:** `hpo/submit_all_experiments.sh`

```bash
#!/bin/bash

# Submit all HPO experiments with dependencies
# Usage: bash hpo/submit_all_experiments.sh [--test]

TEST_MODE=""
if [[ "$1" == "--test" ]]; then
    TEST_MODE="--test"
    TRIALS=3
    echo "TEST MODE: Running with 3 trials only"
else
    TRIALS=50
fi

# Priority 1: Heat dataset (main thesis focus)
echo "Submitting Priority 1 experiments (Heat)..."
sbatch hpo/hpo_job.slurm NHITS_Q heat $TRIALS
sbatch hpo/hpo_job.slurm TFT_Q heat $TRIALS
sbatch hpo/hpo_job.slurm TIMESNET_Q heat $TRIALS

# Priority 2: Water Centrum
echo "Submitting Priority 2 experiments (Water Centrum)..."
sbatch hpo/hpo_job.slurm NHITS_Q water_centrum $TRIALS
sbatch hpo/hpo_job.slurm TFT_Q water_centrum $TRIALS
sbatch hpo/hpo_job.slurm TIMESNET_Q water_centrum $TRIALS

# Priority 3: Water Tommerby
echo "Submitting Priority 3 experiments (Water Tommerby)..."
sbatch hpo/hpo_job.slurm NHITS_Q water_tommerby $TRIALS
sbatch hpo/hpo_job.slurm TFT_Q water_tommerby $TRIALS
sbatch hpo/hpo_job.slurm TIMESNET_Q water_tommerby $TRIALS

echo "All experiments submitted!"
echo "Monitor progress: squeue -u $USER"
```

---

### Phase 4: Results Management (Priority: MEDIUM)

**Goal:** Organize and consolidate HPO results for easy benchmarker integration

#### Task 4.1: Standardize Result Format

**Output Location:** `results/best_params_{MODEL}_{DATASET}.json`

```json
{
  "model": "NHITS_Q",
  "dataset": "heat",
  "optimization_date": "2026-01-16",
  "n_trials": 50,
  "best_trial": 42,
  "best_mae": 0.185,
  "best_params": {
    "num_stacks": 4,
    "num_blocks": 2,
    "num_layers": 3,
    "layer_widths": 512,
    "lr": 0.000634,
    "dropout": 0.15,
    "weight_decay": 1.2e-06
  },
  "validation_metrics": {
    "mae": 0.185,
    "rmse": 0.312,
    "mape": 4.56
  },
  "training_time_hours": 9.2,
  "total_trials_completed": 50,
  "pruned_trials": 5,
  "optuna_study_name": "nhits_q_heat_20260116"
}
```

#### Task 4.2: Create Results Consolidation Script

**Create:** `hpo/consolidate_results.py`

```python
"""
Consolidate HPO results and prepare for benchmarker integration.

Usage:
  python hpo/consolidate_results.py
  python hpo/consolidate_results.py --model NHITS_Q --dataset heat
"""
```

**Features:**
- Load all Optuna study databases
- Extract best parameters
- Generate comparison table
- Copy to `results/` directory for benchmarker
- Create visualization of search spaces explored

#### Task 4.3: Update Benchmarker Integration

**Current Code:**
```python
nhits_best = self._load_json("results/best_params_NHITS.json")
tft_best = self._load_json("results/best_params_TFT.json")
timesnet_best = self._load_json("results/best_params_TIMESNET.json")
```

**Enhanced Code:**
```python
# Try dataset-specific params first, fallback to general
def _load_best_params(self, model: str, dataset: str):
    # Try dataset-specific
    specific = f"results/best_params_{model}_{dataset}.json"
    if os.path.exists(specific):
        return self._load_json(specific)
    
    # Fallback to general (backward compatible)
    general = f"results/best_params_{model}.json"
    return self._load_json(general)

# In __init__:
nhits_best = self._load_best_params("NHITS", self.dataset)
tft_best = self._load_best_params("TFT", self.dataset)
timesnet_best = self._load_best_params("TIMESNET", self.dataset)
```

---

### Phase 5: Advanced Features (Priority: LOW)

**Goal:** Implement Stage 2 quantile calibration and multi-objective optimization

#### Task 5.1: Stage 2 Quantile Calibration

**Concept:** After finding best architecture (Stage 1), optimize quantile levels to achieve target coverage

**Search Space:**
```python
# Fixed: Architecture from Stage 1
# Vary: Quantile levels
params = {
    "q_low": trial.suggest_float("q_low", 0.05, 0.20),   # Instead of fixed 0.1
    "q_mid": 0.5,                                         # Always median
    "q_high": trial.suggest_float("q_high", 0.80, 0.95)  # Instead of fixed 0.9
}

# Objective: Minimize deviation from target coverage
target_picp = 80.0  # 80% target coverage
actual_picp = calculate_picp(predictions)
loss = abs(actual_picp - target_picp)  # Want PICP close to 80%

# Secondary: Minimize interval width (don't make it too wide)
penalty = mean_interval_width / mean_actual * 100
objective = loss + 0.1 * penalty
```

#### Task 5.2: Multi-Objective Optimization

**Optimize Both:**
1. **MAE** (point forecast accuracy)
2. **PICP** (interval calibration)

**Using Optuna's Multi-Objective:**
```python
study = optuna.create_study(
    directions=["minimize", "maximize"],  # Minimize MAE, Maximize PICP
    study_name="multi_obj_nhits_q"
)

def objective(trial):
    params = suggest_params(trial)
    model = train_model(params)
    mae, picp = evaluate(model)
    return mae, picp  # Return both objectives

study.optimize(objective, n_trials=50)

# Extract Pareto-optimal solutions
pareto_front = study.best_trials
```

---

## Implementation Roadmap

### Week 1: Setup & Infrastructure (Jan 16-22)

**Day 1-2 (Setup):**
- [ ] Review existing `hpo_tuner.py` code
- [ ] Create `hpo/run_hpo.py` unified runner
- [ ] Update `hpo/hpo_job.slurm` with new structure
- [ ] Create logging directory: `hpo/results/logs/`

**Day 3-4 (TFT Implementation):**
- [ ] Implement `train_tft()` function
- [ ] Test TFT HPO locally (3 trials)
- [ ] Verify training completes without errors
- [ ] Validate output format

**Day 5 (Testing):**
- [ ] Run test experiments for all 3 models (3 trials each)
- [ ] Verify result saving works correctly
- [ ] Check GPU utilization and memory usage

### Week 2: Execution (Jan 23-29)

**Day 1 (Batch Submission):**
- [ ] Create `submit_all_experiments.sh`
- [ ] Submit Priority 1 experiments (heat dataset)
- [ ] Monitor first jobs for issues

**Day 2-7 (Monitoring):**
- [ ] Check job progress daily
- [ ] Handle any failed jobs
- [ ] Collect intermediate results
- [ ] Estimate completion times

### Week 3: Analysis & Integration (Jan 30 - Feb 5)

**Day 1-2 (Results Consolidation):**
- [ ] Run `consolidate_results.py`
- [ ] Generate comparison tables
- [ ] Visualize hyperparameter importance
- [ ] Document findings

**Day 3-4 (Benchmarker Integration):**
- [ ] Update benchmarker to load dataset-specific params
- [ ] Re-run benchmarks with optimized parameters
- [ ] Compare baseline vs optimized performance
- [ ] Generate improvement reports

**Day 5 (Documentation):**
- [ ] Document HPO results in thesis
- [ ] Create visualizations (search space plots)
- [ ] Write methodology section
- [ ] Update README files

---

## Quick Start Commands

### 1. Local Testing (Do This First!)

```bash
# Test NHITS_Q on heat data (3 trials, ~30 minutes)
python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 3 --test

# Test TFT_Q on heat data (3 trials, ~45 minutes)
python hpo/run_hpo.py --model TFT_Q --dataset heat --trials 3 --test

# Test TIMESNET_Q on heat data (3 trials, ~30 minutes)
python hpo/run_hpo.py --model TIMESNET_Q --dataset heat --trials 3 --test
```

### 2. Single SLURM Job

```bash
# Submit one experiment
sbatch hpo/hpo_job.slurm NHITS_Q heat 50

# Check status
squeue -u $USER
tail -f hpo/results/logs/hpo_<jobid>.log
```

### 3. Batch Submission (All Experiments)

```bash
# Test mode (3 trials each, ~4 hours total)
bash hpo/submit_all_experiments.sh --test

# Production mode (50 trials each, ~80-100 hours total)
bash hpo/submit_all_experiments.sh
```

### 4. Monitor Progress

```bash
# All jobs
squeue -u $USER

# Specific model/dataset
ls -lh hpo/results/stage1/
tail -f hpo/results/logs/hpo_*.log

# Check best results so far
python hpo/consolidate_results.py --summary
```

### 5. After Completion

```bash
# Consolidate all results
python hpo/consolidate_results.py

# Check what was generated
ls -lh results/best_params_*.json

# Re-run benchmarks with optimized params
python test_benchmarker_quick.py  # Should auto-load new params
```

---

## Expected Performance Gains

Based on existing results and HPO literature:

| Model | Baseline MAE | Expected Optimized MAE | Improvement |
|-------|--------------|------------------------|-------------|
| NHITS_Q (heat) | ~0.20 | 0.18-0.19 | 5-10% |
| TFT_Q (heat) | ~0.22 | 0.19-0.21 | 5-15% |
| TIMESNET_Q (heat) | ~0.21 | 0.19-0.20 | 5-10% |

**PICP Improvement:**
- Current: 40-60% coverage (poor calibration)
- After Stage 2: 75-85% coverage (good calibration)

---

## Risk Mitigation

### Risk 1: Long Training Times
**Mitigation:**
- Start with test mode (3 trials)
- Use GPU partitions (A100 if available)
- Reduce `n_epochs` during HPO (10 instead of 100)
- Implement early stopping

### Risk 2: Failed Jobs
**Mitigation:**
- Save checkpoints after each trial
- Use Optuna's pruning (stop bad trials early)
- Monitor logs actively
- Have restart scripts ready

### Risk 3: Poor Hyperparameter Choices
**Mitigation:**
- Start with informed search spaces (based on literature)
- Use logarithmic scales for learning rates
- Analyze failed trials to adjust bounds
- Implement constraints (e.g., num_stacks * num_blocks < 20)

### Risk 4: Overfitting to Validation Set
**Mitigation:**
- Use separate calibration set
- Implement cross-validation if time permits
- Monitor train vs val metrics
- Test final params on held-out test set

---

## Success Metrics

### Quantitative
- ✅ MAE improves by ≥5% on at least 2/3 models
- ✅ PICP reaches 75-85% (currently 40-60%)
- ✅ Training time remains under 12 hours per experiment
- ✅ All 9 experiments complete successfully

### Qualitative
- ✅ Hyperparameters are interpretable and reasonable
- ✅ Results are reproducible
- ✅ Documentation is complete
- ✅ Integration with benchmarker is seamless

---

## Next Steps After HPO

1. **Thesis Integration:**
   - Add HPO methodology section
   - Include search space visualization
   - Document performance improvements
   - Explain hyperparameter choices

2. **Production Deployment:**
   - Use optimized parameters as defaults
   - Create model cards documenting hyperparameters
   - Set up automated retraining with HPO

3. **Future Work:**
   - Implement Stage 2 (quantile calibration)
   - Try neural architecture search (NAS)
   - Explore ensemble methods with diverse hyperparameters

---

## Resources & References

### Internal Documentation
- [MODEL_CONFIGURATION_AND_PREDICTION_FLOW.md](../docs/MODEL_CONFIGURATION_AND_PREDICTION_FLOW.md)
- [benchmarker.py](../benchmarker.py)
- [model_preprocessing.py](../model_preprocessing.py)

### Optuna Documentation
- [Optuna Quick Start](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)
- [Pruning Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [Multi-Objective](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)

### Papers
- NHiTS: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (2022)
- TFT: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2020)
- TimesNet: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (2023)

---

**Status:** Plan Created - Ready for Implementation  
**Next Action:** Start with Phase 1, Task 1.1 (Enhance `hpo_tuner.py`)
