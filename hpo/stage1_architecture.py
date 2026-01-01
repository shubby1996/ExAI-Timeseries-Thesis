#!/usr/bin/env python3
"""
Stage 1: Architecture Optimization
Optimizes model architecture and learning parameters for best MAE.
Quantiles are fixed at [0.1, 0.5, 0.9].

Usage:
    python stage1_architecture.py --model NHITS_Q --dataset water --trials 50
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
from darts.metrics import mae as darts_mae
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss
import warnings
warnings.filterwarnings('ignore')

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

def train_nhits_q(csv_path: str, params: dict, dataset: str):
    """Train NHITS with Quantile loss"""
    try:
        cfg = get_feature_config(dataset)
        ds_cfg = DATASETS[dataset]
        
        # Prepare data
        state, t_sc, v_sc, _ = mp.prepare_model_data(
            csv_path,
            to_naive(ds_cfg["train_end"]),
            to_naive(ds_cfg["val_end"]),
            cfg
        )
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            mode="min"
        )
        
        # Model parameters (quantiles fixed at [0.1, 0.5, 0.9])
        model_params = {
            "input_chunk_length": 168,
            "output_chunk_length": 24,
            "batch_size": 32,
            "n_epochs": params.get("n_epochs", 15),
            "random_state": 42,
            "force_reset": True,
            "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            "optimizer_kwargs": {
                "lr": params["lr"],
                "weight_decay": params["weight_decay"]
            },
            "num_stacks": params["num_stacks"],
            "num_blocks": params["num_blocks"],
            "num_layers": params["num_layers"],
            "layer_widths": params["layer_widths"],
            "dropout": params["dropout"],
            "pl_trainer_kwargs": {
                "callbacks": [early_stop],
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
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
        
        # Evaluate on validation set (median prediction)
        pred = model.predict(n=24, series=t_sc["target"], past_covariates=tp, num_samples=1)
        mae_val = darts_mae(v_sc["target"][:24], pred)
        
        return float(mae_val)
    except Exception as e:
        print(f"NHITS training failed: {type(e).__name__}: {str(e)[:200]}")
        raise

def train_timesnet_q(csv_path: str, params: dict, dataset: str):
    """Train TimesNet with Quantile loss"""
    try:
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
        
        train_df.rename(columns={cfg.time_col: "ds", cfg.target_col: "y"}, inplace=True)
        val_df.rename(columns={cfg.time_col: "ds", cfg.target_col: "y"}, inplace=True)
        train_df["unique_id"] = "series_1"
        val_df["unique_id"] = "series_1"
        
        # Calculate max_steps based on epochs (same formula as benchmarker.py)
        # NeuralForecast uses max_steps (total gradient updates), not epochs
        # To match NHITS epoch behavior: max_steps = epochs * (samples / batch_size)
        batch_size = 32
        n_epochs_desired = 10  # Keep low for HPO speed (NHITS uses 10 in HPO)
        n_samples = len(train_df)
        steps_per_epoch = max(1, n_samples // batch_size)
        max_steps_calculated = n_epochs_desired * steps_per_epoch
        
        # Model configuration (quantiles fixed at [10, 50, 90])
        # Note: No early stopping used - matching benchmarker.py behavior
        model = TimesNet(
            h=24,
            input_size=168,
            loss=MQLoss(level=[10, 50, 90]),  # Fixed quantiles
            max_steps=max_steps_calculated,
            hidden_size=params["hidden_size"],
            conv_hidden_size=params["conv_hidden_size"],
            top_k=params["top_k"],
            learning_rate=params["lr"],
            dropout=params["dropout"],
            random_seed=42,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        nf = NeuralForecast(models=[model], freq='1h')
        # val_size must be passed to fit() for early stopping to work
        val_size = int(len(train_df) * 0.1)  # Use 10% of train data for validation
        nf.fit(df=train_df, val_size=val_size)
        
        # Evaluate on validation set (first 24 hours)
        val_df_h = val_df.head(24).reset_index(drop=True)
        preds = nf.predict(df=train_df, futr_df=val_df_h.drop(columns=['y']))
        
        # Extract median prediction
        col = [c for c in preds.columns if 'median' in c.lower() or '-50' in c][0]
        y_hat = preds[col].values
        y_true = val_df_h['y'].values
        mae = np.mean(np.abs(y_hat - y_true))
        
        return float(mae)
    except Exception as e:
        print(f"TimesNet training failed: {type(e).__name__}: {str(e)[:200]}")
        raise

def objective(trial, model: str, dataset: str, csv_path: str):
    """Optuna objective function"""
    
    if model == "NHITS_Q":
        params = {
            "num_stacks": trial.suggest_int("num_stacks", 1, 5),
            "num_blocks": trial.suggest_int("num_blocks", 1, 3),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "layer_widths": trial.suggest_categorical("layer_widths", [128, 256, 512, 1024]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "n_epochs": 10  # Keep low for HPO speed
        }
        return train_nhits_q(csv_path, params, dataset)
    
    elif model == "TIMESNET_Q":
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "conv_hidden_size": trial.suggest_categorical("conv_hidden_size", [32, 64, 128, 256]),
            "top_k": trial.suggest_int("top_k", 1, 5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5)
            # max_steps now calculated dynamically in train_timesnet_q() based on actual data size
        }
        return train_timesnet_q(csv_path, params, dataset)
    
    else:
        raise ValueError(f"Unknown model: {model}")

def main():
    parser = argparse.ArgumentParser(description='Stage 1: Architecture Optimization')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['NHITS_Q', 'TIMESNET_Q'],
                        help='Model to optimize')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['heat', 'water'],
                        help='Dataset to use')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (1 trial)')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    ds_cfg = DATASETS[args.dataset]
    csv_path = os.path.join(project_root, ds_cfg["csv_path"])
    
    # Create results directory
    experiment_name = f"{args.dataset}_{args.model.lower()}"
    results_dir = os.path.join(project_root, "hpo", "results", "stage1", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup Optuna study
    study_name = f"stage1_{experiment_name}"
    storage_path = f"sqlite:///{results_dir}/study.db"
    
    print("="*80)
    print("STAGE 1: ARCHITECTURE OPTIMIZATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Objective: Minimize MAE")
    print(f"Trials: {args.trials if not args.test else 1}")
    print(f"Results: {results_dir}")
    print(f"Quantiles: [0.1, 0.5, 0.9] (FIXED)")
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
        lambda trial: objective(trial, args.model, args.dataset, csv_path),
        n_trials=n_trials
    )
    
    # Save results
    print("\n" + "="*80)
    print("BEST RESULTS")
    print("="*80)
    print(f"Best MAE: {study.best_value:.6f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    best_params_file = os.path.join(results_dir, "best_params.json")
    result_data = {
        "stage": 1,
        "model": args.model,
        "dataset": args.dataset,
        "objective": "MAE",
        "best_mae": study.best_value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "quantiles": [0.1, 0.5, 0.9],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(best_params_file, "w") as f:
        json.dump(result_data, f, indent=4)
    
    # Also save to project results for benchmarker
    project_results_file = os.path.join(project_root, "results", f"best_params_{args.model}.json")
    with open(project_results_file, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    
    print(f"\n✓ Results saved to: {best_params_file}")
    print(f"✓ Params copied to: {project_results_file}")
    print("="*80)
    print("\nNext step: Run Stage 2 calibration with these parameters")
    print(f"  python hpo/stage2_calibration.py --model {args.model} --dataset {args.dataset}")
    print("="*80)

if __name__ == "__main__":
    main()
