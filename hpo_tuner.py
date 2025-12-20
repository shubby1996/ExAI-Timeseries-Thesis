import os
import sys
import json
import optuna
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Optional

# Darts and NeuralForecast imports
from darts.models import NHiTSModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mae as darts_mae
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss

# Project imports
import model_preprocessing as mp

def to_naive(ts_str: str):
    return pd.Timestamp(ts_str).tz_localize(None)

def train_nhits(csv_path: str, params: Dict[str, Any]):
    train_end = "2018-12-31 23:00:00+00:00"
    val_end = "2019-12-31 23:00:00+00:00"
    
    cfg = mp.default_feature_config()
    state, t_sc, v_sc, _ = mp.prepare_model_data(csv_path, to_naive(train_end), to_naive(val_end), cfg)
    
    model_params = {
        "input_chunk_length": 168,
        "output_chunk_length": 24,
        "batch_size": 32,
        "n_epochs": params.get("n_epochs", 10),
        "random_state": 42,
        "force_reset": True,
        "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        "optimizer_kwargs": {"lr": params["lr"], "weight_decay": params["weight_decay"]},
        "num_stacks": params["num_stacks"],
        "num_blocks": params["num_blocks"],
        "num_layers": params["num_layers"],
        "layer_widths": params["layer_widths"],
        "dropout": params["dropout"],
    }
    
    tp, vp = t_sc["past_covariates"], v_sc["past_covariates"]
    if t_sc["future_covariates"]: tp = tp.stack(t_sc["future_covariates"])
    if v_sc["future_covariates"]: vp = vp.stack(v_sc["future_covariates"])
    
    model = NHiTSModel(**model_params)
    model.fit(t_sc["target"], past_covariates=tp, val_series=v_sc["target"], val_past_covariates=vp)
    
    # We use validation loss (MSE by default in fit output or we calculate it)
    # For HPO, let's just get the last validation loss if possible, or evaluate
    preds = model.predict(n=24, series=t_sc["target"], past_covariates=tp, num_samples=1)
    # Simple evaluation on val set
    val_targets = v_sc["target"]
    # For speed in HPO, we might just use a subset or the internal val_loss from trainer if available.
    # But NHiTS fit doesn't return the loss easily without reaching into the trainer.
    # Let's do a simple evaluation.
    mae = model.backtest(v_sc["target"], past_covariates=vp, start=0.5, forecast_horizon=24, metric=darts_mae, retrain=False)
    return mae

def train_timesnet(csv_path: str, params: Dict[str, Any]):
    df_full = mp.load_and_validate_features(csv_path)
    nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"})
    nf_df["unique_id"] = "nordbyen"
    cfg = mp.default_feature_config()
    
    # TimesNet doesn't support hist_exog_list, so treat all exogenous as future (assumes weather is forecasted)
    num_ex = [c for c in (cfg.past_covariates_cols + cfg.future_covariates_cols) 
              if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
    
    nf_df = nf_df[["unique_id", "ds", "y"] + num_ex].dropna()
    nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
    
    train_end = "2018-12-31 23:00:00"
    val_end = "2019-12-31 23:00:00"
    
    train_df = nf_df[nf_df['ds'] <= to_naive(train_end)].reset_index(drop=True)
    val_df = nf_df[(nf_df['ds'] > to_naive(train_end)) & (nf_df['ds'] <= to_naive(val_end))].reset_index(drop=True)
    
    model = TimesNet(
        h=24, 
        input_size=168, 
        futr_exog_list=num_ex,  # All exogenous treated as future (weather assumed forecasted)
        scaler_type="robust",    # Robust scaling for exogenous variables
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
        max_steps=params.get("max_steps", 100),
        learning_rate=params["lr"],
        hidden_size=params["hidden_size"],
        conv_hidden_size=params["conv_hidden_size"],
        top_k=params["top_k"],
        dropout=params["dropout"]
    )
    
    nf = NeuralForecast(models=[model], freq='h')
    nf.fit(df=train_df, val_size=len(val_df))
    
    # Validation MAE on first 24h of validation set
    val_df_h = val_df.head(24).reset_index(drop=True)
    preds = nf.predict(df=train_df, futr_df=val_df_h.drop(columns=['y']))
    
    # Extract median prediction
    col = [c for c in preds.columns if c.endswith('-q-0.5') or c.endswith('median')][0]
    y_hat = preds[col].values
    y_true = val_df_h['y'].values
    mae = np.mean(np.abs(y_hat - y_true))
    return float(mae)

def objective(trial, model_type, csv_path):
    if model_type == "NHITS":
        params = {
            "num_stacks": trial.suggest_int("num_stacks", 1, 5),
            "num_blocks": trial.suggest_int("num_blocks", 1, 3),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "layer_widths": trial.suggest_categorical("layer_widths", [128, 256, 512, 1024]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "n_epochs": 10 # Keep relatively low for HPO speed
        }
        return train_nhits(csv_path, params)
    else: # TIMESNET
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "conv_hidden_size": trial.suggest_categorical("conv_hidden_size", [32, 64, 128, 256]),
            "top_k": trial.suggest_int("top_k", 1, 5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "max_steps": 500  # Default for TimesNet HPO
        }
        return train_timesnet(csv_path, params)

if __name__ == "__main__":
    model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "NHITS"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    test_mode = "--test" in sys.argv
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nordbyen_features_engineered.csv")
    
    def test_objective(trial):
        if model_name == "NHITS":
            params = {
                "num_stacks": 1, "num_blocks": 1, "num_layers": 1, "layer_widths": 128,
                "lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-5, "n_epochs": 1
            }
            return train_nhits(csv_path, params)
        else:
            params = {
                "hidden_size": 32, "conv_hidden_size": 32, "top_k": 1,
                "lr": 1e-3, "dropout": 0.1, "max_steps": 1
            }
            return train_timesnet(csv_path, params)

    study = optuna.create_study(direction="minimize")
    if test_mode:
        study.optimize(test_objective, n_trials=1)
    else:
        study.optimize(lambda trial: objective(trial, model_name, csv_path), n_trials=n_trials)
    
    print(f"\nBest trials for {model_name}:")
    print(study.best_trial.params)
    
    # Save best params
    os.makedirs("results", exist_ok=True)
    with open(f"results/best_params_{model_name}.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
