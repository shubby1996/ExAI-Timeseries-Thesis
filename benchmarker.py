import os
import sys
from datetime import datetime
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error
from properscoring import crps_ensemble

# Darts imports
from darts import TimeSeries
from darts.models import NHiTSModel, TFTModel
from darts.utils.likelihood_models import QuantileRegression

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss, MSE

# Project imports
import model_preprocessing as mp

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask): return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mape_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.sum(np.abs(y_true)), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)

def mase_denominator(series: np.ndarray, m: int = 24) -> float:
    series = np.asarray(series)
    if len(series) <= m: return np.nan
    diffs = np.abs(series[m:] - series[:-m])
    if len(diffs) == 0: return np.nan
    return float(np.mean(diffs))

def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    if scale is None or np.isnan(scale) or scale == 0: return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def calculate_winkler_score(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray, alpha: float = 0.2) -> float:
    """Winkler score (interval score) - penalizes width and misses.
    IS = (u - l) + (2/alpha) * max(0, l - y) + (2/alpha) * max(0, y - u)
    Lower is better.
    """
    if len(y_true) == 0: return np.nan
    width = y_high - y_low
    miss_below = np.maximum(0, y_low - y_true)
    miss_above = np.maximum(0, y_true - y_high)
    return float(np.mean(width + (2.0 / alpha) * (miss_below + miss_above)))

def calculate_calibration_curve(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate calibration curve: empirical vs nominal coverage at different quantile levels.
    Returns dict with 'nominal' and 'empirical' arrays for plotting."""
    if len(y_true) == 0: return {'nominal': np.array([]), 'empirical': np.array([])}
    
    # Define quantile levels to test
    quantile_levels = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    empirical_coverage = []
    
    for q in quantile_levels:
        # For a symmetric interval with coverage (1 - 2*q)
        # We approximate quantiles from our p10, p50, p90 interval
        # Simple approach: scale the interval
        nominal_cov = 1 - 2 * q
        width = y_high - y_low
        margin = width * q / 0.1  # 0.1 is the quantile level for p10/p90
        emp_low = y_low - margin
        emp_high = y_high + margin
        emp_cov = np.mean((y_true >= emp_low) & (y_true <= emp_high))
        empirical_coverage.append(emp_cov)
    
    return {'nominal': 1 - 2 * quantile_levels, 'empirical': np.array(empirical_coverage)}

def calculate_picp(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    if len(y_true) == 0: return 0.0
    within_interval = (y_true >= y_low) & (y_true <= y_high)
    return np.mean(within_interval) * 100

def calculate_miw(y_low: np.ndarray, y_high: np.ndarray) -> float:
    if len(y_low) == 0: return 0.0
    return np.mean(y_high - y_low)

def calculate_crps(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Calculate CRPS from ensemble samples.
    samples shape: (n_predictions, n_samples) or list of arrays
    """
    if len(y_true) == 0: return 0.0
    
    # Debug output
    print(f"[CRPS Debug] y_true length: {len(y_true)}")
    if isinstance(samples, list):
        print(f"[CRPS Debug] samples is list with length: {len(samples)}")
    else:
        print(f"[CRPS Debug] samples shape: {samples.shape}")
    
    # Safety check
    if isinstance(samples, list) and len(samples) != len(y_true):
        print(f"[CRPS ERROR] Mismatch! y_true has {len(y_true)} elements, samples has {len(samples)} elements")
        print(f"[CRPS ERROR] Using minimum length: {min(len(y_true), len(samples))}")
        n = min(len(y_true), len(samples))
        y_true = y_true[:n]
        samples = samples[:n]
    
    crps_values = []
    for i, obs in enumerate(y_true):
        if isinstance(samples, list):
            ensemble = samples[i]
        else:
            ensemble = samples[i] if samples.ndim > 1 else samples
        crps_values.append(crps_ensemble(obs, ensemble))
    return np.mean(crps_values)

def to_naive(ts_str: str):
    return pd.Timestamp(ts_str).tz_localize(None)

class ModelAdapter(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name, self.config = name, config
        self.model, self.state = None, None

    @abstractmethod
    def train(self, csv_path: str, train_end_str: str, val_end_str: str): pass
    @abstractmethod
    def evaluate(self, csv_path: str, test_start_str: str, n_predictions: int = 50) -> Tuple[Dict[str, float], pd.DataFrame]: pass

class DartsAdapter(ModelAdapter):
    def train(self, csv_path: str, train_end_str: str, val_end_str: str):
        print(f"\n[{self.name}] Training...")
        # Auto-detect water vs heat data (include Tommerby water set)
        lower_path = csv_path.lower()
        cfg = mp.water_feature_config() if ("water" in lower_path or "centrum" in lower_path or "tommerby" in lower_path) else mp.default_feature_config()
        self.state, t_sc, v_sc, _ = mp.prepare_model_data(csv_path, to_naive(train_end_str), to_naive(val_end_str), cfg)
        
        # Default core params
        cp = {
            "input_chunk_length": 168,
            "output_chunk_length": 24,
            "batch_size": 32,
            "n_epochs": self.config.get("n_epochs", 10),
            "random_state": 42,
            "force_reset": True,
             "pl_trainer_kwargs": {
                "logger": True,
                "enable_checkpointing": False,
                "default_root_dir": "lightning_logs"
            }
        }

        # Switch between quantile (probabilistic) and point (MSE) objectives
        if self.config.get("quantile", True):
            cp["likelihood"] = QuantileRegression(quantiles=[0.1, 0.5, 0.9])
        
        # Override with HPO results if available
        if "best_params" in self.config and self.config["best_params"] is not None:
            best = self.config["best_params"]
            print(f"  Using optimized hyperparameters from HPO")
            cp.update({
                "num_stacks": best.get("num_stacks", 3),
                "num_blocks": best.get("num_blocks", 1),
                "num_layers": best.get("num_layers", 2),
                "layer_widths": best.get("layer_widths", 512),
                "dropout": best.get("dropout", 0.1),
                "optimizer_kwargs": {"lr": best.get("lr", 1e-3), "weight_decay": best.get("weight_decay", 1e-5)}
            })
        else:
            print(f"  Using default hyperparameters (no HPO results found)")
        
        #It grabs the historical features (weather, lags, rolling stats) for both training (tp) and validation (vp).
        tp, vp = t_sc["past_covariates"], v_sc["past_covariates"]

        #Lines 81-82 (Stacking): It checks if you have future-known features (calendar variables like "hour", "day_of_week", "holidays").
        #.stack(): This function essentially glues the two time series together along the "feature" axis.
        #Before Stack: You might just have [Temperature, Wind].
        #After Stack: You have [Temperature, Wind, Hour, DayOfWeek, IsHoliday].
        if t_sc["future_covariates"]: tp = tp.stack(t_sc["future_covariates"])
        if v_sc["future_covariates"]: vp = vp.stack(v_sc["future_covariates"])
        
        self.model = NHiTSModel(**cp)
        self.model.fit(t_sc["target"], past_covariates=tp, val_series=v_sc["target"], val_past_covariates=vp)
        
        os.makedirs("models", exist_ok=True)
        self.model.save(os.path.join("models", f"{self.name}.pt"))
        with open(os.path.join("models", f"{self.name}_preprocessing_state.pkl"), "wb") as f: pickle.dump(self.state, f)

    def evaluate(self, csv_path: str, test_start_str: str, n_predictions: int = 50) -> Tuple[Dict[str, float], pd.DataFrame]:
        print(f"[{self.name}] Evaluating (Walk-forward)...")
        if not self.state:
            with open(os.path.join("models", f"{self.name}_preprocessing_state.pkl"), "rb") as f: self.state = pickle.load(f)
        df_full = mp.load_and_validate_features(csv_path)
        df_full.index = df_full.index.tz_localize(None)
        sc_dict = mp.apply_state_to_full_df(df_full, self.state)
        st, sp, sf = sc_dict["target"], sc_dict["past_covariates"], sc_dict["future_covariates"]
        ts_naive = to_naive(test_start_str)
        mase_scale = mase_denominator(df_full.loc[df_full.index < ts_naive, self.state.feature_config.target_col].values, m=24)
        mase_scale = mase_denominator(df_full.loc[df_full.index < ts_naive, self.state.feature_config.target_col].values, m=24)
        
        all_rows = []
        all_samples = []  # Store samples for CRPS
        all_actuals = []
        is_quantile = self.config.get("quantile", True)
        for i in range(n_predictions):
            ps = ts_naive + pd.Timedelta(hours=i * 24)
            ht = st[:ps - pd.Timedelta(hours=1)]
            if len(ht) < 168: continue
            as_sl = df_full[self.state.feature_config.target_col][ps : ps + pd.Timedelta(hours=23)]
            if len(as_sl) < 24: break
            
            hp = sp.stack(sf)[:ps - pd.Timedelta(hours=1)] if sf else sp[:ps-pd.Timedelta(hours=1)]
            preds = self.model.predict(
                n=24,
                series=ht,
                past_covariates=hp,
                num_samples=100 if is_quantile else 1
            )

            po = self.state.target_scaler.inverse_transform(preds)

            if is_quantile:
                p10 = po.quantile(0.1).values().flatten()
                p50 = po.quantile(0.5).values().flatten()
                p90 = po.quantile(0.9).values().flatten()
                samples = po.all_values(copy=False)[:, :, 0]  # (24, num_samples)
                all_samples.extend([samples[j, :] for j in range(samples.shape[0])])
            else:
                p50 = po.values().flatten()
                p10 = p50
                p90 = p50
                # For point forecasts, CRPS / interval metrics are not applicable
                all_samples.extend([np.array([v]) for v in p50])
            
            actuals = as_sl.values
            all_actuals.extend(actuals)
            times = as_sl.index
            
            for t, a, l, m, h in zip(times, actuals, p10, p50, p90):
                all_rows.append({"timestamp": t, "actual": a, "p10": l, "p50": m, "p90": h})
        
        pdf = pd.DataFrame(all_rows)
        metrics = {
            "MAE": mean_absolute_error(pdf["actual"], pdf["p50"]),
            "RMSE": np.sqrt(mean_squared_error(pdf["actual"], pdf["p50"])),
            "MAPE": calculate_mape(pdf["actual"].values, pdf["p50"].values),
            "MAPE_EPS": calculate_mape_eps(pdf["actual"].values, pdf["p50"].values),
            "sMAPE": calculate_smape(pdf["actual"].values, pdf["p50"].values),
            "WAPE": calculate_wape(pdf["actual"].values, pdf["p50"].values),
            "MASE": calculate_mase(pdf["actual"].values, pdf["p50"].values, mase_scale),
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "Winkler": calculate_winkler_score(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "CRPS": calculate_crps(np.array(all_actuals), all_samples) if is_quantile else np.nan
        }
        return metrics, pdf

class TFTAdapter(ModelAdapter):
    """Adapter for Temporal Fusion Transformer (TFT) from Darts."""
    def train(self, csv_path: str, train_end_str: str, val_end_str: str):
        print(f"\n[{self.name}] Training...")
        # Auto-detect water vs heat data (include Tommerby water set)
        lower_path = csv_path.lower()
        cfg = mp.water_feature_config() if ("water" in lower_path or "centrum" in lower_path or "tommerby" in lower_path) else mp.default_feature_config()
        self.state, t_sc, v_sc, _ = mp.prepare_model_data(csv_path, to_naive(train_end_str), to_naive(val_end_str), cfg)
        
        # Default TFT core params
        cp = {
            "input_chunk_length": 168,
            "output_chunk_length": 24,
            "batch_size": 32,
            "n_epochs": self.config.get("n_epochs", 100),
            "hidden_size": 64,
            "lstm_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "random_state": 42,
            "force_reset": True,
            "pl_trainer_kwargs": {
                "logger": True,
                "enable_checkpointing": False,
                "default_root_dir": "lightning_logs"
            }
        }

        # Switch between quantile (probabilistic) and point (MSE) objectives
        if self.config.get("quantile", True):
            cp["likelihood"] = QuantileRegression(quantiles=[0.1, 0.5, 0.9])
        
        # Override with HPO results if available
        if "best_params" in self.config and self.config["best_params"] is not None:
            best = self.config["best_params"]
            print(f"  Using optimized hyperparameters from HPO")
            cp.update({
                "hidden_size": best.get("hidden_size", 64),
                "lstm_layers": best.get("lstm_layers", 1),
                "num_attention_heads": best.get("num_attention_heads", 4),
                "dropout": best.get("dropout", 0.1),
                "optimizer_kwargs": {"lr": best.get("lr", 1e-3)}
            })
        else:
            print(f"  Using default hyperparameters (no HPO results found)")
        
        # TFT uses separate past and future covariates
        tp, vp = t_sc["past_covariates"], v_sc["past_covariates"]
        tf, vf = t_sc["future_covariates"], v_sc["future_covariates"]
        
        self.model = TFTModel(**cp)
        self.model.fit(
            t_sc["target"], 
            past_covariates=tp, 
            future_covariates=tf,
            val_series=v_sc["target"], 
            val_past_covariates=vp,
            val_future_covariates=vf
        )
        
        os.makedirs("models", exist_ok=True)
        self.model.save(os.path.join("models", f"{self.name}.pt"))
        with open(os.path.join("models", f"{self.name}_preprocessing_state.pkl"), "wb") as f: pickle.dump(self.state, f)

    def evaluate(self, csv_path: str, test_start_str: str, n_predictions: int = 50) -> Tuple[Dict[str, float], pd.DataFrame]:
        print(f"[{self.name}] Evaluating (Walk-forward)...")
        if not self.state:
            with open(os.path.join("models", f"{self.name}_preprocessing_state.pkl"), "rb") as f: self.state = pickle.load(f)
        df_full = mp.load_and_validate_features(csv_path)
        df_full.index = df_full.index.tz_localize(None)
        sc_dict = mp.apply_state_to_full_df(df_full, self.state)
        st, sp, sf = sc_dict["target"], sc_dict["past_covariates"], sc_dict["future_covariates"]
        ts_naive = to_naive(test_start_str)
        
        all_rows = []
        all_samples = []
        all_actuals = []
        is_quantile = self.config.get("quantile", True)
        for i in range(n_predictions):
            ps = ts_naive + pd.Timedelta(hours=i * 24)
            ht = st[:ps - pd.Timedelta(hours=1)]
            if len(ht) < 168: continue
            as_sl = df_full[self.state.feature_config.target_col][ps : ps + pd.Timedelta(hours=23)]
            if len(as_sl) < 24: break
            
            # TFT needs both past and future covariates separately
            hp = sp[:ps - pd.Timedelta(hours=1)]
            hf = sf[:ps + pd.Timedelta(hours=23)] if sf else None
            
            preds = self.model.predict(
                n=24,
                series=ht,
                past_covariates=hp,
                future_covariates=hf,
                num_samples=100 if is_quantile else 1
            )

            po = self.state.target_scaler.inverse_transform(preds)

            if is_quantile:
                p10 = po.quantile(0.1).values().flatten()
                p50 = po.quantile(0.5).values().flatten()
                p90 = po.quantile(0.9).values().flatten()
                samples = po.all_values(copy=False)[:, :, 0]
                all_samples.extend([samples[j, :] for j in range(samples.shape[0])])
            else:
                p50 = po.values().flatten()
                p10 = p50
                p90 = p50
                all_samples.extend([np.array([v]) for v in p50])
            
            actuals = as_sl.values
            all_actuals.extend(actuals)
            times = as_sl.index
            
            for t, a, l, m, h in zip(times, actuals, p10, p50, p90):
                all_rows.append({"timestamp": t, "actual": a, "p10": l, "p50": m, "p90": h})
        
        pdf = pd.DataFrame(all_rows)
        metrics = {
            "MAE": mean_absolute_error(pdf["actual"], pdf["p50"]),
            "RMSE": np.sqrt(mean_squared_error(pdf["actual"], pdf["p50"])),
            "MAPE": calculate_mape(pdf["actual"].values, pdf["p50"].values),
            "MAPE_EPS": calculate_mape_eps(pdf["actual"].values, pdf["p50"].values),
            "sMAPE": calculate_smape(pdf["actual"].values, pdf["p50"].values),
            "WAPE": calculate_wape(pdf["actual"].values, pdf["p50"].values),
            "MASE": calculate_mase(pdf["actual"].values, pdf["p50"].values, mase_scale),
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "Winkler": calculate_winkler_score(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "CRPS": calculate_crps(np.array(all_actuals), all_samples) if is_quantile else np.nan
        }
        return metrics, pdf

class NeuralForecastAdapter(ModelAdapter):
    def _prepare_df(self, csv_path):
        df_full = mp.load_and_validate_features(csv_path)
        lower_path = csv_path.lower()
        nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"} if ("heat" in lower_path or "nordbyen" in lower_path) else {"timestamp": "ds", "water_consumption": "y"})
        nf_df["unique_id"] = "nordbyen" if "nordbyen" in lower_path else ("tommerby" if "tommerby" in lower_path else "centrum")
        # Auto-detect water vs heat data for feature config (include Tommerby water set)
        cfg = mp.water_feature_config() if ("water" in lower_path or "centrum" in lower_path or "tommerby" in lower_path) else mp.default_feature_config()
        
        # TimesNet doesn't support hist_exog_list, so treat all exogenous as future (assumes weather is forecasted)
        futr_ex = [c for c in (cfg.past_covariates_cols + cfg.future_covariates_cols) 
                   if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
        
        all_cols = ["unique_id", "ds", "y"] + futr_ex
        return nf_df[all_cols].dropna(), futr_ex

    def train(self, csv_path: str, train_end_str: str, val_end_str: str):
        print(f"\n[{self.name}] Training...")
        nf_df, futr_ex = self._prepare_df(csv_path)
        nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
        train_df = nf_df[nf_df['ds'] <= to_naive(train_end_str)].reset_index(drop=True)
        
        # Calculate max_steps based on epochs
        # NeuralForecast uses max_steps (total gradient updates), not epochs
        # To match NHITS epoch behavior: max_steps = epochs * (samples / batch_size)
        batch_size = 32
        n_epochs_desired = self.config.get("n_epochs", 50)
        n_samples = len(train_df)
        steps_per_epoch = max(1, n_samples // batch_size)
        max_steps_calculated = n_epochs_desired * steps_per_epoch
        
        print(f"  Training setup: {n_epochs_desired} epochs × {steps_per_epoch} steps/epoch = {max_steps_calculated} total steps")
        
        # Baseline core params
        model_params = {
            "h": 24,
            "input_size": 168,
            "futr_exog_list": futr_ex,  # All exogenous treated as future (weather assumed forecasted)
            "scaler_type": "robust",     # Robust scaling for exogenous variables
            "loss": MQLoss(quantiles=[0.1, 0.5, 0.9]) if self.config.get("quantile", True) else MSE(),
            "max_steps": max_steps_calculated,
            "batch_size": batch_size,
            # PyTorch Lightning Trainer parameters (passed directly, not via trainer_kwargs)
            "logger": True,
            "enable_checkpointing": False
        }
        
        # Override with HPO results if available
        if "best_params" in self.config and self.config["best_params"] is not None:
            best = self.config["best_params"]
            print(f"  Using optimized hyperparameters from HPO")
            model_params.update({
                "hidden_size": best.get("hidden_size", 64),
                "conv_hidden_size": best.get("conv_hidden_size", 64),
                "top_k": best.get("top_k", 2),
                "learning_rate": best.get("lr", 1e-3),
                "dropout": best.get("dropout", 0.1)
            })
        else:
            print(f"  Using default hyperparameters (no HPO results found)")
            
        model = TimesNet(**model_params)
        nf = NeuralForecast(models=[model], freq='h')
        nf.fit(df=train_df)
        self.model = nf
        nf.save(path=os.path.join("models", self.name), overwrite=True)

    def evaluate(self, csv_path: str, test_start_str: str, n_predictions: int = 50) -> Tuple[Dict[str, float], pd.DataFrame]:
        print(f"[{self.name}] Evaluating (Walk-forward)...")
        nf_df, futr_ex = self._prepare_df(csv_path)
        nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
        ts_naive = to_naive(test_start_str)
        mase_scale = mase_denominator(nf_df.loc[nf_df['ds'] < ts_naive, 'y'].values, m=24)
        
        all_rows = []
        all_samples = []  # Approximate samples from quantiles for CRPS
        all_actuals = []
        is_quantile = self.config.get("quantile", True)
        for i in range(n_predictions):
            ps = ts_naive + pd.Timedelta(hours=i * 24)
            hist_df = nf_df[(nf_df['ds'] >= ps - pd.Timedelta(hours=168)) & (nf_df['ds'] < ps)]
            fut_df = nf_df[(nf_df['ds'] >= ps) & (nf_df['ds'] <= ps + pd.Timedelta(hours=23))]
            if len(hist_df) < 168 or len(fut_df) < 24: break
            
            fcst = self.model.predict(df=hist_df.reset_index(drop=True), futr_df=fut_df.reset_index(drop=True))
            def find_col(suffixes):
                for s in suffixes:
                    for c in fcst.columns:
                        if c.endswith(s): return c
                return None

            if is_quantile:
                cm, cl, ch = find_col(['median', '-q-0.5']), find_col(['-lo-80.0', '-q-0.1']), find_col(['-hi-80.0', '-q-0.9'])
                p50 = fcst[cm].values.tolist() if cm else fcst.iloc[:, -1].values.tolist()
                p10 = fcst[cl].values.tolist() if cl else p50
                p90 = fcst[ch].values.tolist() if ch else p50
            else:
                # Deterministic TimesNet outputs a single column (forecast mean)
                cm = find_col(['TimesNet', 'y_hat'])
                p50 = fcst[cm].values.tolist() if cm else fcst.iloc[:, -1].values.tolist()
                p10 = p50
                p90 = p50
            actuals = fut_df['y'].values.tolist()
            times = fut_df['ds'].tolist()
            
            if is_quantile:
                # Approximate samples from quantiles for CRPS calculation
                for l, m, h in zip(p10, p50, p90):
                    # Ensure quantiles are properly ordered and avoid negative scale
                    scale = max(abs(h - l) / 2.56, 1e-6)  # Use abs() and minimum threshold
                    samples = np.random.normal(m, scale, 100)
                    all_samples.append(samples)
            else:
                for m in p50:
                    all_samples.append(np.array([m]))
            all_actuals.extend(actuals)
            
            for t, a, l, m, h in zip(times, actuals, p10, p50, p90):
                all_rows.append({"timestamp": t, "actual": a, "p10": l, "p50": m, "p90": h})
        
        pdf = pd.DataFrame(all_rows)
        metrics = {
            "MAE": mean_absolute_error(pdf["actual"], pdf["p50"]),
            "RMSE": np.sqrt(mean_squared_error(pdf["actual"], pdf["p50"])),
            "MAPE": calculate_mape(pdf["actual"].values, pdf["p50"].values),
            "MAPE_EPS": calculate_mape_eps(pdf["actual"].values, pdf["p50"].values),
            "sMAPE": calculate_smape(pdf["actual"].values, pdf["p50"].values),
            "WAPE": calculate_wape(pdf["actual"].values, pdf["p50"].values),
            "MASE": calculate_mase(pdf["actual"].values, pdf["p50"].values, mase_scale),
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "Winkler": calculate_winkler_score(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "CRPS": calculate_crps(np.array(all_actuals), all_samples) if is_quantile else np.nan
        }
        return metrics, pdf

class Benchmarker:
    def __init__(self, csv_path: str, models_to_run: List[str], dataset: str = None):
        self.csv_path, self.results = csv_path, []
        self.models_to_run = [m.upper() for m in models_to_run]
        # Infer dataset from path if not provided
        if dataset is None:
            lower_path = csv_path.lower()
            if 'nordbyen' in lower_path:
                self.dataset = 'Heat (Nordbyen)'
            elif 'centrum' in lower_path:
                self.dataset = 'Water (Centrum)'
            elif 'tommerby' in lower_path:
                self.dataset = 'Water (Tommerby)'
            else:
                self.dataset = 'Unknown'
        else:
            self.dataset = dataset
        
        # Determine dataset-specific results folder
        lower_path = csv_path.lower()
        if 'nordbyen' in lower_path or 'heat' in self.dataset.lower():
            self.dataset_results_folder = 'nordbyen_heat_benchmark/results'
        elif 'tommerby' in lower_path or 'tommerby' in self.dataset.lower():
            self.dataset_results_folder = 'water_tommerby_benchmark/results'
        elif 'centrum' in lower_path or ('water' in self.dataset.lower() and 'centrum' in self.dataset.lower()):
            self.dataset_results_folder = 'water_centrum_benchmark/results'
        else:
            self.dataset_results_folder = 'results/unknown_dataset'
        
        # Get job ID from environment (SLURM) or use 'local' for local runs
        self.job_id = os.environ.get('SLURM_JOB_ID', 'local')
        
        # Load optimized params if they exist
        nhits_best = self._load_json("results/best_params_NHITS.json")
        timesnet_best = self._load_json("results/best_params_TIMESNET.json")
        tft_best = self._load_json("results/best_params_TFT.json")
        
        self.configs = {
            "NHITS_Q": {"type": "NHITS", "quantile": True, "n_epochs": 100, "best_params": nhits_best},
            "NHITS_MSE": {"type": "NHITS", "quantile": False, "n_epochs": 100, "best_params": None},
            "TIMESNET_Q": {"type": "TIMESNET", "quantile": True, "n_epochs": 150, "best_params": timesnet_best},
            "TIMESNET_MSE": {"type": "TIMESNET", "quantile": False, "n_epochs": 150, "best_params": None},
            "TFT_Q": {"type": "TFT", "quantile": True, "n_epochs": 100, "best_params": tft_best},
            "TFT_MSE": {"type": "TFT", "quantile": False, "n_epochs": 100, "best_params": None},
            # Backward-compatible aliases
            "NHITS": {"type": "NHITS", "quantile": True, "n_epochs": 100, "best_params": nhits_best},
            "TIMESNET": {"type": "TIMESNET", "quantile": True, "n_epochs": 150, "best_params": timesnet_best},
            "TFT": {"type": "TFT", "quantile": True, "n_epochs": 100, "best_params": tft_best},
        }

    def _load_json(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f: return json.load(f)
        return None
    
    def _print_improvement_summary(self, history_file: str):
        """Print summary showing improvement trends over time."""
        try:
            history = pd.read_csv(history_file)
            print("\n" + "="*70)
            print("IMPROVEMENT SUMMARY (vs. Previous Best)")
            print("="*70)
            
            for model in history['Model'].unique():
                model_history = history[history['Model'] == model].sort_values('run_date')
                if len(model_history) < 2:
                    print(f"\n{model}: First run - no comparison available")
                    continue
                
                current = model_history.iloc[-1]
                previous_best = model_history.iloc[:-1]
                
                print(f"\n{model}:")
                for metric in ['MAE', 'RMSE', 'MAPE', 'MIW', 'CRPS']:
                    if metric in current:
                        curr_val = current[metric]
                        prev_best_val = previous_best[metric].min()
                        
                        if curr_val < prev_best_val:
                            improvement = ((prev_best_val - curr_val) / prev_best_val) * 100
                            print(f"  {metric:8s}: {curr_val:.4f} (↓ {improvement:.2f}% improvement ✓)")
                        elif curr_val > prev_best_val:
                            degradation = ((curr_val - prev_best_val) / prev_best_val) * 100
                            print(f"  {metric:8s}: {curr_val:.4f} (↑ {degradation:.2f}% worse)")
                        else:
                            print(f"  {metric:8s}: {curr_val:.4f} (no change)")
                
                print(f"  Total runs: {len(model_history)}")
                print(f"  Best run date: {previous_best.loc[previous_best['MAE'].idxmin(), 'run_date']}")
                
        except Exception as e:
            print(f"\n⚠️  Could not generate improvement summary: {e}")

    def run(self):
        print("\n" + "="*70)
        print("BENCHMARKER CONFIGURATION")
        print("="*70)
        for mk in self.models_to_run:
            if mk not in self.configs: continue
            cfg = self.configs[mk]
            has_best = cfg.get("best_params") is not None
            hpo_status = "✓ Using HPO best params" if has_best else "✗ No HPO params (using defaults)"
            print(f"{mk}: n_epochs={cfg.get('n_epochs', 10)}, HPO Status: {hpo_status}")
        print("="*70 + "\n")
        
        for mk in self.models_to_run:
            if mk not in self.configs: continue
            cfg = self.configs[mk]
            # Select appropriate adapter based on model type
            if cfg["type"].upper() == "TIMESNET":
                adapter = NeuralForecastAdapter(mk, cfg)
            elif cfg["type"].upper() == "TFT":
                adapter = TFTAdapter(mk, cfg)
            else:  # NHITS and other Darts models
                adapter = DartsAdapter(mk, cfg)
            adapter.train(self.csv_path, "2018-12-31 23:00:00+00:00", "2019-12-31 23:00:00+00:00")
            metrics, pdf = adapter.evaluate(self.csv_path, "2020-01-01 00:00:00+00:00")
            metrics["Model"] = mk; self.results.append(metrics)
            
            # Save predictions to dataset-specific folder with job ID
            os.makedirs(self.dataset_results_folder, exist_ok=True)
            predictions_filename = f"{mk}_predictions_{self.job_id}.csv"
            predictions_path = os.path.join(self.dataset_results_folder, predictions_filename)
            pdf.to_csv(predictions_path, index=False)
            print(f"Saved full predictions for {mk} to {predictions_path}")
            
        report_df = pd.DataFrame(self.results)
        print("\n" + "="*70 + "\nBENCHMARK RESULTS\n" + "="*70)
        print(report_df.to_string(index=False))
        
        # Save results with timestamp for historical tracking
        os.makedirs("results", exist_ok=True)
        os.makedirs(self.dataset_results_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset identifier for filename (remove spaces and parentheses)
        dataset_id = self.dataset.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Save timestamped copy to dataset-specific folder
        timestamped_filename = f"benchmark_results_{timestamp}_{dataset_id}_{self.job_id}.csv"
        timestamped_path = os.path.join(self.dataset_results_folder, timestamped_filename)
        report_df.to_csv(timestamped_path, index=False)
        print(f"\n✓ Results saved:")
        print(f"  - Timestamped: {timestamped_path}")
        
        # Append to history file with metadata
        history_file = "results/benchmark_history.csv"
        for result in self.results:
            result_with_meta = result.copy()
            result_with_meta['dataset'] = self.dataset
            result_with_meta['timestamp'] = timestamp
            result_with_meta['run_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_with_meta['n_epochs'] = self.configs[result['Model']].get('n_epochs', 'N/A')
            result_with_meta['has_hpo'] = self.configs[result['Model']].get('best_params') is not None
            
            history_df = pd.DataFrame([result_with_meta])
            if os.path.exists(history_file):
                history_df.to_csv(history_file, mode='a', header=False, index=False)
            else:
                history_df.to_csv(history_file, index=False)
        
        print(f"  - History: {history_file} (cumulative)")
        
        # Print summary of improvements if history exists
        if os.path.exists(history_file):
            self._print_improvement_summary(history_file)

if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["NHITS_Q", "NHITS_MSE", "TIMESNET_Q", "TIMESNET_MSE"]
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processing", "nordbyen_processing", "nordbyen_features_engineered.csv")
    Benchmarker(csv_path, models).run()
