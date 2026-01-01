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
from darts.models import NHiTSModel
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
        # Auto-detect water vs heat data
        cfg = mp.water_feature_config() if "water" in csv_path.lower() or "centrum" in csv_path.lower() else mp.default_feature_config()
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
            # Use calibrated quantiles from Stage 2 if available, else default
            default_quantiles = [0.1, 0.5, 0.9]
            if "best_params" in self.config and self.config["best_params"] is not None:
                best = self.config["best_params"]
                # Check for Stage 2 calibrated quantiles
                if "calibrated_quantiles" in best:
                    quantiles = best["calibrated_quantiles"]
                    print(f"  Using calibrated quantiles from Stage 2: {quantiles}")
                else:
                    quantiles = default_quantiles
                    print(f"  Using default quantiles (Stage 2 not run): {quantiles}")
            else:
                quantiles = default_quantiles
                print(f"  Using default quantiles (no HPO): {quantiles}")
            cp["likelihood"] = QuantileRegression(quantiles=quantiles)
        
        # Override with HPO results if available
        if "best_params" in self.config and self.config["best_params"] is not None:
            best = self.config["best_params"]
            # Extract architecture params (may be nested under 'architecture_params' in Stage 2)
            arch_params = best.get("architecture_params", best)
            print(f"  Using optimized hyperparameters from HPO")
            cp.update({
                "num_stacks": arch_params.get("num_stacks", 3),
                "num_blocks": arch_params.get("num_blocks", 1),
                "num_layers": arch_params.get("num_layers", 2),
                "layer_widths": arch_params.get("layer_widths", 512),
                "dropout": arch_params.get("dropout", 0.1),
                "optimizer_kwargs": {"lr": arch_params.get("lr", 1e-4), "weight_decay": arch_params.get("weight_decay", 1e-5)}  # FIXED: lr was 1e-3, NeuralForecast default is 1e-4
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
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "CRPS": calculate_crps(np.array(all_actuals), all_samples) if is_quantile else np.nan
        }
        return metrics, pdf

class NeuralForecastAdapter(ModelAdapter):
    def _prepare_df(self, csv_path):
        df_full = mp.load_and_validate_features(csv_path)
        nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"} if "heat" in csv_path.lower() or "nordbyen" in csv_path.lower() else {"timestamp": "ds", "water_consumption": "y"})
        nf_df["unique_id"] = "nordbyen" if "nordbyen" in csv_path.lower() else "centrum"
        # Auto-detect water vs heat data for feature config
        cfg = mp.water_feature_config() if "water" in csv_path.lower() or "centrum" in csv_path.lower() else mp.default_feature_config()
        
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
        
        # Determine quantiles (use calibrated if available from Stage 2)
        default_quantiles = [0.1, 0.5, 0.9]
        if self.config.get("quantile", True):
            if "best_params" in self.config and self.config["best_params"] is not None:
                best = self.config["best_params"]
                # Check for Stage 2 calibrated quantiles
                if "calibrated_quantiles" in best:
                    quantiles = best["calibrated_quantiles"]
                    print(f"  Using calibrated quantiles from Stage 2: {quantiles}")
                else:
                    quantiles = default_quantiles
                    print(f"  Using default quantiles (Stage 2 not run): {quantiles}")
            else:
                quantiles = default_quantiles
                print(f"  Using default quantiles (no HPO): {quantiles}")
        else:
            quantiles = default_quantiles
        
        # Baseline core params
        model_params = {
            "h": 24,
            "input_size": 168,
            "futr_exog_list": futr_ex,  # All exogenous treated as future (weather assumed forecasted)
            "scaler_type": "robust",     # Robust scaling for exogenous variables
            "loss": MQLoss(level=quantiles) if self.config.get("quantile", True) else MSE(),
            "max_steps": max_steps_calculated,
            "batch_size": batch_size,
            # PyTorch Lightning Trainer parameters (passed directly, not via trainer_kwargs)
            "logger": True,
            "enable_checkpointing": False
        }
        
        # Override with HPO results if available
        if "best_params" in self.config and self.config["best_params"] is not None:
            best = self.config["best_params"]
            # Extract architecture params (may be nested under 'architecture_params' in Stage 2)
            arch_params = best.get("architecture_params", best)
            print(f"  Using optimized hyperparameters from HPO")
            print(f"  DEBUG best_params keys: {list(best.keys())}")
            print(f"  DEBUG arch_params keys: {list(arch_params.keys())}")
            print(f"  DEBUG arch_params values: {arch_params}")
            
            # CRITICAL FIX: Use correct NeuralForecast defaults as fallback
            model_params.update({
                "hidden_size": arch_params.get("hidden_size", 64),
                "conv_hidden_size": arch_params.get("conv_hidden_size", 64),
                "top_k": arch_params.get("top_k", 5),  # FIXED: was 2, NeuralForecast default is 5
                "learning_rate": arch_params.get("lr", 1e-4),  # FIXED: was 1e-3, NeuralForecast default is 1e-4
                "dropout": arch_params.get("dropout", 0.1)
            })
            print(f"  DEBUG final model_params: hidden_size={model_params['hidden_size']}, top_k={model_params['top_k']}, lr={model_params['learning_rate']}, dropout={model_params['dropout']}")
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
            
            # DEBUG: Print columns on first iteration
            if i == 0:
                print(f"  DEBUG: Prediction columns: {fcst.columns.tolist()}")
                print(f"  DEBUG: First row of predictions: {fcst.iloc[0].to_dict()}")
            
            def find_col(suffixes):
                for s in suffixes:
                    for c in fcst.columns:
                        if c.endswith(s): return c
                return None

            if is_quantile:
                # Search for quantile columns - handle multiple naming conventions:
                # - NeuralForecast MQLoss: ModelName-lo-0.1, ModelName-median, ModelName-hi-0.9
                # - Legacy format: ModelName-lo-80.0, ModelName-median, ModelName-hi-80.0
                # - Alternative: ModelName-q-0.1, ModelName-q-0.5, ModelName-q-0.9
                cm = find_col(['median', '-median', '-q-0.5'])
                cl = find_col(['-lo-0.1', '-lo-80.0', '-q-0.1'])
                ch = find_col(['-hi-0.9', '-hi-80.0', '-q-0.9'])
                
                # DEBUG: Print what columns were found AND their actual values
                if i == 0:
                    print(f"  DEBUG: Found median column: {cm}")
                    print(f"  DEBUG: Found lo column: {cl}")
                    print(f"  DEBUG: Found hi column: {ch}")
                    if cl and ch and cm:
                        # Print first value from each to verify correct order
                        lo_val = fcst[cl].iloc[0]
                        med_val = fcst[cm].iloc[0]
                        hi_val = fcst[ch].iloc[0]
                        print(f"  DEBUG: First prediction values:")
                        print(f"         {cl} = {lo_val:.6f}")
                        print(f"         {cm} = {med_val:.6f}")
                        print(f"         {ch} = {hi_val:.6f}")
                        is_correct = lo_val < med_val < hi_val
                        if is_correct:
                            print(f"  DEBUG: ✓ Quantile order is CORRECT (lo < med < hi)")
                        else:
                            print(f"  DEBUG: ✗ Quantile order is WRONG - model may have training issue!")
                
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
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values) if is_quantile else np.nan,
            "CRPS": calculate_crps(np.array(all_actuals), all_samples) if is_quantile else np.nan
        }
        return metrics, pdf

class Benchmarker:
    def __init__(self, csv_path: str, models_to_run: List[str], dataset: str = None, results_dir: str = "results"):
        self.csv_path, self.results = csv_path, []
        self.models_to_run = [m.upper() for m in models_to_run]
        self.results_dir = results_dir  # Dataset-specific results directory
        # Infer dataset from path if not provided
        if dataset is None:
            if 'nordbyen' in csv_path.lower():
                self.dataset = 'Heat (Nordbyen)'
            elif 'centrum' in csv_path.lower():
                self.dataset = 'Water (Centrum)'
            else:
                self.dataset = 'Unknown'
        else:
            self.dataset = dataset
        
        # DISABLED: Do not load HPO parameters - use defaults only
        # Stage 2 calibrated quantiles were causing model failures
        # If you want to re-enable HPO, uncomment the lines below:
        # nhits_q_best = self._load_json(f"{results_dir}/best_params_NHITS_Q.json") or self._load_json("results/best_params_NHITS_Q.json")
        # timesnet_q_best = self._load_json(f"{results_dir}/best_params_TIMESNET_Q.json") or self._load_json("results/best_params_TIMESNET_Q.json")
        nhits_q_best = None
        timesnet_q_best = None
        
        self.configs = {
            "NHITS_Q": {"type": "NHITS", "quantile": True, "n_epochs": 100, "best_params": nhits_q_best},
            "NHITS_MSE": {"type": "NHITS", "quantile": False, "n_epochs": 100, "best_params": None},
            "TIMESNET_Q": {"type": "TIMESNET", "quantile": True, "n_epochs": 150, "best_params": timesnet_q_best},
            "TIMESNET_MSE": {"type": "TIMESNET", "quantile": False, "n_epochs": 150, "best_params": None},
            # Backward-compatible aliases
            "NHITS": {"type": "NHITS", "quantile": True, "n_epochs": 100, "best_params": nhits_q_best},
            "TIMESNET": {"type": "TIMESNET", "quantile": True, "n_epochs": 150, "best_params": timesnet_q_best},
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
            adapter = NeuralForecastAdapter(mk, cfg) if cfg["type"].upper() == "TIMESNET" else DartsAdapter(mk, cfg)
            adapter.train(self.csv_path, "2018-12-31 23:00:00+00:00", "2019-12-31 23:00:00+00:00")
            metrics, pdf = adapter.evaluate(self.csv_path, "2020-01-01 00:00:00+00:00")
            metrics["Model"] = mk; self.results.append(metrics)
            
            # Save predictions to both project root and dataset-specific folders
            os.makedirs("results", exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            pdf.to_csv(f"results/{mk}_predictions.csv", index=False)
            pdf.to_csv(f"{self.results_dir}/{mk}_predictions.csv", index=False)
            print(f"Saved full predictions for {mk} to results/{mk}_predictions.csv")
            
        report_df = pd.DataFrame(self.results)
        print("\n" + "="*70 + "\nBENCHMARK RESULTS\n" + "="*70)
        print(report_df.to_string(index=False))
        
        # Save results with timestamp for historical tracking
        os.makedirs("results", exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save current results (overwrite) - to both locations
        report_df.to_csv("results/benchmark_results.csv", index=False)
        report_df.to_csv(f"{self.results_dir}/benchmark_results.csv", index=False)
        
        # Save timestamped copy for history - to both locations
        report_df.to_csv(f"results/benchmark_results_{timestamp}.csv", index=False)
        report_df.to_csv(f"{self.results_dir}/benchmark_results_{timestamp}.csv", index=False)
        print(f"\n✓ Results saved:")
        print(f"  - Current: results/benchmark_results.csv")
        print(f"  - Current: {self.results_dir}/benchmark_results.csv")
        print(f"  - History: results/benchmark_results_{timestamp}.csv")
        print(f"  - History: {self.results_dir}/benchmark_results_{timestamp}.csv")
        
        # Append to history file with metadata - to both locations
        history_file = "results/benchmark_history.csv"
        dataset_history_file = f"{self.results_dir}/benchmark_history.csv"
        for result in self.results:
            result_with_meta = result.copy()
            result_with_meta['dataset'] = self.dataset
            result_with_meta['timestamp'] = timestamp
            result_with_meta['run_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_with_meta['n_epochs'] = self.configs[result['Model']].get('n_epochs', 'N/A')
            result_with_meta['has_hpo'] = self.configs[result['Model']].get('best_params') is not None
            
            history_df = pd.DataFrame([result_with_meta])
            # Save to project root
            if os.path.exists(history_file):
                history_df.to_csv(history_file, mode='a', header=False, index=False)
            else:
                history_df.to_csv(history_file, index=False)
            # Save to dataset-specific folder
            if os.path.exists(dataset_history_file):
                history_df.to_csv(dataset_history_file, mode='a', header=False, index=False)
            else:
                history_df.to_csv(dataset_history_file, index=False)
        
        print(f"  - History: {history_file} (cumulative)")
        print(f"  - History: {dataset_history_file} (cumulative)")
        
        # Print summary of improvements if history exists
        if os.path.exists(history_file):
            self._print_improvement_summary(history_file)

if __name__ == "__main__":
    # Usage: python benchmarker.py [dataset] [models...]
    # Examples:
    #   python benchmarker.py water_centrum NHITS_Q TIMESNET_Q
    #   python benchmarker.py nordbyen_heat NHITS_Q NHITS_MSE
    #   python benchmarker.py  # defaults to nordbyen_heat with all models
    
    if len(sys.argv) > 1 and sys.argv[1] in ['water_centrum', 'nordbyen_heat']:
        dataset_name = sys.argv[1]
        models = sys.argv[2:] if len(sys.argv) > 2 else ["NHITS_Q", "NHITS_MSE", "TIMESNET_Q", "TIMESNET_MSE"]
    else:
        dataset_name = "nordbyen_heat"
        models = sys.argv[1:] if len(sys.argv) > 1 else ["NHITS_Q", "NHITS_MSE", "TIMESNET_Q", "TIMESNET_MSE"]
    
    # Configure dataset-specific paths
    if dataset_name == "water_centrum":
        csv_path = "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv"
        dataset_display = "Water (Centrum)"
        results_dir = "water_centrum_benchmark/results"
    else:  # nordbyen_heat
        csv_path = "processing/nordbyen_processing/nordbyen_features_engineered.csv"
        dataset_display = "Heat (Nordbyen)"
        results_dir = "nordbyen_heat_benchmark/results"
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKER - {dataset_display}")
    print(f"{'='*70}")
    print(f"Data: {csv_path}")
    print(f"Models: {', '.join(models)}")
    print(f"Results: {results_dir}")
    print(f"{'='*70}\n")
    
    Benchmarker(csv_path, models, dataset=dataset_display, results_dir=results_dir).run()

