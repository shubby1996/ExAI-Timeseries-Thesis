import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Darts imports
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.utils.likelihood_models import QuantileRegression

# NeuralForecast imports
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss

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
        cfg = mp.default_feature_config()
        self.state, t_sc, v_sc, _ = mp.prepare_model_data(csv_path, to_naive(train_end_str), to_naive(val_end_str), cfg)
        
        # Default core params
        cp = {
            "input_chunk_length": 168,
            "output_chunk_length": 24,
            "batch_size": 32,
            "n_epochs": self.config.get("n_epochs", 10),
            "random_state": 42,
            "force_reset": True,
            "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9])
        }
        
        # Override with HPO results if available
        if "best_params" in self.config:
            best = self.config["best_params"]
            cp.update({
                "num_stacks": best.get("num_stacks", 3),
                "num_blocks": best.get("num_blocks", 1),
                "num_layers": best.get("num_layers", 2),
                "layer_widths": best.get("layer_widths", 512),
                "dropout": best.get("dropout", 0.1),
                "optimizer_kwargs": {"lr": best.get("lr", 1e-3), "weight_decay": best.get("weight_decay", 1e-5)}
            })
        
        tp, vp = t_sc["past_covariates"], v_sc["past_covariates"]
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
        for i in range(n_predictions):
            ps = ts_naive + pd.Timedelta(hours=i * 24)
            ht = st[:ps - pd.Timedelta(hours=1)]
            if len(ht) < 168: continue
            as_sl = df_full['heat_consumption'][ps : ps + pd.Timedelta(hours=23)]
            if len(as_sl) < 24: break
            
            hp = sp.stack(sf)[:ps - pd.Timedelta(hours=1)] if sf else sp[:ps-pd.Timedelta(hours=1)]
            preds = self.model.predict(n=24, series=ht, past_covariates=hp, num_samples=100) # Increased samples for plotter
            
            po = self.state.target_scaler.inverse_transform(preds)
            p10 = po.quantile(0.1).values().flatten()
            p50 = po.quantile(0.5).values().flatten()
            p90 = po.quantile(0.9).values().flatten()
            actuals = as_sl.values
            times = as_sl.index
            
            for t, a, l, m, h in zip(times, actuals, p10, p50, p90):
                all_rows.append({"timestamp": t, "actual": a, "p10": l, "p50": m, "p90": h})
        
        pdf = pd.DataFrame(all_rows)
        metrics = {
            "MAE": mean_absolute_error(pdf["actual"], pdf["p50"]),
            "RMSE": np.sqrt(mean_squared_error(pdf["actual"], pdf["p50"])),
            "MAPE": calculate_mape(pdf["actual"].values, pdf["p50"].values),
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values),
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values)
        }
        return metrics, pdf

class NeuralForecastAdapter(ModelAdapter):
    def _prepare_df(self, csv_path):
        df_full = mp.load_and_validate_features(csv_path)
        nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"})
        nf_df["unique_id"] = "nordbyen"
        cfg = mp.default_feature_config()
        num_ex = [c for c in (cfg.past_covariates_cols + cfg.future_covariates_cols) if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
        return nf_df[["unique_id", "ds", "y"] + num_ex].dropna(), num_ex

    def train(self, csv_path: str, train_end_str: str, val_end_str: str):
        print(f"\n[{self.name}] Training...")
        nf_df, exog = self._prepare_df(csv_path)
        nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
        train_df = nf_df[nf_df['ds'] <= to_naive(train_end_str)].reset_index(drop=True)
        
        # Baseline core params
        model_params = {
            "h": 24,
            "input_size": 168,
            "futr_exog_list": exog,
            "loss": MQLoss(quantiles=[0.1, 0.5, 0.9]),
            "max_steps": self.config.get("n_epochs", 50)
        }
        
        # Override with HPO results if available
        if "best_params" in self.config:
            best = self.config["best_params"]
            model_params.update({
                "hidden_size": best.get("hidden_size", 64),
                "conv_hidden_size": best.get("conv_hidden_size", 64),
                "top_k": best.get("top_k", 2),
                "learning_rate": best.get("lr", 1e-3),
                "dropout": best.get("dropout", 0.1)
            })
            
        model = TimesNet(**model_params)
        nf = NeuralForecast(models=[model], freq='h')
        nf.fit(df=train_df)
        self.model = nf
        nf.save(path=os.path.join("models", self.name), overwrite=True)

    def evaluate(self, csv_path: str, test_start_str: str, n_predictions: int = 50) -> Tuple[Dict[str, float], pd.DataFrame]:
        print(f"[{self.name}] Evaluating (Walk-forward)...")
        nf_df, _ = self._prepare_df(csv_path)
        nf_df['ds'] = nf_df['ds'].dt.tz_localize(None)
        ts_naive = to_naive(test_start_str)
        
        all_rows = []
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
            cm, cl, ch = find_col(['median', '-q-0.5']), find_col(['-lo-80.0', '-q-0.1']), find_col(['-hi-80.0', '-q-0.9'])
            
            p50 = fcst[cm].values.tolist() if cm else fcst.iloc[:, -1].values.tolist()
            p10 = fcst[cl].values.tolist() if cl else p50 # Fallback
            p90 = fcst[ch].values.tolist() if ch else p50 # Fallback
            actuals = fut_df['y'].values.tolist()
            times = fut_df['ds'].tolist()
            
            for t, a, l, m, h in zip(times, actuals, p10, p50, p90):
                all_rows.append({"timestamp": t, "actual": a, "p10": l, "p50": m, "p90": h})
        
        pdf = pd.DataFrame(all_rows)
        metrics = {
            "MAE": mean_absolute_error(pdf["actual"], pdf["p50"]),
            "RMSE": np.sqrt(mean_squared_error(pdf["actual"], pdf["p50"])),
            "MAPE": calculate_mape(pdf["actual"].values, pdf["p50"].values),
            "PICP": calculate_picp(pdf["actual"].values, pdf["p10"].values, pdf["p90"].values),
            "MIW": calculate_miw(pdf["p10"].values, pdf["p90"].values)
        }
        return metrics, pdf

class Benchmarker:
    def __init__(self, csv_path: str, models_to_run: List[str]):
        self.csv_path, self.results = csv_path, []
        self.models_to_run = [m.upper() for m in models_to_run]
        
        # Load optimized params if they exist
        nhits_best = self._load_json("results/best_params_NHITS.json")
        timesnet_best = self._load_json("results/best_params_TIMESNET.json")
        
        self.configs = {
            "NHITS": {"type": "NHITS", "n_epochs": 15, "best_params": nhits_best},
            "TIMESNET": {"type": "TimesNet", "n_epochs": 1000, "best_params": timesnet_best}
        }

    def _load_json(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f: return json.load(f)
        return None

    def run(self):
        for mk in self.models_to_run:
            if mk not in self.configs: continue
            cfg = self.configs[mk]; adapter = NeuralForecastAdapter(mk, cfg) if mk == "TIMESNET" else DartsAdapter(mk, cfg)
            adapter.train(self.csv_path, "2018-12-31 23:00:00+00:00", "2019-12-31 23:00:00+00:00")
            metrics, pdf = adapter.evaluate(self.csv_path, "2020-01-01 00:00:00+00:00")
            metrics["Model"] = mk; self.results.append(metrics)
            
            os.makedirs("results", exist_ok=True)
            pdf.to_csv(f"results/{mk}_predictions.csv", index=False)
            print(f"Saved full predictions for {mk} to results/{mk}_predictions.csv")
            
        report_df = pd.DataFrame(self.results)
        print("\n" + "="*70 + "\nBENCHMARK RESULTS\n" + "="*70)
        print(report_df.to_string(index=False))
        os.makedirs("results", exist_ok=True); report_df.to_csv("results/benchmark_results.csv", index=False)

if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["NHITS", "TIMESNET"]
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nordbyen_features_engineered.csv")
    Benchmarker(csv_path, models).run()
