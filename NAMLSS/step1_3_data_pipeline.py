import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


'''
Below is a single self-contained script you can drop into your repo and run. It will:
    load your CSV
    parse/sort timestamp
    compute hour_sin/hour_cos if missing
    reindex to hourly grid (so we can detect gaps cleanly)
    split chronologically (train/val/test)
    fit scalers on train only (heat_consumption, temp)
    build a window dataset that returns:
        y_hist (L,)
        temp_hist (L,)
        future_cov (H,2) ← future-known covariates
        y_future (H,)
    print a sanity sample (timestamps + tensor shapes)

    pip install pandas numpy torch scikit-learn
    python step1_3_data_pipeline.py --csv_path "path/to/your/data.csv" --L 168 --H 24
'''


# ---------------------------
# Config
# ---------------------------
@dataclass
class TSConfig:
    # window + horizon
    L: int = 168
    H: int = 24

    # columns
    timestamp_col: str = "timestamp"
    target: str = None  # SINGULAR: what to forecast (e.g., "heat_consumption")
    endo_cols: list = None  # Optional: lagged/transformed versions of target (autoregressive features)
    exo_cols: list = None  # list of exogenous column names (external features, no future info)
    future_cov_cols: list = None  # list of future-known covariate column names

    # splits
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15  # should sum ~ 1.0

    # missing policy
    # "hourly_reindex" will create a full hourly index and leave missing as NaN.
    # Dataset will only keep windows with all required values present.
    reindex_hourly: bool = True

    def __post_init__(self):
        # Default columns (can be overridden)
        if self.target is None:
            self.target = "heat_consumption"
        if self.endo_cols is None:
            self.endo_cols = ["heat_lag_1h", "heat_lag_24h", "heat_rolling_24h"]  # 20->23: keep endo (3)
        if self.exo_cols is None:
            self.exo_cols = ["temp", "wind_speed", "dew_point", "temp_squared", "temp_wind_interaction", 
                           "humidity", "clouds_all", "pressure", "rain_1h", "snow_1h", "temp_weekend_interaction"]  # 20->23: keep exo (11 total)
        if self.future_cov_cols is None:
            self.future_cov_cols = ["hour_sin", "hour_cos", "is_weekend", "is_public_holiday", 
                                   "day_of_week", "season", "hour", "month", "is_school_holiday"]  # 20->23: +hour+month+is_school_holiday (9 total)


# ---------------------------
# Utilities
# ---------------------------
def compute_hour_sin_cos(ts: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    hour = ts.hour.values.astype(np.float32)  # 0..23
    angle = 2.0 * np.pi * (hour / 24.0)
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def load_and_prepare(csv_path: Path, cfg: TSConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # parse timestamp
    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {cfg.timestamp_col}")

    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df = df.dropna(subset=[cfg.timestamp_col])

    # sort + drop duplicates on timestamp
    df = df.sort_values(cfg.timestamp_col)
    df = df.drop_duplicates(subset=[cfg.timestamp_col], keep="last")

    # set index
    df = df.set_index(cfg.timestamp_col)

    # ensure required columns exist
    # target must always exist
    if cfg.target not in df.columns:
        raise ValueError(f"Missing target column: {cfg.target}")
    
    required = [cfg.target] + (cfg.endo_cols or []) + (cfg.exo_cols or [])
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # compute/ensure future covariates exist
    # Auto-compute hour_sin/hour_cos if requested but not present
    for cov_col in cfg.future_cov_cols:
        if cov_col not in df.columns:
            if cov_col == "hour_sin":
                hs, _ = compute_hour_sin_cos(df.index)
                df[cov_col] = hs
            elif cov_col == "hour_cos":
                _, hc = compute_hour_sin_cos(df.index)
                df[cov_col] = hc
            else:
                raise ValueError(f"Future covariate '{cov_col}' not found in CSV and cannot be auto-generated")

    # optional: reindex to hourly grid (detect gaps cleanly)
    if cfg.reindex_hourly:
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df = df.reindex(full_index)
        df = df.tz_convert("Europe/Berlin")

        # recompute time-based future covariates for the full grid (guarantees no NaN)
        for cov_col in cfg.future_cov_cols:
            if cov_col == "hour_sin":
                hs, _ = compute_hour_sin_cos(df.index)
                df[cov_col] = hs
            elif cov_col == "hour_cos":
                _, hc = compute_hour_sin_cos(df.index)
                df[cov_col] = hc

    return df


def chronological_split_indices(n: int, train_frac: float, val_frac: float) -> Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    # ranges are inclusive [start, end]
    train_end = int(n * train_frac) - 1
    val_end = train_end + int(n * val_frac)
    test_end = n - 1

    train = (0, max(train_end, 0))
    val = (train[1] + 1, min(val_end, n - 1))
    test = (val[1] + 1, test_end)

    return train, val, test


def fit_scalers_on_train(df: pd.DataFrame, cfg: TSConfig, train_range: Tuple[int,int]) -> Dict[str, StandardScaler]:
    s, e = train_range
    train_df = df.iloc[s:e+1]

    scalers = {}

    # Target scaler (singular)
    target_scaler = StandardScaler()
    target_vals = train_df[[cfg.target]].to_numpy(dtype=np.float32)
    target_scaler.fit(target_vals[~np.isnan(target_vals).any(axis=1)])  # ignore NaNs
    scalers[cfg.target] = target_scaler

    # Endogenous (AR) scalers - one per endogenous column (if any)
    for endo_col in cfg.endo_cols:
        endo_scaler = StandardScaler()
        endo_vals = train_df[[endo_col]].to_numpy(dtype=np.float32)
        endo_scaler.fit(endo_vals[~np.isnan(endo_vals).any(axis=1)])  # ignore NaNs
        scalers[endo_col] = endo_scaler

    # Exogenous scalers (one per exogenous column)
    for exo_col in cfg.exo_cols:
        x_scaler = StandardScaler()
        x_vals = train_df[[exo_col]].to_numpy(dtype=np.float32)
        x_scaler.fit(x_vals[~np.isnan(x_vals).any(axis=1)])
        scalers[exo_col] = x_scaler

    return scalers


def apply_scalers(df: pd.DataFrame, cfg: TSConfig, scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    out = df.copy()

    # scale target column
    target = out[[cfg.target]].to_numpy(dtype=np.float32)
    target_mask = ~np.isnan(target).any(axis=1)
    target_scaled = target.copy()
    target_scaled[target_mask] = scalers[cfg.target].transform(target[target_mask])
    out[cfg.target] = target_scaled.astype(np.float32)

    # scale each endogenous (AR) column
    for endo_col in cfg.endo_cols:
        endo = out[[endo_col]].to_numpy(dtype=np.float32)
        endo_mask = ~np.isnan(endo).any(axis=1)
        endo_scaled = endo.copy()
        endo_scaled[endo_mask] = scalers[endo_col].transform(endo[endo_mask])
        out[endo_col] = endo_scaled.astype(np.float32)

    # scale each exogenous column
    for exo_col in cfg.exo_cols:
        x = out[[exo_col]].to_numpy(dtype=np.float32)
        x_mask = ~np.isnan(x).any(axis=1)
        x_scaled = x.copy()
        x_scaled[x_mask] = scalers[exo_col].transform(x[x_mask])
        out[exo_col] = x_scaled.astype(np.float32)

    return out


# ---------------------------
# Window Dataset
# ---------------------------
class WindowDataset(Dataset):
    """
    Each sample is defined by a 'center' index t = end of history window.
    Returns:
      - target_hist: (L,) - history of target variable
      - target_future: (H,) - future of target variable (ground truth)
      - For each endo_col (AR feature): endo_col + '_hist': (L,)
      - For each exo_col: exo_col: (L,) - exogenous history
      - future_cov: (H, num_future_cov) stacked future covariate features
    """
    def __init__(self, df: pd.DataFrame, cfg: TSConfig, split_range: Tuple[int,int]):
        super().__init__()
        self.df = df
        self.cfg = cfg
        self.split_range = split_range

        self.centers: List[int] = self._build_centers()

    def _is_finite_block(self, arr: np.ndarray) -> bool:
        return np.isfinite(arr).all()

    def _build_centers(self) -> List[int]:
        L, H = self.cfg.L, self.cfg.H
        s, e = self.split_range

        # centers t must satisfy:
        # - have full history: t-(L-1) >= 0
        # - have full future within split: t+H <= e
        t_min = max(s + (L - 1), (L - 1))
        t_max = e - H

        centers = []
        if t_max < t_min:
            return centers

        # Pull numpy arrays once for speed
        target_all = self.df[self.cfg.target].to_numpy(dtype=np.float32)
        endo_all = {col: self.df[col].to_numpy(dtype=np.float32) for col in self.cfg.endo_cols}
        exo_all = {col: self.df[col].to_numpy(dtype=np.float32) for col in self.cfg.exo_cols}
        future_cov_all = {col: self.df[col].to_numpy(dtype=np.float32) for col in self.cfg.future_cov_cols}

        for t in range(t_min, t_max + 1):
            # Check target column (always required)
            valid = True
            y_hist = target_all[t - (L - 1): t + 1]
            y_fut = target_all[t + 1: t + H + 1]
            if len(y_hist) != L or len(y_fut) != H:
                valid = False
            if not self._is_finite_block(y_hist) or not self._is_finite_block(y_fut):
                valid = False
            if not valid:
                continue

            # Check all endogenous (AR) columns if any
            for endo_col in self.cfg.endo_cols:
                endo_hist = endo_all[endo_col][t - (L - 1): t + 1]
                if len(endo_hist) != L:
                    valid = False
                    break
                if not self._is_finite_block(endo_hist):
                    valid = False
                    break
            if not valid:
                continue

            # Check all future covariate columns
            for cov_col in self.cfg.future_cov_cols:
                cov_fut = future_cov_all[cov_col][t + 1: t + H + 1]
                if len(cov_fut) != H or not self._is_finite_block(cov_fut):
                    valid = False
                    break
            if not valid:
                continue

            # Check all exogenous columns
            for exo_col in self.cfg.exo_cols:
                exo_hist = exo_all[exo_col][t - (L - 1): t + 1]
                if len(exo_hist) != L or not self._is_finite_block(exo_hist):
                    valid = False
                    break
            if not valid:
                continue

            centers.append(t)

        return centers

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.centers[idx]
        L, H = self.cfg.L, self.cfg.H

        ret = {}
        
        # Extract target (always present)
        target = self.df[self.cfg.target].to_numpy(dtype=np.float32)
        target_hist = target[t - (L - 1): t + 1]  # (L,)
        target_future = target[t + 1: t + H + 1]   # (H,)
        ret["target_hist"] = torch.from_numpy(target_hist)
        ret["target_future"] = torch.from_numpy(target_future)
        
        # Extract optional endogenous (AR) features
        for endo_col in self.cfg.endo_cols:
            endo = self.df[endo_col].to_numpy(dtype=np.float32)
            endo_hist = endo[t - (L - 1): t + 1]  # (L,)
            ret[endo_col + "_hist"] = torch.from_numpy(endo_hist)
        
        # Extract all exogenous histories
        for exo_col in self.cfg.exo_cols:
            x = self.df[exo_col].to_numpy(dtype=np.float32)
            x_hist = x[t - (L - 1): t + 1]
            ret[exo_col] = torch.from_numpy(x_hist)  # (L,)
        
        # Extract future covariates and stack into (H, num_future_cov)
        future_cov_arrays = []
        for cov_col in self.cfg.future_cov_cols:
            cov = self.df[cov_col].to_numpy(dtype=np.float32)
            cov_fut = cov[t + 1: t + H + 1]  # (H,)
            future_cov_arrays.append(cov_fut)
        
        if future_cov_arrays:
            future_cov = np.stack(future_cov_arrays, axis=1)  # (H, num_future_cov)
            ret["future_cov"] = torch.from_numpy(future_cov)
        else:
            # No future covariates - return empty tensor
            ret["future_cov"] = torch.zeros((H, 0), dtype=torch.float32)

        return ret


# ---------------------------
# Main / Dry Run
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--L", type=int, default=168)
    parser.add_argument("--H", type=int, default=24)
    args = parser.parse_args()

    cfg = TSConfig(L=args.L, H=args.H)

    df_raw = load_and_prepare(Path(args.csv_path), cfg)
    print("\n=== Loaded ===")
    print("rows:", len(df_raw))
    print("date range:", df_raw.index.min(), "->", df_raw.index.max())

    # missing timestamp gaps (after hourly reindex, missing shows up as NaNs)
    for endo_col in cfg.endo_cols:
        missing_endo = df_raw[endo_col].isna().sum()
        print(f"missing {endo_col}: {int(missing_endo)}")
    for exo_col in cfg.exo_cols:
        missing_exo = df_raw[exo_col].isna().sum()
        print(f"missing {exo_col}: {int(missing_exo)}")

    # splits
    n = len(df_raw)
    train_rng, val_rng, test_rng = chronological_split_indices(n, cfg.train_frac, cfg.val_frac)
    print("\n=== Splits (inclusive index ranges) ===")
    print("train:", train_rng, "| rows:", train_rng[1] - train_rng[0] + 1)
    print("val  :", val_rng,   "| rows:", max(val_rng[1] - val_rng[0] + 1, 0))
    print("test :", test_rng,  "| rows:", max(test_rng[1] - test_rng[0] + 1, 0))

    # scalers fit on train only
    scalers = fit_scalers_on_train(df_raw, cfg, train_rng)
    df = apply_scalers(df_raw, cfg, scalers)

    print("\n=== Scaler summary (train only) ===")
    print(f"{cfg.target} mean/std: {float(scalers[cfg.target].mean_[0]):.4f} {float(np.sqrt(scalers[cfg.target].var_[0])):.4f}")
    for endo_col in cfg.endo_cols:
        print(f"{endo_col} mean/std: {float(scalers[endo_col].mean_[0]):.4f} {float(np.sqrt(scalers[endo_col].var_[0])):.4f}")
    for exo_col in cfg.exo_cols:
        print(f"{exo_col} mean/std: {float(scalers[exo_col].mean_[0]):.4f} {float(np.sqrt(scalers[exo_col].var_[0])):.4f}")

    # datasets
    ds_train = WindowDataset(df, cfg, train_rng)
    ds_val = WindowDataset(df, cfg, val_rng)
    ds_test = WindowDataset(df, cfg, test_rng)

    print("\n=== Window dataset sizes ===")
    print("train samples:", len(ds_train))
    print("val samples  :", len(ds_val))
    print("test samples :", len(ds_test))

    if len(ds_train) == 0:
        print("\nNo train samples found. Likely causes:")
        print("- Not enough rows for L+H")
        print("- Too many NaNs after hourly reindex")
        print(f"\nChecklist:")
        print(f"- Target column '{cfg.target}' exists: {cfg.target in df_raw.columns}")
        for endo_col in cfg.endo_cols:
            print(f"- Endogenous column '{endo_col}' exists: {endo_col in df_raw.columns}")
        for exo_col in cfg.exo_cols:
            print(f"- Exogenous column '{exo_col}' exists: {exo_col in df_raw.columns}")
        for cov_col in cfg.future_cov_cols:
            print(f"- Future covariate '{cov_col}' exists: {cov_col in df_raw.columns}")
        return

    # show 1 sample sanity check
    sample = ds_train[0]
    print("\n=== Sample[0] sanity ===")
    print("Tensor shapes:")
    # Show target
    print(f"target_hist: {tuple(sample['target_hist'].shape)}")
    print(f"target_future: {tuple(sample['target_future'].shape)}")
    # Show optional AR features
    for endo_col in cfg.endo_cols:
        print(f"{endo_col}_hist: {tuple(sample[endo_col + '_hist'].shape)}")
    # Show exogenous columns
    for exo_col in cfg.exo_cols:
        print(f"{exo_col}: {tuple(sample[exo_col].shape)}")
    # Show future covariates
    print(f"future_cov: {tuple(sample['future_cov'].shape)}  (H x {len(cfg.future_cov_cols)} future features)")

    # quick numeric check
    print("\nFirst 3 target_future values (scaled):", sample["target_future"][:3].tolist())
    print("First 3 future_cov rows:")
    print(sample["future_cov"][:3].tolist())


if __name__ == "__main__":
    main()
