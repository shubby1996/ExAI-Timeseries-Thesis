"""
TimesNet Prediction Script for Nordbyen Heat Consumption Forecasting.

This script:
- Loads a trained NeuralForecast TimesNet model (deterministic or probabilistic)
- Prepares history + future exogenous dataframes
- Generates a 24-hour forecast and saves CSV to `data/`
"""

import os
import sys
import pandas as pd
from typing import Optional

# NeuralForecast
from neuralforecast import NeuralForecast

# Add parent directory to path to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp
from model_preprocessing import append_future_calendar_and_holidays


def predict_timesnet(
    csv_path: str,
    model_dir: str,
    n: int = 24,
    model_name: str = "timesnet_probabilistic_q",
    save_path: Optional[str] = None,
 ) -> pd.DataFrame:
    """Generate `n`-hour forecast using a saved NeuralForecast model directory.

    Returns a DataFrame containing timestamp and prediction columns.
    """
    print("=" * 70)
    print("TimesNet PREDICTION - NORDBYEN HEAT CONSUMPTION")
    print("=" * 70)

    # 1. Load engineered CSV
    print("[1/4] Loading engineered CSV...")
    df_full = mp.load_and_validate_features(csv_path)
    last_ts = df_full.index.max()
    print(f"  ✓ Loaded {len(df_full)} rows. Last timestamp: {last_ts}")

    # 2. Append future calendar/holiday rows and convert to NeuralForecast long format
    print("[2/4] Preparing NeuralForecast input frames (including future calendar features)...")

    # Extend the dataframe with n future rows so we have future covariates for forecasting
    df_extended = append_future_calendar_and_holidays(df_full, n_future=n, freq="H",
                                                     school_holidays_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "school_holidays.csv"))

    idx_name = df_extended.index.name if df_extended.index.name is not None else "index"
    nf_df = df_extended.reset_index().rename(columns={idx_name: "ds", "heat_consumption": "y"})
    nf_df["unique_id"] = "nordbyen"

    cfg = mp.default_feature_config()
    # For future dataframe we only need future covariates (calendar/holidays)
    # Include both past and future covariates as columns so we can supply
    # filled values for past covariates in the future horizon (e.g. persistence).
    all_exog_cols = cfg.past_covariates_cols + cfg.future_covariates_cols

    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + all_exog_cols]

    # Ensure numeric types
    for col in ["y"] + all_exog_cols:
        nf_df[col] = nf_df[col].astype("float32")

    # 3. Build history and future frames for next `n` hours
    future_dates = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=n, freq="H")
    futr_df = nf_df[nf_df["ds"].isin(future_dates)].reset_index(drop=True)

    # Fill past covariates in the future horizon with the last known value (persistence)
    # This provides a simple placeholder if no weather forecast is available.
    for col in cfg.past_covariates_cols:
        if col in futr_df.columns:
            # last known value from history
            last_val = nf_df[nf_df['ds'] <= last_ts][col].iloc[-1]
            futr_df[col] = futr_df[col].fillna(last_val)

    # History: provide a sufficient context (use last 168 hours)
    history_start = last_ts - pd.Timedelta(hours=168) + pd.Timedelta(hours=1)
    history_df = nf_df[(nf_df["ds"] >= history_start) & (nf_df["ds"] <= last_ts)].reset_index(drop=True)

    print(f"  ✓ History rows: {len(history_df)}, Future rows: {len(futr_df)}")

    # 4. Load model and predict
    print("[3/4] Loading TimesNet model and predicting...")
    # Resolve model path: prefer explicit path, otherwise try to auto-detect
    nf_path = os.path.join(model_dir, model_name)
    if not os.path.exists(nf_path):
        # Try to auto-detect any TimesNet model directory under the provided model_dir
        searched_dirs = []
        def search_dir(d):
            if not os.path.exists(d):
                return None
            searched_dirs.append(d)
            cand = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
            if not cand:
                return None
            preferred = [
                "timesnet_probabilistic_q",
                "timesnet_probabilistic",
                "timesnet_probabilistic_q_smoke",
                "timesnet_mae",
                "timesnet_mse",
            ]
            for p in preferred:
                if p in cand:
                    return os.path.join(d, p)
            timesnet_dirs = [x for x in cand if "timesnet" in x.lower()]
            if timesnet_dirs:
                timesnet_dirs.sort(key=lambda dd: os.path.getmtime(os.path.join(d, dd)), reverse=True)
                return os.path.join(d, timesnet_dirs[0])
            return None

        chosen_path = search_dir(model_dir)
        if chosen_path is None:
            # also check top-level models folder next to data/ (e.g., data/models)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            alt_models = os.path.join(project_root, "models")
            chosen_path = search_dir(alt_models)

        if chosen_path:
            nf_path = chosen_path
            print(f"[INFO] Auto-detected TimesNet model directory: {nf_path}")
        else:
            raise FileNotFoundError(f"TimesNet model directory not found: {nf_path} (searched: {searched_dirs})")

    nf = NeuralForecast.load(path=nf_path)
    fcst = nf.predict(df=history_df, futr_df=futr_df)

    # 5. Extract sensible prediction column (median or first available)
    candidates = [c for c in fcst.columns if "median" in c or "0.5" in c]
    if candidates:
        pred_col = candidates[0]
    elif "TimesNet" in fcst.columns:
        pred_col = "TimesNet"
    else:
        pred_col = fcst.columns[0]

    preds = fcst[pred_col].values

    result_df = pd.DataFrame({
        "timestamp": future_dates,
        "predicted": preds,
    })

    # Optionally save
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "timesnet_predictions_future_24h.csv")
    result_df.to_csv(save_path, index=False)
    print(f"[4/4] Predictions saved to: {save_path}")

    return result_df


if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_DIR = os.path.join(DATA_DIR, "timesnet", "models")

    predict_timesnet(
        csv_path=CSV_PATH,
        model_dir=MODEL_DIR,
        n=24,
        model_name="timesnet_mae",
    )
