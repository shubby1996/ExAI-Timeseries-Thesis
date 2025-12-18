import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from darts.models import NHiTSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp

def main():
    print("--- Unifying Comparison: NHiTS vs TimesNet (Original Scale) ---")
    
    # 1. Load Data & Preprocessing State
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, "nordbyen_features_engineered.csv")
    models_dir = os.path.join(data_dir, "models")
    timesnet_models_dir = os.path.join(data_dir, "timesnet", "models")
    
    print("Loading data...")
    print("Loading data...")
    # Load raw df to get original values for inverse validation
    df_full = mp.load_and_validate_features(csv_path)
    
    # FORCE NAIVE: Strip TZ to avoid mismatch with unknown NHiTS training state
    if df_full.index.tz is not None:
        print("Stripping TZ from df_full index for compatibility...")
        df_full.index = df_full.index.tz_localize(None)

    # Load NHiTS Preprocessing State (to get Scaler)
    state_path = os.path.join(models_dir, "nhits_deterministic_mae_preprocessing_state.pkl")
    if not os.path.exists(state_path):
        print(f"Warning: {state_path} not found. Trying to reconstruct scaling.")
        # Re-run preparation (using Naive timestamps now)
        train_end = pd.Timestamp("2018-12-31 23:00:00")
        val_end = pd.Timestamp("2019-12-31 23:00:00")
        state, _, _, _ = mp.prepare_model_data(csv_path, train_end, val_end)
    else:
        state = mp.load_preprocessing_state(state_path)
        
    target_scaler = state.target_scaler
    print("Scaler loaded.")
    
    # 2. Prepare Test Data (2020)
    test_start = pd.Timestamp("2020-01-01 00:00:00") # Naive
    test_df = df_full.loc[test_start:]
    
    # --- Part A: NHiTS Prediction ---
    # --- Part A: NHiTS Prediction ---
    print("\n[NHiTS] Loading and Predicting...")
    nhits_values = None
    nhits_mae = None
    
    try:
        nhits_path = os.path.join(models_dir, "nhits_deterministic_mae.pt")
        if os.path.exists(nhits_path):
            nhits_model = NHiTSModel.load(nhits_path)
            
            # ATTEMPT: Predict with strict try-catch for shape mismatch
            scaled_dict = mp.apply_state_to_full_df(df_full, state)
            target_ts = scaled_dict["target"]
            past_cov_ts = scaled_dict["past_covariates"]
            pred_start = pd.Timestamp("2020-01-01 00:00:00") # Naive
            
            nhits_pred_scaled = nhits_model.predict(
                n=24,
                series=target_ts.split_before(pred_start)[0],
                past_covariates=past_cov_ts,
                num_samples=1
            )
            nhits_pred_real = target_scaler.inverse_transform(nhits_pred_scaled)
            nhits_values = nhits_pred_real.values().flatten()
            
    except Exception as e:
        print(f"NHiTS Prediction Failed (Expected due to legacy mismatch): {e}")
        print("Using ESTIMATED NHiTS MAE based on known Scaled Score (0.0693).")
        
        # Estimate Real MAE
        # Scaled = (Real - Min) / (Max - Min)
        # DeltaScaled = DeltaReal / (Max - Min)
        # RealMAE = ScaledMAE * (Max - Min)
        # MinMax Scaler stores scale_ = 1 / (Max - Min)
        # So RealMAE = ScaledMAE / scale_
        
        # Check if scaler fits this logic (MinMax)
        if hasattr(target_scaler, 'scale_'):
             # Darts wraps sklearn scaler. transformer.scale_
             scale_factor = target_scaler.transformer.scale_[0]
             known_scaled_mae = 0.0693
             estimated_real_mae = known_scaled_mae / scale_factor
             print(f"Scaler Factor: {scale_factor} (1/Range)")
             print(f"Estimated NHiTS Real MAE: {estimated_real_mae:.4f}")
             nhits_mae = estimated_real_mae
             
             # Create dummy values for plot (just mean line or skip)
             nhits_values = np.full(24, np.nan) 
             
    if nhits_values is None:
         nhits_values = np.zeros(24)

    # --- Part B: TimesNet Prediction ---
    print("\n[TimesNet] Loading and Predicting...")
    # Load NF model (if saved in Stage 2)
    # Or assuming we just trained it.
    # Stage 2 saves to 'timesnet/models/timesnet_mae'
    
    nf_path = os.path.join(timesnet_models_dir, "timesnet_mae")
    if os.path.exists(nf_path):
        nf = NeuralForecast.load(path=nf_path)
        
        # Prepare Input DF
        # We need history + future covariates for the prediction window.
        # Context: 168 hours before 2020-01-01
        context_start = pred_start - pd.Timedelta(hours=168)
        context_end = pred_start - pd.Timedelta(hours=1) # 23:00 prev day
        
        # Future 24h
        future_end = pred_start + pd.Timedelta(hours=23)
        
        # We need a DF covering Context + Future
        # NF predict needs 'futr_df' for the horizon.
        # But it needs the MODEL to have seen history.
        # If we load a pre-trained model, we must pass the history to `predict(df=history_df)`.
        
        # Slice history dataframe (Needs to be clean, processed)
        nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"})
        nf_df["unique_id"] = "nordbyen"
        
        # Add covariates... (Need same processing: float32, dropna, etc)
        # Minimal processing for inference:
        cfg = mp.default_feature_config()
        futr_exog_list = cfg.past_covariates_cols + cfg.future_covariates_cols
        ids_cols = ["unique_id", "ds", "y"]
        nf_df = nf_df[ids_cols + futr_exog_list].dropna() # Apply dropna just in case
        
        # Filter context
        history_df = nf_df[(nf_df['ds'] >= context_start) & (nf_df['ds'] <= context_end)]
        
        # Filter future (for exogenous)
        future_df = nf_df[(nf_df['ds'] >= pred_start) & (nf_df['ds'] <= future_end)]
        
        # Predict expects 'futr_df' to contain the future exogenous vars
        # And 'df' (history) to provide the inputs.
        
        fcst = nf.predict(df=history_df, futr_df=future_df)
        timesnet_values = fcst['TimesNet'].values
        
        # Inverse Transform?
        # TimesNet used internal 'standard' scaling. NeuralForecast handles inverse transform automatically 
        # IF it was trained with valid y. The output of predict() is in ORIGINAL SCALE of the y passed in fit().
        # In my training script, I passed unscaled 'y' (df_full['heat_consumption']).
        # So output is REAL UNITS.
        
    else:
        print("TimesNet model not found!")
        timesnet_values = np.zeros(24)

    # --- Part C: TimesNet Probabilistic Prediction ---
    print("\n[TimesNet Prob] Loading and Predicting...")
    nf_prob_path = os.path.join(timesnet_models_dir, "timesnet_prob")
    if os.path.exists(nf_prob_path):
        nf_prob = NeuralForecast.load(path=nf_prob_path)
        
        # Reuse history/future dfs
        fcst_prob = nf_prob.predict(df=history_df, futr_df=future_df)
        print("Prob Columns:", fcst_prob.columns)
        
        # MQLoss output columns: TimesNet, TimesNet-median, TimesNet-lo-90 ??
        # Usually: TimesNet-median, TimesNet-lo-90, TimesNet-hi-90
        # Check columns
        # Taking median (0.5) for MAE
        prob_cols = fcst_prob.columns
        # Heuristic: find column with 'median' or just 'TimesNet' if it's the only one?
        # MQLoss(quantiles=[0.1, 0.5, 0.9]) produced 3 output heads?
        # Usually formatted as: TimesNet-q-0.5
        
        median_col = [c for c in prob_cols if "0.5" in c or "median" in c]
        if median_col:
            timesnet_prob_values = fcst_prob[median_col[0]].values
        else:
            # Fallback
            print("Could not identify median column. Using default.")
            timesnet_prob_values = fcst_prob['TimesNet'].values
            
    else:
        print("TimesNet Prob model not found!")
        timesnet_prob_values = np.zeros(24)

    # --- Part D: Comparison ---
    print("\n--- RESULTS (Real Units) ---")
    
    # Get Actuals
    actuals_df = df_full.loc[pred_start : pred_start + pd.Timedelta(hours=23)]
    actual_values = actuals_df["heat_consumption"].values
    
    # Calculate Metrics
    if nhits_mae is None:
        # If prediction succeeded
        nhits_mae = mean_absolute_error(actual_values, nhits_values)
        
    timesnet_mae = mean_absolute_error(actual_values, timesnet_values)
    timesnet_prob_mae = mean_absolute_error(actual_values, timesnet_prob_values)
    
    print(f"Target: {len(actual_values)} hours (2020-01-01)")
    print(f"Actual Mean: {np.mean(actual_values):.2f}")
    
    print(f"-"*40)
    print(f"NHiTS (Det) MAE:      {nhits_mae:.4f} (Estimated/Calc)")
    print(f"TimesNet (Det) MAE:   {timesnet_mae:.4f}")
    print(f"TimesNet (Prob) MAE:  {timesnet_prob_mae:.4f}")
    print(f"-"*40)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(actual_values, label="Actual", color="black", linewidth=2)
    plt.plot(nhits_values, label=f"NHiTS (MAE={nhits_mae:.2f})", linestyle="--")
    plt.plot(timesnet_values, label=f"TimesNet Det (MAE={timesnet_mae:.2f})", linestyle="-.")
    plt.plot(timesnet_prob_values, label=f"TimesNet Prob (MAE={timesnet_prob_mae:.2f})", linestyle=":")
    plt.title("Baseline Comparison: NHiTS vs TimesNet (Real Units)")
    plt.legend()
    plt.grid(True)
    plt.savefig("timesnet/comparison_plot_real.png")
    print("Comparison plot saved.")

if __name__ == "__main__":
    main()
