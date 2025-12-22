import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MAE
import torch

# Add parent directory to path to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp

def main():
    print("--- TimeSNet Stage 2: MAE Optimization (Deterministic Baseline) ---")
    
    # 1. Load Data
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, "nordbyen_processing", "nordbyen_features_engineered.csv")
    
    print(f"Loading data from: {csv_path}")
    df_full = mp.load_and_validate_features(csv_path)
    
    # 2. Split Data (Using same dates as Darts models)
    # Train: < 2019-01-01
    # Val: 2019-01-01 to 2019-12-31
    # Test: >= 2020-01-01
    # FINDING: Must use UTC localization to match df_full index
    train_end = pd.Timestamp("2018-12-31 23:00:00").tz_localize("UTC")
    val_end = pd.Timestamp("2019-12-31 23:00:00").tz_localize("UTC")
    
    # Let's convert the whole df to NeuralForecast format
    print("Converting to NeuralForecast Long Format...")
    nf_df = df_full.reset_index()
    nf_df = nf_df.rename(columns={
        "timestamp": "ds",
        "heat_consumption": "y"
    })
    nf_df["unique_id"] = "nordbyen"
    
    # Define Covariates
    cfg = mp.default_feature_config()
    
    # FINDING: NeuralForecast TimesNet does not support hist_exog_list, so we treat everything as 'future' (known covariates)
    # This implies we assume we have the weather forecast for the horizon.
    futr_exog_list = cfg.past_covariates_cols + cfg.future_covariates_cols
    
    # Ensure all columns are numeric/float32 for PyTorch
    all_cols = ["y"] + futr_exog_list
    for col in all_cols:
        nf_df[col] = nf_df[col].astype("float32")
        
    print(f"Combined Exogenous Variables ({len(futr_exog_list)}): {futr_exog_list}")
    
    # FINDING: Filter only relevant columns to remove any string/object columns (like 'public_holiday_name')
    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + futr_exog_list]
    
    # FINDING: Drop NaNs (crucial for NeuralForecast, especially with lag features)
    initial_len = len(nf_df)
    nf_df = nf_df.dropna()
    print(f"Dropped {initial_len - len(nf_df)} rows containing NaNs (likely initial lags). Remaining: {len(nf_df)}")

    # Split into Train and Test (Validation handling is implicit or manual)
    train_mask = nf_df['ds'] <= train_end
    train_df = nf_df[train_mask].reset_index(drop=True)
    
    print(f"Training Data: {len(train_df)} rows (End: {train_df['ds'].max()})")

    # 3. Initialize Model
    # Horizon = 24 hours
    # Input Size = 168 hours (7 days) - same as Darts TCN/NHiTS
    HORIZON = 24
    INPUT_SIZE = 168
    
    model = TimesNet(
        h=HORIZON,
        input_size=INPUT_SIZE,
        hist_exog_list=None, # Not supported
        futr_exog_list=futr_exog_list,
        loss=MAE(), # <--- CHANGED TO MAE
        scaler_type='standard', # Internal scaling
        # Optimization params
        learning_rate=1e-3,
        max_steps=20, # Very fast finish
    )
    
    # 4. Train
    print("\n--- Training Model ---")
    nf = NeuralForecast(
        models=[model],
        freq='H'
    )
    
    nf.fit(df=train_df)
    print("Training Complete.")
    
    # 5. Predict / Sanity Check
    print("\n--- Generating Forecast ---")
    last_train_date = train_df['ds'].max()
    future_dates = pd.date_range(last_train_date + pd.Timedelta(hours=1), periods=HORIZON, freq='H')
    futr_df = nf_df[nf_df['ds'].isin(future_dates)].reset_index(drop=True)
    
    forecasts = nf.predict(futr_df=futr_df)
    print("\nForecasts Head:")
    print(forecasts.head())
    
    # 6. Basic Plotting & Metric Calc
    actuals = futr_df['y'].values
    preds = forecasts['TimesNet'].values
    
    mae_val = np.mean(np.abs(actuals - preds))
    print(f"\nSample MAE: {mae_val:.4f}")
    
    # Save Model (Optional, or just save/plot)
    # Save Model (Standardized Location)
    model_save_dir = os.path.join(data_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, "timesnet_deterministic_mae")
    nf.save(path=save_path, model_index=None, overwrite=True)
    print(f"Model saved to {save_path}")

    # Save plot (Standardized Location)
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, actuals, label='Actual', marker='o')
    plt.plot(future_dates, preds, label='Prediction (MAE)', marker='x')
    plt.title(f"TimesNet Stage 2 (MAE)\nSample MAE: {mae_val:.4f}")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_dir, "stage2_mae_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
