import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MQLoss
import torch

# Add parent directory to path to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp

def main():
    print("--- TimeSNet Stage 3: Probabilistic / Quantile Regression ---")
    
    # 1. Load Data
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, "nordbyen_processing", "nordbyen_features_engineered.csv")
    
    print(f"Loading data from: {csv_path}")
    df_full = mp.load_and_validate_features(csv_path)
    
    train_end = pd.Timestamp("2018-12-31 23:00:00").tz_localize("UTC")
    
    print("Converting to NeuralForecast Long Format...")
    nf_df = df_full.reset_index()
    nf_df = nf_df.rename(columns={
        "timestamp": "ds",
        "heat_consumption": "y"
    })
    nf_df["unique_id"] = "nordbyen"
    
    cfg = mp.default_feature_config()
    futr_exog_list = cfg.past_covariates_cols + cfg.future_covariates_cols
    
    all_cols = ["y"] + futr_exog_list
    for col in all_cols:
        nf_df[col] = nf_df[col].astype("float32")
        
    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + futr_exog_list]
    
    initial_len = len(nf_df)
    nf_df = nf_df.dropna()
    print(f"Dropped {initial_len - len(nf_df)} rows containing NaNs. Remaining: {len(nf_df)}")

    train_mask = nf_df['ds'] <= train_end
    train_df = nf_df[train_mask].reset_index(drop=True)
    
    print(f"Training Data: {len(train_df)} rows")

    # 3. Initialize Model
    HORIZON = 24
    INPUT_SIZE = 168
    QUANTILES = [0.1, 0.5, 0.9]
    
    model = TimesNet(
        h=HORIZON,
        input_size=INPUT_SIZE,
        hist_exog_list=None,
        futr_exog_list=futr_exog_list,
        loss=MQLoss(quantiles=QUANTILES), # <--- PROBABILISTIC LOSS
        scaler_type='standard',
        learning_rate=1e-4, 
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
    
    # 5. Predict
    print("\n--- Generating Forecast ---")
    last_train_date = train_df['ds'].max()
    future_dates = pd.date_range(last_train_date + pd.Timedelta(hours=1), periods=HORIZON, freq='H')
    futr_df = nf_df[nf_df['ds'].isin(future_dates)].reset_index(drop=True)
    
    forecasts = nf.predict(futr_df=futr_df)
    print("\nForecasts Head:")
    print(forecasts.head())
    
    # Check Columns: Should be TimesNet-lo-90, TimesNet-median, TimesNet-hi-90 ?
    # MQLoss output columns are usually: ModelName-q-0.1, ModelName-q-0.5, ModelName-q-0.9
    # Or just 'TimesNet' (mean/median) and others?
    # Let's inspect print output during run.
    
    # 6. Plotting
    actuals = futr_df['y'].values
    # Attempt to identify quantile columns
    cols = forecasts.columns
    # Usually: TimesNet, TimesNet-lo-90... 
    # With MQLoss, it uses the quantile names.
    # We will look for cols containing 'TimesNet'
    
    # Assumption for MQLoss in NeuralForecast:
    # 0.5 is usually the main prediction if in quantiles?
    
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, actuals, label='Actual', marker='o', color='black')
    
    # Plot if we can find them
    # NeuralForecast naming convention varies by version/loss.
    # Often it is: TimesNet-q-0.1, TimesNet-q-0.5, TimesNet-q-0.9
    
    q_cols = [c for c in cols if "TimesNet" in c]
    print(f"Prediction Columns: {q_cols}")
    
    # Try to plot
    for c in q_cols:
        plt.plot(future_dates, forecasts[c].values, label=c, alpha=0.7)
        
    plt.title(f"TimesNet Stage 3 (Probabilistic)")
    plt.legend()
    plt.grid(True)
    # Save Plot (Standardized Location)
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "stage3_prob_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Save Model (Standardized Location)
    model_save_dir = os.path.join(data_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, "timesnet_probabilistic_q")
    nf.save(path=save_path, model_index=None, overwrite=True)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
