import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import MSE
import torch

# Add parent directory to path to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp

def main():
    print("--- TimeSNet Stage 1: MSE Optimization (Sanity Check) ---")
    
    # 1. Load Data
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(data_dir, "nordbyen_processing", "nordbyen_features_engineered.csv")
    
    print(f"Loading data from: {csv_path}")
    df_full = mp.load_and_validate_features(csv_path)
    
    # 2. Split Data (Using same dates as Darts models)
    # Train: < 2019-01-01
    # Val: 2019-01-01 to 2019-12-31
    # Test: >= 2020-01-01
    train_end = pd.Timestamp("2018-12-31 23:00:00").tz_localize("UTC")
    val_end = pd.Timestamp("2019-12-31 23:00:00").tz_localize("UTC")
    
    # We need to prepare the FULL dataframe in Long Format first, then split?
    # NeuralForecast usually handles splitting internally via 'CrossValidation' or just simple slicing if we feed it correct inputs.
    # But standard way is to train on training set.
    
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
    
    # NeuralForecast TimesNet does not support hist_exog_list, so we treat everything as 'future' (known covariates)
    # This implies we assume we have the weather forecast for the horizon.
    futr_exog_list = cfg.past_covariates_cols + cfg.future_covariates_cols
    
    # Ensure all columns are numeric/float32 for PyTorch
    all_cols = ["y"] + futr_exog_list
    for col in all_cols:
        nf_df[col] = nf_df[col].astype("float32")
        
    print(f"Combined Exogenous Variables ({len(futr_exog_list)}): {futr_exog_list}")
    
    # Filter only relevant columns to remove any string/object columns (like 'public_holiday_name')
    # that might cause NeuralForecast to crash during internal conversion.
    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + futr_exog_list]
    
    # Drop NaNs (crucial for NeuralForecast, especially with lag features)
    initial_len = len(nf_df)
    nf_df = nf_df.dropna()
    print(f"Dropped {initial_len - len(nf_df)} rows containing NaNs (likely initial lags). Remaining: {len(nf_df)}")

    # Split into Train and Test (Validation handling is implicit or manual)
    # For training, we need input up to train_end
    # Actually, NF models need 'y' for the training part.
    
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
        hist_exog_list=None,
        futr_exog_list=futr_exog_list,
        loss=MSE(),
        max_steps=500, # Fast sanity check
        batch_size=32,
        scaler_type='standard', # Internal scaling
    )
    
    # 4. Train
    print("\n--- Training Model ---")
    nf = NeuralForecast(
        models=[model],
        freq='H'
    )
    
    nf.fit(df=train_df)
    print("Training Complete.")
    
    # Save Model (Standardized Location)
    print("\n--- Saving Model ---")
    model_save_dir = os.path.join(data_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, "timesnet_mse")
    nf.save(path=save_path, model_index=None, overwrite=True)
    print(f"Model saved to {save_path}")
    
    # 5. Predict / Sanity Check
    # We want to predict for a specific window in validation/test to see if it looks crazy.
    # Let's predict the first 24 hours of 2019 (Validation set)
    
    # We need to provide the future values for futr_exog
    # and recent history for hist_exog.
    # NeuralForecast's predict() method handles this if we pass the full history or specific future df.
    
    # For simplicity, let's predict using the 'cross_validation' method which is robust
    # We'll forecast the immediate next 24 hours after training end.
    
    print("\n--- Generating Forecast ---")
    # Prepare a dataframe containing the future exogenous variables for the forecast horizon
    # The 'predict' method automatically looks for future exogenous variables in the provided 'futr_df' 
    # if they are not in the training set (or if we want to extend).
    
    # Actually, easiest way to verify is to verify on a holdout set provided to fit? 
    # Or just use `predict` with the trained object.
    
    # Create a future dataframe for the next 24 hours
    last_train_date = train_df['ds'].max()
    future_dates = pd.date_range(last_train_date + pd.Timedelta(hours=1), periods=HORIZON, freq='H')
    
    # We need to extract the corresponding future covariates from the full original dataframe
    futr_df = nf_df[nf_df['ds'].isin(future_dates)].reset_index(drop=True)
    
    # Note: TimesNet might need 'hist_exog' (past weather) which is unknown for future?
    # Wait, 'hist_exog' are PAST covariates. We know them up to the prediction time usually?
    # In a real forecast, we don't know future 'rain'. 
    # Darts TCN treated weather as PAST covariates (only known up to T).
    # TimesNet 'hist_exog_list' implies columns that are only used as history input.
    # 'futr_exog_list' are columns known in future (calendar).
    
    # So if we put 'temp' in hist_exog_list, the model will NOT expect it in the forecast horizon. Code is correct.
    
    forecasts = nf.predict(futr_df=futr_df)
    print("\nForecasts Head:")
    print(forecasts.head())
    
    # 6. Basic Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    # Get actuals for this period
    actuals = futr_df['y'].values
    preds = forecasts['TimesNet'].values
    
    plt.plot(future_dates, actuals, label='Actual', marker='o')
    plt.plot(future_dates, preds, label='Prediction (MSE)', marker='x')
    plt.title(f"TimesNet Stage 1 Sanity Check (24h Forecast)\nLoss: MSE")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("timesnet/stage1_mse_plot.png")
    print("\nPlot saved to data/timesnet/stage1_mse_plot.png")
    
    # Calc MSE on this tiny sample
    mse_val = np.mean((actuals - preds)**2)
    print(f"\nSample MSE: {mse_val:.4f}")
    
    if mse_val > 1000: # Arbitrary large number check for explosion
        print("WARNING: MSE seems extremely high. Model might have diverged.")
    else:
        print("SUCCESS: Model produced reasonable-range predictions.")

if __name__ == "__main__":
    main()
