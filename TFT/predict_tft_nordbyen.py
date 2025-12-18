"""
TFT Model Prediction Script for Nordbyen Heat Consumption Forecasting.

This script:
1. Loads the trained TFT model
2. Loads the preprocessing state (scalers + config)
3. Extends dataset with future calendar features
4. Generates predictions for the next n hours
5. Returns predictions in original units
"""

import os
import pandas as pd
import sys
from typing import Optional

from darts import TimeSeries
from darts.models import TFTModel

# Add parent directory to path to allow importing model_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    load_and_validate_features,
    load_preprocessing_state,
    apply_state_to_full_df,
    append_future_calendar_and_holidays,
)


def predict_next_horizon(
    csv_path: str,
    model_path: str,
    prep_state_path: str,
    n: int = 24,
    school_holidays_path: Optional[str] = None,
    num_samples: int = 100,
    print_summary: bool = True,
) -> TimeSeries:
    """
    Load the trained TFT model and preprocessing state, and predict the next `n` hours.
    
    Returns a probabilistic TimeSeries (with samples) if num_samples > 1.
    """
    print("=" * 70)
    print("TFT PREDICTION - NORDBYEN HEAT CONSUMPTION")
    print("=" * 70)

    # 1. Load feature config and preprocessing state
    print("\n[1/6] Loading preprocessing state...")
    cfg = default_feature_config()
    state = load_preprocessing_state(prep_state_path)
    print(f"  ✓ Loaded preprocessing state")

    # 2. Load and validate the full engineered CSV
    print("\n[2/6] Loading engineered CSV...")
    df_full = load_and_validate_features(csv_path, cfg)
    print(f"  ✓ Loaded {len(df_full)} rows")
    print(f"  ✓ Last timestamp: {df_full.index.max()}")

    # 3. Extend with future calendar + holidays
    print(f"\n[3/6] Extending with {n} future hours (calendar + holidays)...")
    df_extended = append_future_calendar_and_holidays(
        df_full=df_full,
        n_future=n,
        freq="H",
        school_holidays_path=school_holidays_path,
        country="DK",
    )
    print(f"  ✓ Extended to {len(df_extended)} rows")
    print(f"  ✓ New last timestamp: {df_extended.index.max()}")

    # 4. Build and scale full series using the stored scalers
    print("\n[4/6] Building and scaling TimeSeries...")
    scaled_series = apply_state_to_full_df(df_extended, state)
    
    # IMPORTANT: Target and Past Covariates must end at the "current" time (T)
    # Future Covariates must extend to T + n
    # df_extended has length T + n
    
    target_scaled = scaled_series["target"][:-n]
    past_scaled = scaled_series["past_covariates"][:-n] if scaled_series["past_covariates"] else None
    future_scaled = scaled_series["future_covariates"] # Keep full length (T + n)
    
    print(f"  ✓ Scaled series ready")
    print(f"  ✓ Target length: {len(target_scaled)} (ends at {target_scaled.end_time()})")
    print(f"  ✓ Future covariates length: {len(future_scaled)} (ends at {future_scaled.end_time()})")

    # 5. Load the trained TFT model
    print("\n[5/6] Loading trained TFT model...")
    model = TFTModel.load(model_path)
    print(f"  ✓ Model loaded")

    # 6. Predict `n` steps ahead
    print(f"\n[6/6] Generating {n}-hour probabilistic forecast ({num_samples} samples)...")
    pred_scaled: TimeSeries = model.predict(
        n=n,
        series=target_scaled,
        past_covariates=past_scaled,
        future_covariates=future_scaled,
        num_samples=num_samples,
    )

    # 7. Inverse transform predictions to original scale
    pred_orig: TimeSeries = state.target_scaler.inverse_transform(pred_scaled)

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    if print_summary:
        start = pred_orig.start_time()
        end = pred_orig.end_time()
        print(f"\n✓ Forecast horizon: {start} to {end}")
        
        # Calculate quantiles for summary
        p10 = pred_orig.quantile(0.1).values().flatten()
        p50 = pred_orig.quantile(0.5).values().flatten()
        p90 = pred_orig.quantile(0.9).values().flatten()
        
        print(f"\nSample predictions (first 5 hours):")
        print(f"{'Time':<20} {'Low (10%)':<12} {'Median':<12} {'High (90%)':<12}")
        print("-" * 60)
        times = pred_orig.time_index
        for i in range(min(5, n)):
            print(f"{str(times[i]):<20} {p10[i]:.2f} MW     {p50[i]:.2f} MW     {p90[i]:.2f} MW")
            
        print(f"\nStatistics (Median):")
        print(f"  Mean: {p50.mean():.4f} MW")
        print(f"  Min:  {p50.min():.4f} MW")
        print(f"  Max:  {p50.max():.4f} MW")

    return pred_orig

    return pred_orig


if __name__ == "__main__":
    # Configuration
    # Configuration
    # Use relative path to data directory (parent of this script)
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_PATH = os.path.join(DATA_DIR, "models", "tft_nordbyen.pt")
    PREP_STATE_PATH = os.path.join(DATA_DIR, "models", "tft_nordbyen_preprocessing_state.pkl")
    SCHOOL_HOL_PATH = os.path.join(DATA_DIR, "school_holidays.csv")

    # Predict next 24 hours from the last timestamp in the CSV
    # NOTE: This now performs TRUE forward forecasting by automatically extending
    # the dataset with future calendar/holiday features.
    
    pred = predict_next_horizon(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        prep_state_path=PREP_STATE_PATH,
        n=24,
        school_holidays_path=SCHOOL_HOL_PATH,
        num_samples=100,
        print_summary=True,
    )
    
    # Extract quantiles for saving
    p10 = pred.quantile(0.1).values().flatten()
    p50 = pred.quantile(0.5).values().flatten()
    p90 = pred.quantile(0.9).values().flatten()
    
    # Create DataFrame
    pred_df = pd.DataFrame({
        'timestamp': pred.time_index,
        'predicted': p50,
        'predicted_low': p10,
        'predicted_high': p90
    })
    
    output_path = os.path.join(DATA_DIR, "predictions_future_24h.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
