"""
TCN Model Evaluation - Fresh Implementation
Simple evaluation script for Nordbyen heat forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from darts.models import TCNModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import prepare_model_data


def calculate_mape(y_true, y_pred):
    """Calculate MAPE"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_tcn_fresh(
    csv_path: str,
    model_name: str,
    train_end_str: str,
    val_end_str: str,
    n_predictions: int = 50,
):
    """Evaluate TCN model on test set"""
    
    print("=" * 70)
    print("TCN MODEL EVALUATION - FRESH")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
    train_end = pd.Timestamp(train_end_str)
    val_end = pd.Timestamp(val_end_str)
    
    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path, train_end, val_end
    )
    
    print(f"  Test: {len(test_scaled['target'])} samples")
    
    # Create and load model
    print("\n[2/4] Loading model...")
    model = TCNModel(
        input_chunk_length=168,
        output_chunk_length=24,
        kernel_size=3,
        num_filters=32,
        num_layers=4,
        dilation_base=2,
        weight_norm=True,
        dropout=0.0,
        batch_size=32,
        n_epochs=1,
        model_name=f"{model_name}_eval",
        random_state=42,
    )
    
    # Initialize model
    dummy_target = train_scaled['target'][:200]
    dummy_past = train_scaled['past_covariates'][:200]
    dummy_future = train_scaled['future_covariates'][:200]
    if dummy_future:
        dummy_past = dummy_past.stack(dummy_future) if dummy_past else dummy_future
    
    model.fit(series=dummy_target, past_covariates=dummy_past, epochs=1, verbose=False)
    
    # Load weights
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    statedict_path = os.path.join(DATA_DIR, "models", f"{model_name}_statedict.pt")
    state_dict = torch.load(statedict_path)
    model.model.load_state_dict(state_dict)
    model.model.eval()  # CRITICAL: Set to eval mode
    
    print(f"  Model loaded and set to eval mode")
    
    # Prepare test data
    test_target = test_scaled['target']
    test_past = test_scaled['past_covariates']
    test_future = test_scaled['future_covariates']
    if test_future:
        test_past = test_past.stack(test_future) if test_past else test_future
    
    # Evaluate
    print("\n[3/4] Running predictions...")
    all_actuals = []
    all_predictions = []
    all_timestamps = []
    
    for i in range(n_predictions):
        start_idx = i * 24
        end_idx = start_idx + 168
        
        if end_idx + 24 > len(test_target):
            break
        
        try:
            pred_scaled = model.predict(
                n=24,
                series=test_target[:end_idx],
                past_covariates=test_past[:end_idx],
            )
            
            actual_scaled = test_target[end_idx:end_idx+24]
            pred_orig = state.target_scaler.inverse_transform(pred_scaled)
            actual_orig = state.target_scaler.inverse_transform(actual_scaled)
            
            pred_values = pred_orig.values().flatten()
            actual_values = actual_orig.values().flatten()
            
            if not np.isnan(pred_values).any() and not np.isnan(actual_values).any():
                all_predictions.extend(pred_values)
                all_actuals.extend(actual_values)
                all_timestamps.extend(pred_orig.time_index.tolist())
                
                if (i + 1) % 10 == 0:
                    print(f"  Window {i+1}/{n_predictions}")
        except Exception as e:
            print(f"  Error at window {i+1}: {e}")
            continue
    
    print(f"  Completed {len(all_actuals)//24} windows")
    
    # Calculate metrics
    print("\n[4/4] Calculating metrics...")
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    
    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    mape = calculate_mape(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)
    
    print("\n" + "=" * 70)
    print("RESULTS - TCN FRESH")
    print("=" * 70)
    print(f"\nMAE:  {mae:.4f} MW")
    print(f"RMSE: {rmse:.4f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    print(f"\nSamples: {len(all_actuals)}")
    print("=" * 70)
    
    # Save results
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    metrics_df = pd.DataFrame([{
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'n_samples': len(all_actuals),
    }])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "tcn_fresh_metrics.csv"), index=False)
    
    results_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': all_actuals,
        'predicted': all_predictions,
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, "tcn_fresh_predictions.csv"), index=False)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return metrics_df, results_df


if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    
    evaluate_tcn_fresh(
        csv_path=CSV_PATH,
        model_name="tcn_nordbyen_fresh",
        train_end_str="2018-12-31 23:00:00+00:00",
        val_end_str="2019-12-31 23:00:00+00:00",
        n_predictions=50,
    )
