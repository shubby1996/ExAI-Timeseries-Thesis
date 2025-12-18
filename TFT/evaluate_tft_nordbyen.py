"""
TFT Model Evaluation Script for Nordbyen Heat Consumption Forecasting.

This script:
1. Loads the trained model and test data
2. Performs walk-forward validation on the test set
3. Calculates performance metrics (MAE, RMSE, MAPE, R²)
4. Saves results to CSV
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from darts import TimeSeries
from darts.models import TFTModel

# Add parent directory to path to allow importing model_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
    load_preprocessing_state,
)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MAPE in percentage
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_picp(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).
    Percentage of actual values falling within the prediction interval.
    """
    within_interval = (y_true >= y_low) & (y_true <= y_high)
    return np.mean(within_interval) * 100


def calculate_miw(y_low: np.ndarray, y_high: np.ndarray) -> float:
    """
    Calculate Mean Interval Width (MIW).
    Average width of the prediction interval.
    """
    return np.mean(y_high - y_low)


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Calculate Quantile Loss (Pinball Loss) for a specific quantile q.
    L(y, ŷ, q) = q * (y - ŷ) if y >= ŷ else (1 - q) * (ŷ - y)
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(q * errors, (q - 1) * errors))


def evaluate_on_test_set(
    csv_path: str,
    model_path: str,
    prep_state_path: str,
    train_end_str: str,
    val_end_str: str,
    stride: int = 24,  # Evaluate every stride hours
    n_predictions: int = 10,  # Number of prediction windows
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate TFT model on test set using walk-forward validation.
    
    Parameters
    ----------
    csv_path : str
        Path to engineered features CSV
    model_path : str
        Path to trained model
    prep_state_path : str
        Path to preprocessing state
    train_end_str : str
        Training end date (for data split)
    val_end_str : str
        Validation end date (for data split)
    stride : int
        Hours between each prediction window
    n_predictions : int
        Number of prediction windows to evaluate
    
    Returns
    -------
    metrics : dict
        Dictionary with MAE, RMSE, MAPE, R² scores
    results_df : pd.DataFrame
        DataFrame with all predictions and actuals
    """
    print("=" * 70)
    print("TFT MODEL EVALUATION - TEST SET")
    print("=" * 70)
    
    # Load data and model
    print("\n[1/4] Loading data and model...")
    train_end = pd.Timestamp(train_end_str)
    val_end = pd.Timestamp(val_end_str)
    
    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path, train_end, val_end
    )
    
    model = TFTModel.load(model_path)
    print(f"  ✓ Model loaded")
    print(f"  ✓ Test set: {len(test_scaled['target'])} samples")
    
    # Extract test series
    target_test = test_scaled['target']
    past_test = test_scaled['past_covariates']
    future_test = test_scaled['future_covariates']
    
    # Prepare for walk-forward validation
    print(f"\n[2/4] Performing walk-forward validation...")
    print(f"  - Prediction horizon: 24 hours")
    print(f"  - Stride: {stride} hours")
    print(f"  - Number of windows: {n_predictions}")
    
    all_actuals = []
    all_predictions = []
    all_predictions_low = []
    all_predictions_high = []
    all_timestamps = []
    
    input_chunk = 168  # 7 days
    output_chunk = 24  # 24 hours
    
    # Walk through test set
    for i in range(n_predictions):
        start_idx = i * stride
        end_idx = start_idx + input_chunk
        
        # Check if we have enough data
        if end_idx + output_chunk > len(target_test):
            print(f"  ✓ Completed {i} prediction windows (reached end of test set)")
            break
        
        # Get historical context (input window)
        historical_target = target_test[:end_idx]
        historical_past = past_test[:end_idx] if past_test else None
        
        # Get future covariates for forecast horizon
        future_for_forecast = future_test[end_idx:end_idx+output_chunk] if future_test else None
        
        # Make prediction
        try:
            pred_scaled = model.predict(
                n=output_chunk,
                series=historical_target,
                past_covariates=historical_past,
                future_covariates=future_test,  # Full future covariates series
                num_samples=100,  # Generate probabilistic forecast
            )
            
            # Inverse transform (handles probabilistic series)
            pred_orig = state.target_scaler.inverse_transform(pred_scaled)
            
            actual_scaled = target_test[end_idx:end_idx+output_chunk]
            actual_orig = state.target_scaler.inverse_transform(actual_scaled)
            
            # Extract quantiles using Darts quantile method
            # quantile() returns a deterministic TimeSeries
            p_low = pred_orig.quantile(0.1).values().flatten()
            p_med = pred_orig.quantile(0.5).values().flatten()
            p_high = pred_orig.quantile(0.9).values().flatten()
            
            # Store results
            all_predictions.extend(p_med)
            all_predictions_low.extend(p_low)
            all_predictions_high.extend(p_high)
            all_actuals.extend(actual_orig.values().flatten())
            all_timestamps.extend(pred_orig.time_index.tolist())
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ✓ Window {i+1}/{n_predictions} complete")
                
        except Exception as e:
            print(f"  ✗ Error at window {i+1}: {repr(e)}")
            continue
    
    print(f"\n[3/4] Calculating metrics...")
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    all_predictions_low = np.array(all_predictions_low)
    all_predictions_high = np.array(all_predictions_high)
    
    # Calculate metrics
    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    mape = calculate_mape(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)
    
    # Uncertainty metrics
    picp = calculate_picp(all_actuals, all_predictions_low, all_predictions_high)
    miw = calculate_miw(all_predictions_low, all_predictions_high)
    
    # Average Quantile Loss (approx CRPS)
    ql_10 = calculate_quantile_loss(all_actuals, all_predictions_low, 0.1)
    ql_50 = calculate_quantile_loss(all_actuals, all_predictions, 0.5)
    ql_90 = calculate_quantile_loss(all_actuals, all_predictions_high, 0.9)
    avg_ql = (ql_10 + ql_50 + ql_90) / 3
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'PICP (Coverage)': picp,
        'MIW': miw,
        'Avg Quantile Loss': avg_ql,
        'n_samples': len(all_actuals)
    }
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': all_actuals,
        'predicted': all_predictions,
        'predicted_low': all_predictions_low,
        'predicted_high': all_predictions_high,
        'error': all_actuals - all_predictions,
        'abs_error': np.abs(all_actuals - all_predictions),
        'pct_error': np.abs((all_actuals - all_predictions) / all_actuals) * 100
    })
    
    print("\n[4/4] Evaluation complete!")
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Value':<15} {'Interpretation'}")
    print("-" * 70)
    print(f"{'MAE':<25} {mae:.4f} MW      Average absolute error")
    print(f"{'RMSE':<25} {rmse:.4f} MW      Root mean squared error")
    print(f"{'MAPE':<25} {mape:.2f}%         Mean absolute percentage error")
    print(f"{'R²':<25} {r2:.4f}          Variance explained")
    print("-" * 70)
    print(f"{'PICP (Coverage 80%)':<25} {picp:.2f}%         % Actuals inside interval (Target: 80%)")
    print(f"{'MIW':<25} {miw:.4f} MW      Mean Interval Width (Sharpness)")
    print(f"{'Avg Quantile Loss':<25} {avg_ql:.4f}          Combined accuracy & uncertainty score")
    print("-" * 70)
    print(f"{'Samples':<25} {len(all_actuals):<15} Total predictions evaluated")
    print("=" * 70)
    
    return metrics, results_df


if __name__ == "__main__":
    # Configuration
    # Configuration
    # Use relative path to data directory (parent of this script)
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_PATH = os.path.join(DATA_DIR, "models", "tft_nordbyen.pt")
    PREP_STATE_PATH = os.path.join(DATA_DIR, "models", "tft_nordbyen_preprocessing_state.pkl")
    
    # Same splits used in training
    TRAIN_END = "2018-12-31 23:00:00+00:00"
    VAL_END = "2019-12-31 23:00:00+00:00"
    
    # Run evaluation
    metrics, results_df = evaluate_on_test_set(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        prep_state_path=PREP_STATE_PATH,
        train_end_str=TRAIN_END,
        val_end_str=VAL_END,
        stride=24,  # Evaluate every 24 hours
        n_predictions=50,  # 50 prediction windows (~50 days of forecasts)
    )
    
    # Save results
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Save detailed results
    results_path = os.path.join(RESULTS_DIR, "evaluation_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✓ Predictions saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
