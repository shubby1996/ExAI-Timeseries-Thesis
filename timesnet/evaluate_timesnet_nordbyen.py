"""
TimesNet Evaluation Script for Nordbyen Heat Consumption Forecasting.

Performs walk-forward evaluation using a saved NeuralForecast TimesNet model.
Saves metrics and detailed predictions to `data/results/`.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neuralforecast import NeuralForecast

# Add parent directory to path to import model_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_preprocessing as mp


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_picp(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Calculate Prediction Interval Coverage Probability (PICP)."""
    if len(y_true) == 0:
        return 0.0
    within_interval = (y_true >= y_low) & (y_true <= y_high)
    return np.mean(within_interval) * 100


def calculate_miw(y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Calculate Mean Interval Width (MIW)."""
    if len(y_low) == 0:
        return 0.0
    return np.mean(y_high - y_low)


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Calculate Quantile Loss (Pinball Loss) for a specific quantile q."""
    if len(y_true) == 0:
        return 0.0
    errors = y_true - y_pred
    return np.mean(np.maximum(q * errors, (q - 1) * errors))


def evaluate_timesnet(
    csv_path: str,
    model_dir: str,
    model_name: str = "timesnet_probabilistic_q",
    stride: int = 24,
    n_predictions: int = 50,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    print("=" * 70)
    print("TimesNet EVALUATION - TEST SET")
    print("=" * 70)

    # Load full df
    df_full = mp.load_and_validate_features(csv_path)

    # Prepare NeuralForecast long df
    nf_df = df_full.reset_index().rename(columns={"timestamp": "ds", "heat_consumption": "y"})
    nf_df["unique_id"] = "nordbyen"

    cfg = mp.default_feature_config()
    exog_cols = cfg.past_covariates_cols + cfg.future_covariates_cols
    
    # Filter numeric exog for NeuralForecast
    numeric_exog = [c for c in exog_cols if c in nf_df.columns and np.issubdtype(nf_df[c].dtype, np.number)]
    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + numeric_exog].dropna()

    # Define test start
    test_start = pd.Timestamp("2020-01-01 00:00:00")
    tz = nf_df['ds'].dt.tz
    if tz is not None:
        test_start = test_start.tz_localize(tz)

    # Load NF model
    nf_path = os.path.join(model_dir, model_name)
    if not os.path.exists(nf_path):
        raise FileNotFoundError(f"TimesNet model directory not found: {nf_path}")

    print(f"Loading model from: {nf_path}")
    nf = NeuralForecast.load(path=nf_path)

    all_actuals = []
    all_predictions = []
    all_predictions_low = []
    all_predictions_high = []
    all_timestamps = []

    for i in range(n_predictions):
        pred_start = test_start + pd.Timedelta(hours=i * stride)
        pred_end = pred_start + pd.Timedelta(hours=23)

        history_start = pred_start - pd.Timedelta(hours=168)
        history_df = nf_df[(nf_df['ds'] >= history_start) & (nf_df['ds'] <= pred_start - pd.Timedelta(hours=1))]
        future_df = nf_df[(nf_df['ds'] >= pred_start) & (nf_df['ds'] <= pred_end)]

        if len(history_df) < 168 or len(future_df) < 24:
            break

        try:
            fcst = nf.predict(df=history_df.reset_index(drop=True), futr_df=future_df.reset_index(drop=True))
            
            # Identify columns using NeuralForecast naming conventions
            # Probabilistic: ModelName-lo-80.0, ModelName-median, ModelName-hi-80.0
            # OR ModelName-q-0.1, ModelName-q-0.5, ModelName-q-0.9
            cols = fcst.columns.tolist()
            
            def find_col(suffixes):
                for s in suffixes:
                    for c in cols:
                        if c.endswith(s):
                            return c
                return None

            col_med = find_col(['median', '-q-0.5', 'TimesNet'])
            col_low = find_col(['-lo-80.0', '-q-0.1'])
            col_high = find_col(['-hi-80.0', '-q-0.9'])

            p_med = fcst[col_med].values if col_med else fcst.iloc[:, -1].values
            p_low = fcst[col_low].values if col_low else p_med
            p_high = fcst[col_high].values if col_high else p_med

            actuals = future_df['y'].values

            all_predictions.extend(p_med.tolist())
            all_predictions_low.extend(p_low.tolist())
            all_predictions_high.extend(p_high.tolist())
            all_actuals.extend(actuals.tolist())
            all_timestamps.extend(list(pd.date_range(pred_start, periods=24, freq='h')))

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  ✓ Completed window {i+1}/{n_predictions}")

        except Exception as e:
            print(f"  ✗ Error predicting window {i+1}: {e}")
            continue

    if not all_actuals:
        print("✗ No predictions were successfully made.")
        return {}, pd.DataFrame()

    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    all_predictions_low = np.array(all_predictions_low)
    all_predictions_high = np.array(all_predictions_high)

    # Metrics calculation
    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    mape = calculate_mape(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)
    
    picp = calculate_picp(all_actuals, all_predictions_low, all_predictions_high)
    miw = calculate_miw(all_predictions_low, all_predictions_high)
    ql_10 = calculate_quantile_loss(all_actuals, all_predictions_low, 0.1)
    ql_50 = calculate_quantile_loss(all_actuals, all_predictions, 0.5)
    ql_90 = calculate_quantile_loss(all_actuals, all_predictions_high, 0.9)
    avg_ql = (ql_10 + ql_50 + ql_90) / 3

    metrics = {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2,
        'PICP (Coverage)': picp, 'MIW': miw, 'Avg Quantile Loss': avg_ql,
        'n_samples': len(all_actuals)
    }

    results_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': all_actuals,
        'predicted': all_predictions,
        'predicted_low': all_predictions_low,
        'predicted_high': all_predictions_high,
    })

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, 'timesnet_evaluation_metrics.csv'), index=False)
    results_df.to_csv(os.path.join(results_dir, 'timesnet_evaluation_predictions.csv'), index=False)

    print("\nPERFORMANCE METRICS")
    for m, v in metrics.items(): print(f"{m:<25} {v:.4f}")

    return metrics, results_df


if __name__ == '__main__':
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    evaluate_timesnet(
        csv_path=os.path.join(DATA_DIR, 'nordbyen_features_engineered.csv'),
        model_dir=os.path.join(DATA_DIR, 'models'),
        stride=24, n_predictions=50
    )
