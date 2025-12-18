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
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_timesnet(
    csv_path: str,
    model_dir: str,
    model_name: str = "timesnet_probabilistic_q",
    train_end_str: str = "2018-12-31 23:00:00",
    val_end_str: str = "2019-12-31 23:00:00",
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
    futr_exog_list = cfg.past_covariates_cols + cfg.future_covariates_cols
    ids_cols = ["unique_id", "ds", "y"]
    nf_df = nf_df[ids_cols + futr_exog_list]

    # Define test start (same as other scripts)
    test_start = pd.Timestamp("2020-01-01 00:00:00")
    # Ensure timestamps have same timezone-awareness as nf_df
    tz = nf_df['ds'].dt.tz
    if tz is not None:
        test_start = test_start.tz_localize(tz)

    # Load NF model (auto-detect if missing)
    nf_path = os.path.join(model_dir, model_name)
    if not os.path.exists(nf_path):
        # try to find a timesnet folder under model_dir or ../models
        def search_dir(d):
            if not os.path.exists(d):
                return None
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

        chosen = search_dir(model_dir)
        if chosen is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            alt = os.path.join(project_root, "models")
            chosen = search_dir(alt)

        if chosen:
            nf_path = chosen
            print(f"[INFO] Auto-detected TimesNet model directory: {nf_path}")
        else:
            raise FileNotFoundError(f"TimesNet model directory not found: {nf_path}")

    nf = NeuralForecast.load(path=nf_path)

    # Walk-forward evaluation
    all_actuals = []
    all_predictions = []
    all_timestamps = []

    # We'll evaluate n_predictions windows spaced by `stride` hours starting at test_start
    for i in range(n_predictions):
        pred_start = test_start + pd.Timedelta(hours=i * stride)
        pred_end = pred_start + pd.Timedelta(hours=23)

        # Build history (168 hours up to pred_start - 1) and future (pred_start..pred_end)
        history_start = pred_start - pd.Timedelta(hours=168)
        history_df = nf_df[(nf_df['ds'] >= history_start) & (nf_df['ds'] <= pred_start - pd.Timedelta(hours=1))]
        future_df = nf_df[(nf_df['ds'] >= pred_start) & (nf_df['ds'] <= pred_end)]

        # Debug prints to diagnose empty windows
        if i < 3:
            print(f"Window {i+1}: pred_start={pred_start}, history_rows={len(history_df)}, future_rows={len(future_df)}")

        if len(history_df) < 168 or len(future_df) < 24:
            # Not enough data to evaluate this window
            break

        try:
            fcst = nf.predict(df=history_df.reset_index(drop=True), futr_df=future_df.reset_index(drop=True))
            # Choose median-like column
            candidates = [c for c in fcst.columns if "median" in c or "0.5" in c]
            if candidates:
                pred_col = candidates[0]
            elif "TimesNet" in fcst.columns:
                pred_col = "TimesNet"
            else:
                pred_col = fcst.columns[0]

            preds = fcst[pred_col].values
            actuals = future_df['y'].values

            all_predictions.extend(preds)
            all_actuals.extend(actuals)
            # timestamps
            all_timestamps.extend(list(pd.date_range(pred_start, periods=24, freq='H')))

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ✓ Completed window {i+1}")

        except Exception as e:
            print(f"  ✗ Error predicting window {i+1}: {e}")
            continue

    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)

    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    mape = calculate_mape(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions) if len(all_actuals) > 0 else float('nan')

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'n_samples': len(all_actuals)
    }

    results_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': all_actuals,
        'predicted': all_predictions,
        'error': all_actuals - all_predictions,
        'abs_error': np.abs(all_actuals - all_predictions),
    })

    # Save to results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(results_dir, 'timesnet_evaluation_metrics.csv')
    preds_path = os.path.join(results_dir, 'timesnet_evaluation_predictions.csv')

    metrics_df.to_csv(metrics_path, index=False)
    results_df.to_csv(preds_path, index=False)

    print(f"\n✓ Metrics saved to: {metrics_path}")
    print(f"✓ Predictions saved to: {preds_path}")

    # Calculate probabilistic metrics
    def calculate_picp(y_true, y_low, y_high):
        """Calculate Prediction Interval Coverage Probability (PICP)."""
        within_interval = (y_true >= y_low) & (y_true <= y_high)
        return np.mean(within_interval) * 100

    def calculate_miw(y_low, y_high):
        """Calculate Mean Interval Width (MIW)."""
        return np.mean(y_high - y_low)

    def calculate_quantile_loss(y_true, y_pred, q):
        """Calculate Quantile Loss (Pinball Loss) for a specific quantile q."""
        errors = y_true - y_pred
        return np.mean(np.maximum(q * errors, (q - 1) * errors))

    # Extract probabilistic forecasts
    all_predictions_low = []
    all_predictions_high = []

    for i in range(n_predictions):
        pred_start = test_start + pd.Timedelta(hours=i * stride)
        pred_end = pred_start + pd.Timedelta(hours=23)

        # Build history (168 hours up to pred_start - 1) and future (pred_start..pred_end)
        history_start = pred_start - pd.Timedelta(hours=168)
        history_df = nf_df[(nf_df['ds'] >= history_start) & (nf_df['ds'] <= pred_start - pd.Timedelta(hours=1))]
        future_df = nf_df[(nf_df['ds'] >= pred_start) & (nf_df['ds'] <= pred_end)]

        try:
            fcst = nf.predict(df=history_df.reset_index(drop=True), futr_df=future_df.reset_index(drop=True))
            # Extract quantiles
            p_low = fcst['TimesNet-q-0.1'].values
            p_med = fcst['TimesNet-q-0.5'].values
            p_high = fcst['TimesNet-q-0.9'].values

            all_predictions.extend(p_med)
            all_predictions_low.extend(p_low)
            all_predictions_high.extend(p_high)
            all_actuals.extend(actuals)
            all_timestamps.extend(list(pd.date_range(pred_start, periods=24, freq='H')))

            # Debug: Print available forecast columns
            print(f"Forecast columns: {fcst.columns.tolist()}")

            # Debug: Print quantile values for the first few rows
            print(f"Quantile values (low, median, high):\n{fcst[['TimesNet-q-0.1', 'TimesNet-q-0.5', 'TimesNet-q-0.9']].head()}")

        except Exception as e:
            print(f"  \u2717 Error predicting window {i+1}: {e}")
            continue

    # Calculate probabilistic metrics
    picp = calculate_picp(all_actuals, all_predictions_low, all_predictions_high)
    miw = calculate_miw(all_predictions_low, all_predictions_high)
    ql_10 = calculate_quantile_loss(all_actuals, all_predictions_low, 0.1)
    ql_50 = calculate_quantile_loss(all_actuals, all_predictions, 0.5)
    ql_90 = calculate_quantile_loss(all_actuals, all_predictions_high, 0.9)
    avg_ql = (ql_10 + ql_50 + ql_90) / 3

    metrics.update({
        'PICP (Coverage)': picp,
        'MIW': miw,
        'Avg Quantile Loss': avg_ql
    })

    # Save updated metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(results_dir, 'timesnet_evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n\u2713 Updated metrics saved to: {metrics_path}")

    return metrics, results_df


if __name__ == '__main__':
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    CSV_PATH = os.path.join(DATA_DIR, 'nordbyen_features_engineered.csv')
    MODEL_DIR = os.path.join(DATA_DIR, 'timesnet', 'models')

    evaluate_timesnet(
        csv_path=CSV_PATH,
        model_dir=MODEL_DIR,
        model_name='timesnet_mae',
        stride=24,
        n_predictions=50,
    )
