"""
Conformalized Quantile Regression (CQR) implementation.

This module provides post-hoc calibration for quantile predictions 
to achieve proper prediction interval coverage.

Reference: Romano et al. (2019) "Conformalized Quantile Regression"
"""

import numpy as np
from typing import Tuple, Dict


def compute_nonconformity_scores(
    y_true: np.ndarray, 
    y_low: np.ndarray, 
    y_high: np.ndarray
) -> np.ndarray:
    """
    Compute nonconformity scores for Conformalized Quantile Regression.
    
    Score measures how much the prediction interval misses the true value:
    s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i), 0)
    
    Args:
        y_true: True target values (n_samples,)
        y_low: Lower quantile predictions, e.g., q0.1 (n_samples,)
        y_high: Upper quantile predictions, e.g., q0.9 (n_samples,)
    
    Returns:
        scores: Nonconformity scores (n_samples,)
    """
    # How much does lower bound overshoot?
    lower_violation = y_low - y_true
    # How much does upper bound undershoot?
    upper_violation = y_true - y_high
    # Take the max violation (or 0 if interval captures the point)
    scores = np.maximum(np.maximum(lower_violation, upper_violation), 0.0)
    return scores


def calibrate_intervals(
    y_cal_true: np.ndarray,
    y_cal_low: np.ndarray,
    y_cal_high: np.ndarray,
    alpha: float = 0.2
) -> float:
    """
    Calibrate prediction intervals using a held-out calibration set.
    
    Computes the correction factor (s_hat) that should be added/subtracted 
    to achieve (1-alpha) coverage.
    
    Args:
        y_cal_true: True values on calibration set (n_cal,)
        y_cal_low: Lower quantile predictions on calibration set (n_cal,)
        y_cal_high: Upper quantile predictions on calibration set (n_cal,)
        alpha: Miscoverage level (default 0.2 for 80% coverage)
    
    Returns:
        s_hat: Calibration correction factor
    """
    # Compute nonconformity scores on calibration set
    scores = compute_nonconformity_scores(y_cal_true, y_cal_low, y_cal_high)
    
    # Find the (1-alpha) quantile of scores
    # We use ceil to be conservative (guarantees at least (1-alpha) coverage)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # Ensure it doesn't exceed 1.0
    
    s_hat = np.quantile(scores, q_level)
    return float(s_hat)


def apply_cqr_correction(
    y_low: np.ndarray,
    y_high: np.ndarray,
    s_hat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CQR correction to prediction intervals.
    
    Adjusted interval: [q_low - s_hat, q_high + s_hat]
    
    Args:
        y_low: Lower quantile predictions (n_samples,)
        y_high: Upper quantile predictions (n_samples,)
        s_hat: Calibration correction factor
    
    Returns:
        y_low_calibrated: Adjusted lower bounds (n_samples,)
        y_high_calibrated: Adjusted upper bounds (n_samples,)
    """
    y_low_calibrated = y_low - s_hat
    y_high_calibrated = y_high + s_hat
    return y_low_calibrated, y_high_calibrated


def cqr_calibrate(
    y_cal_true: np.ndarray,
    y_cal_low: np.ndarray,
    y_cal_high: np.ndarray,
    y_test_low: np.ndarray,
    y_test_high: np.ndarray,
    alpha: float = 0.2,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    End-to-end CQR calibration pipeline.
    
    1. Computes calibration correction on calibration set
    2. Applies correction to test predictions
    3. Returns calibrated intervals and diagnostics
    
    Args:
        y_cal_true: True values on calibration set (n_cal,)
        y_cal_low: Lower quantile predictions on calibration set (n_cal,)
        y_cal_high: Upper quantile predictions on calibration set (n_cal,)
        y_test_low: Lower quantile predictions on test set (n_test,)
        y_test_high: Upper quantile predictions on test set (n_test,)
        alpha: Miscoverage level (default 0.2 for 80% coverage)
        verbose: Print calibration info
    
    Returns:
        y_test_low_cal: Calibrated lower bounds for test set (n_test,)
        y_test_high_cal: Calibrated upper bounds for test set (n_test,)
        cal_info: Dictionary with calibration diagnostics
    """
    # Step 1: Calibrate on calibration set
    s_hat = calibrate_intervals(y_cal_true, y_cal_low, y_cal_high, alpha)
    
    # Step 2: Apply to test set
    y_test_low_cal, y_test_high_cal = apply_cqr_correction(y_test_low, y_test_high, s_hat)
    
    # Step 3: Compute diagnostics
    # Coverage on calibration set (should be close to 1-alpha after correction)
    cal_low_adj, cal_high_adj = apply_cqr_correction(y_cal_low, y_cal_high, s_hat)
    cal_coverage_before = np.mean((y_cal_true >= y_cal_low) & (y_cal_true <= y_cal_high))
    cal_coverage_after = np.mean((y_cal_true >= cal_low_adj) & (y_cal_true <= cal_high_adj))
    
    # Width statistics
    cal_width_before = np.mean(y_cal_high - y_cal_low)
    cal_width_after = np.mean(cal_high_adj - cal_low_adj)
    test_width_before = np.mean(y_test_high - y_test_low)
    test_width_after = np.mean(y_test_high_cal - y_test_low)
    
    cal_info = {
        's_hat': s_hat,
        'cal_coverage_before': cal_coverage_before * 100,
        'cal_coverage_after': cal_coverage_after * 100,
        'target_coverage': (1 - alpha) * 100,
        'cal_width_before': cal_width_before,
        'cal_width_after': cal_width_after,
        'test_width_before': test_width_before,
        'test_width_after': test_width_after,
        'width_increase_pct': ((cal_width_after - cal_width_before) / cal_width_before) * 100
    }
    
    if verbose:
        print("\n" + "="*70)
        print("CONFORMALIZED QUANTILE REGRESSION (CQR) CALIBRATION")
        print("="*70)
        print(f"Target Coverage: {cal_info['target_coverage']:.1f}%")
        print(f"Calibration correction (s_hat): {s_hat:.4f}")
        print(f"\nCalibration Set Performance:")
        print(f"  Coverage Before: {cal_info['cal_coverage_before']:.2f}%")
        print(f"  Coverage After:  {cal_info['cal_coverage_after']:.2f}%")
        print(f"  Width Before:    {cal_info['cal_width_before']:.4f}")
        print(f"  Width After:     {cal_info['cal_width_after']:.4f}")
        print(f"  Width Increase:  {cal_info['width_increase_pct']:.2f}%")
        print("="*70 + "\n")
    
    return y_test_low_cal, y_test_high_cal, cal_info


def split_validation_for_calibration(
    predictions_df,
    val_start_timestamp,
    cal_end_timestamp
):
    """
    Split validation predictions into calibration and test-validation sets.
    
    Args:
        predictions_df: DataFrame with columns [timestamp, actual, p10, p50, p90]
        val_start_timestamp: Start of validation period (for calibration)
        cal_end_timestamp: End of calibration period (rest becomes test-val)
    
    Returns:
        cal_df: Calibration set DataFrame
        test_val_df: Test-validation set DataFrame
    """
    cal_df = predictions_df[
        (predictions_df['timestamp'] >= val_start_timestamp) & 
        (predictions_df['timestamp'] < cal_end_timestamp)
    ].copy()
    
    test_val_df = predictions_df[
        predictions_df['timestamp'] >= cal_end_timestamp
    ].copy()
    
    return cal_df, test_val_df
