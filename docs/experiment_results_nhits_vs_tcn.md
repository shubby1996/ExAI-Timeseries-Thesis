# Experiment Analysis: Deterministic vs Probabilistic Forecasting

**Date:** 2025-12-12
**Status:** Phase 2 Complete. TCN Dropped.

## Executive Summary

The goal of this experiment was to determine if adding uncertainty quantification (via Quantile Regression) degrades the central prediction accuracy (Median/MAE) compared to a deterministic model optimized for the same metric (MAE).

**Key Findings:**
1.  **NHiTS (Success):** Probabilistic training **improved** accuracy.
    *   Deterministic MAE: `0.0693`
    *   Probabilistic MAE: `0.0651`
    *   *Conclusion:* Uncertainty acts as a beneficial regularizer for NHiTS.
2.  **TCN (Failure):** The TCN architecture proved numerically unstable with L1 and Quantile losses on this dataset.
    *   Both Deterministic (L1) and Probabilistic (Quantile) training runs diverged.
    *   Only TCN Optimized for MSE (Mean) was stable (`MAE 0.0447`).
    *   *Decision:* TCN is removed from the probabilistic comparison study.

---

## Detailed Results

### 1. NHiTS Performance
NHiTS demonstrated robust performance in both deterministic and probabilistic modes.

| Model Type | Loss Function | Scaled MAE | Status |
| :--- | :--- | :--- | :--- |
| **Deterministic** | L1Loss | **0.0693** | trained successfully |
| **Probabilistic** | Quantile (0.1, 0.5, 0.9) | **0.0651** | **+6.0% Improvement** |

**Interpretation:**
Contrary to the hypothesis that learning multiple quantiles might "distract" the model from the central tendency, it appears to have helped. The multi-objective nature of Quantile Loss (optimizing for 10th, 50th, and 90th percentiles simultaneously) likely provided a richer training signal, preventing overfitting to the mean/median alone.

### 2. TCN Performance (Architecture Analysis)
TCN showed significant sensitivity to the loss function.

| Model Type | Loss Function | Scaled MAE | Status |
| :--- | :--- | :--- | :--- |
| **Fresh Baseline** | MSE (Default) | **0.0447** | Stable, Best Accuracy |
| **Deterministic** | L1Loss | N/A | **Diverged** (Loss Explosion) |
| **Probabilistic** | Quantile (0.1, 0.5, 0.9) | N/A | **Diverged** (Loss ~103.0) |

**Root Cause Analysis:**
*   **Gradient Explosion:** TCN's dilated convolutions combined with the non-smooth gradients of L1/Quantile loss (constant gradient magnitude regardless of error size) led to weight updates that pushed parameters into unstable regions.
*   **Mitigation Failure:** Even with `weight_norm=False`, `dropout=0.1`, `lr=1e-5`, and `gradient_clip_val=0.1`, the model failed to converge.
*   **Conclusion:** TCN is not suitable for Quantile Regression on this specific unnormalized target scale without further architectural changes (e.g., residual scaling or log-space transformation).

## Next Steps

1.  **Drop TCN:** Remove all TCN experimental code (`train_tcn_deterministic.py`, `train_tcn_probabilistic.py`). Retain only the stable `train_tcn_fresh.py` (MSE) as a high-performance benchmark.
2.  **Select New Baseline:** Identify a different deterministic model to compare against NHiTS.
    *   *Candidate:* **TFT (Temporal Fusion Transformer)** - known for handling quantiles natively.
    *   *Candidate:* **XGBoost/LightGBM** - stable, tree-based baseline.
