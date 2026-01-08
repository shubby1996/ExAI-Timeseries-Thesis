# Benchmarking NHiTS and TimesNet

This document summarizes the results of the unified benchmarking run for NHiTS and TimesNet on the Danish heat consumption dataset.

## Framework Overview

- **Dataset**: `nordbyen_features_engineered.csv`
- **Train Split**: Before `2018-12-31`
- **Val Split**: `2019-01-01` to `2019-12-31`
- **Test Split**: `2020-01-01` onwards
- **Evaluation**: Walk-forward validation with 50 windows (24h horizon each).

## Metrics Summary

| Model | MAE | RMSE | MAPE (%) | PICP (%) | MIW |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NHiTS** | 0.2787 | 0.3513 | 8.74 | 63.42 | 0.5992 |
| **TimesNet** | 0.2875 | 0.3568 | 8.83 | 78.08 | 0.9394 |

## Key Insights

1. **Accuracy**: NHiTS slightly outperformed TimesNet in terms of point accuracy (MAE/RMSE).
2. **Confidence Calibration**: TimesNet showed much better coverage (78% vs 63%) but with significantly wider prediction intervals. This suggests TimesNet is more conservative about its uncertainty.
3. **Scaling**: Standardized scaling using the `model_preprocessing.py` module ensured a fair comparison across libraries.

---
*Created on 2025-12-18*
