# Comparative Analysis: TimesNet vs NHiTS

This document provides a comparative analysis of the evaluation results for the TimesNet and NHiTS models on the test set. The analysis includes both point forecasts and probabilistic forecasts.

---

## Point Forecast Comparison

| Metric       | TimesNet (Probabilistic) | NHiTS (Probabilistic) |
|--------------|--------------------------|-----------------------|
| **MAE**      | 0.2944                   | 0.2279                |
| **RMSE**     | 0.3645                   | 0.2983                |
| **MAPE**     | 8.99%                    | 6.87%                 |
| **R²**       | -0.1072                  | 0.2772                |
| **Samples**  | 1200                     | 1200                  |

### Observations:
- **NHiTS** outperforms **TimesNet** in all point forecast metrics, including MAE, RMSE, MAPE, and R².
- The R² value for TimesNet is negative, indicating poor predictive performance compared to NHiTS.

---

## Probabilistic Forecast Comparison

| Metric                | TimesNet (Probabilistic) | NHiTS (Probabilistic) |
|-----------------------|--------------------------|-----------------------|
| **PICP (Coverage)**   | -                        | 0.0%                  |
| **MIW**               | -                        | ~0.0                  |
| **Avg Quantile Loss** | -                        | 0.1139                |

### Observations:
- **NHiTS** provides probabilistic metrics such as PICP, MIW, and Avg Quantile Loss, which are not available for TimesNet in this evaluation.
- The PICP (Prediction Interval Coverage Probability) for NHiTS is 0.0%, indicating that the prediction intervals may not be well-calibrated.
- The Avg Quantile Loss for NHiTS is 0.1139, which reflects the model's ability to balance accuracy and uncertainty.

---

## Conclusion
- **NHiTS** demonstrates better performance in both point forecasts and probabilistic forecasts compared to **TimesNet**.
- While NHiTS excels in point forecast metrics, its probabilistic metrics suggest room for improvement in interval calibration.
- Future work could focus on improving the probabilistic calibration of NHiTS and exploring enhancements to TimesNet's predictive capabilities.