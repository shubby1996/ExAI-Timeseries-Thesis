# Conformalized Quantile Regression (CQR) Implementation

## Overview

This implementation adds **Conformalized Quantile Regression (CQR)** to improve the calibration of prediction intervals from quantile forecasting models. CQR is a post-hoc calibration technique that adjusts prediction intervals to achieve proper coverage guarantees.

## Problem Statement

Standard quantile regression models (NHITS, TimesNet, TFT) often produce prediction intervals with **poor coverage**:
- **Target**: 80% of actual values should fall within the [q0.1, q0.9] interval
- **Reality**: Often only 60-70% coverage (undercoverage)
- **Consequence**: Unreliable uncertainty estimates

## Solution: Conformalized Quantile Regression

CQR adjusts prediction intervals using a held-out calibration set to achieve the target coverage level.

### Algorithm

1. **Train model** on training data (2018)
2. **Get calibration predictions** on first half of validation data (Jan-Jun 2019)
3. **Compute nonconformity scores** on calibration set:
   ```
   s_i = max(q_0.1(x_i) - y_i, y_i - q_0.9(x_i), 0)
   ```
4. **Find calibration quantile** s_hat = (1-α)-quantile of {s_i}
5. **Apply correction** to test predictions:
   ```
   Calibrated interval: [q_0.1 - s_hat, q_0.9 + s_hat]
   ```

### Theoretical Guarantee

Under exchangeability assumption, CQR provides:
```
P(y_new ∈ [q_0.1 - s_hat, q_0.9 + s_hat]) ≥ 1 - α
```

For α = 0.2, we get **at least 80% coverage** on new data.

## Implementation Details

### Files Added

1. **`conformal_calibration.py`**: Core CQR implementation
   - `compute_nonconformity_scores()`: Calculate conformity scores
   - `calibrate_intervals()`: Find calibration correction factor
   - `apply_cqr_correction()`: Apply correction to new predictions
   - `cqr_calibrate()`: End-to-end calibration pipeline

2. **`example_cqr_calibration.py`**: Usage examples

### Files Modified

1. **`benchmarker.py`**:
   - Added `cqr_s_hat` attribute to `ModelAdapter`
   - Implemented `get_calibration_predictions()` in all adapters
   - Modified `evaluate()` to apply CQR correction
   - Updated `Benchmarker.run()` with `use_cqr` parameter

2. **`water_tommerby_benchmark/scripts/run_benchmarker_water.py`**:
   - Added `--no-cqr` flag to disable calibration

## Usage

### Command Line

#### With CQR (default, recommended):
```bash
python benchmarker.py NHITS_Q TIMESNET_Q TFT_Q
```

#### Without CQR:
```bash
python benchmarker.py --no-cqr NHITS_Q TIMESNET_Q TFT_Q
```

#### For Water Tommerby benchmark:
```bash
cd water_tommerby_benchmark/scripts
python run_benchmarker_water.py --models NHITS_Q TIMESNET_Q
python run_benchmarker_water.py --no-cqr --models NHITS_Q  # without CQR
```

### Python API

```python
from benchmarker import Benchmarker

# Initialize benchmarker
csv_path = "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv"
models = ["NHITS_Q", "TIMESNET_Q", "TFT_Q"]

benchmarker = Benchmarker(csv_path, models, dataset="Water (Tommerby)")

# Run with CQR calibration (recommended)
benchmarker.run(use_cqr=True, alpha=0.2)  # 80% coverage target

# Or run without CQR (baseline)
benchmarker.run(use_cqr=False)
```

### Custom Calibration

```python
import conformal_calibration as cqr

# Get predictions on calibration set
y_cal_true = ...   # Actual values
y_cal_low = ...    # q0.1 predictions
y_cal_high = ...   # q0.9 predictions

# Compute calibration factor
s_hat = cqr.calibrate_intervals(y_cal_true, y_cal_low, y_cal_high, alpha=0.2)

# Apply to test predictions
y_test_low_cal, y_test_high_cal = cqr.apply_cqr_correction(
    y_test_low, y_test_high, s_hat
)
```

## Expected Results

### Before CQR (Typical Issues)
- **PICP**: 60-70% (undercoverage)
- **MIW**: Narrow intervals
- **Winkler**: High scores due to miss penalties

### After CQR (Improvements)
- **PICP**: ~80% (target coverage achieved)
- **MIW**: Slightly wider (necessary for proper coverage)
- **Winkler**: Lower scores (fewer misses, net improvement)
- **CRPS**: Similar or slightly better

## Data Split Strategy

```
2018          |  2019 (Validation)      |  2020
Training      |  Cal  |  Test-Val       |  Test
════════      |  ════════════════       |  ════════
              |  Jan-Jun | Jul-Dec      |
              |  CQR     |              |
              |  Calibr. |              |
```

- **Training**: 2018 (model training)
- **Calibration**: Jan-Jun 2019 (CQR calibration)
- **Test-Val**: Jul-Dec 2019 (unused, reserved)
- **Test**: 2020 (final evaluation with calibrated intervals)

## Metrics Impact

| Metric | Before CQR | After CQR | Interpretation |
|--------|------------|-----------|----------------|
| PICP   | 60-70%     | ~80%      | ✓ Proper coverage |
| MIW    | Lower      | Higher    | ✓ Necessary increase |
| Winkler| High       | Lower     | ✓ Net improvement |
| CRPS   | X          | Similar   | ~ Maintained |
| MAE/RMSE | X        | Same      | No change (point forecast) |

## Implementation Notes

1. **Only affects quantile models**: CQR is only applied to models with `quantile=True` (NHITS_Q, TIMESNET_Q, TFT_Q)

2. **Point estimates unchanged**: The median (p50) prediction is not modified, only the interval bounds

3. **Calibration set size**: Uses 6 months (Jan-Jun 2019) for stable calibration. Smaller sets may work but are less reliable.

4. **Coverage guarantee**: The theoretical guarantee holds under the exchangeability assumption (test data is similar to calibration data)

5. **Conservative correction**: Uses ceiling in quantile calculation to guarantee at least (1-α) coverage

## References

- Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression." Advances in Neural Information Processing Systems.
- Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."

## Example Output

```
======================================================================
CONFORMALIZED QUANTILE REGRESSION (CQR) CALIBRATION
======================================================================
Target Coverage: 80.0%
Calibration correction (s_hat): 0.0234

Calibration Set Performance:
  Coverage Before: 67.43%
  Coverage After:  80.12%
  Width Before:    0.0512
  Width After:     0.0980
  Width Increase:  91.34%
======================================================================

[NHITS_Q] Applying CQR calibration (s_hat=0.0234)...
```

## Troubleshooting

### Low PICP even after CQR
- Check if calibration set is representative
- Verify data split is correct
- Consider using more calibration data

### Very wide intervals after CQR
- Normal if base model was severely miscalibrated
- Indicates model needs retraining or better hyperparameters
- Consider HPO to improve base predictions

### No calibration predictions obtained
- Check data availability for calibration period (Jan-Jun 2019)
- Verify model training completed successfully
- Check file paths and timestamps

## Future Enhancements

Possible extensions:
1. **Adaptive α**: Learn optimal miscoverage level per model
2. **Localized CQR**: Different calibration for different prediction horizons
3. **Multi-step CQR**: Proper coverage for full 24-hour forecast
4. **CQR for CRPS**: Calibrate full predictive distribution
