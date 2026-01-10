# Conformalized Quantile Regression (CQR) - Quick Start

## What is CQR?

CQR is a **post-hoc calibration technique** that fixes undercalibrated prediction intervals from quantile forecasting models. It ensures your 80% prediction intervals actually contain 80% of true values.

## The Problem

Your NHITS/TimesNet/TFT models predict quantiles (p10, p50, p90), but:
- **Expected**: 80% of actuals fall within [p10, p90]
- **Reality**: Only 60-70% coverage (intervals too narrow)
- **Impact**: Unreliable uncertainty estimates

## The Solution

CQR adjusts intervals on a calibration set to achieve proper coverage:
```python
# 1. Get predictions on calibration data
s_hat = cqr.calibrate_intervals(y_cal_true, y_cal_low, y_cal_high, alpha=0.2)

# 2. Apply correction to test predictions
y_test_low_cal, y_test_high_cal = cqr.apply_cqr_correction(
    y_test_low, y_test_high, s_hat
)
```

## How to Use

### Quick Start (Automatic)

Run benchmarker with CQR enabled (default):
```bash
python benchmarker.py NHITS_Q TIMESNET_Q TFT_Q
```

Disable CQR (to see baseline):
```bash
python benchmarker.py --no-cqr NHITS_Q TIMESNET_Q TFT_Q
```

### Water Tommerby Benchmark

```bash
cd water_tommerby_benchmark/scripts
python run_benchmarker_water.py --models NHITS_Q TIMESNET_Q TFT_Q
```

### Python API

```python
from benchmarker import Benchmarker

benchmarker = Benchmarker(csv_path, models, dataset="Water (Tommerby)")
benchmarker.run(use_cqr=True, alpha=0.2)  # 80% target coverage
```

## Expected Improvements

| Metric | Before CQR | After CQR | Why? |
|--------|------------|-----------|------|
| **PICP** | 60-70% | ~80% | ✓ Proper coverage achieved |
| **MIW** | Narrow | Wider | ✓ Necessary for proper coverage |
| **Winkler** | High | Lower | ✓ Fewer misses → net improvement |

## Files

- **`conformal_calibration.py`**: Core CQR implementation
- **`example_cqr_usage.py`**: Standalone example with visualization
- **`test_cqr.py`**: Unit tests
- **`docs/CQR_CALIBRATION.md`**: Full documentation

## Testing

Run unit tests:
```bash
python test_cqr.py
```

Run practical example:
```bash
python example_cqr_usage.py
```

## Key Points

1. **Only affects quantile models**: MSE models unchanged
2. **Point forecasts unchanged**: Only interval bounds adjusted
3. **Calibration cost**: Uses 6 months of validation data (Jan-Jun 2019)
4. **Coverage guarantee**: Achieves ≥ (1-α) coverage on test set

## Reference

Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression." NeurIPS.

## Questions?

See full documentation: [`docs/CQR_CALIBRATION.md`](docs/CQR_CALIBRATION.md)
