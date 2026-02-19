# NAMLSS Benchmark Results - 23 Covariates Configuration

**Documentation Date:** February 18, 2026  
**Configuration:** Full feature set (23 covariates)  
**Training Setup:** 30 epochs, early stopping (patience=5), batch_size=128, lr=0.001

---

## Feature Configuration (23 Covariates)

### Endogenous Features (3)
Autoregressive features from the target variable:
- `{target}_lag_1h` - 1-hour lag
- `{target}_lag_24h` - 24-hour lag
- `{target}_rolling_24h` - 24-hour rolling average

### Exogenous Features (11)
External weather and interaction features:
- `temp` - Temperature
- `wind_speed` - Wind speed
- `dew_point` - Dew point
- `temp_squared` - Temperature squared
- `temp_wind_interaction` - Temperature × Wind interaction
- `humidity` - Humidity
- `clouds_all` - Cloud coverage
- `pressure` - Atmospheric pressure
- `rain_1h` - Hourly rainfall
- `snow_1h` - Hourly snowfall
- `temp_weekend_interaction` - Temperature × Weekend interaction

### Future Covariates (9)
Time-based features known in advance:
- `hour_sin` - Hour sine encoding
- `hour_cos` - Hour cosine encoding
- `is_weekend` - Weekend indicator
- `is_public_holiday` - Public holiday indicator
- `day_of_week` - Day of week (0-6)
- `season` - Season encoding
- `hour` - Hour of day (0-23)
- `month` - Month (1-12)
- `is_school_holiday` - School holiday indicator

**Total Feature Streams:** 1 (target) + 3 (endo) + 11 (exo) + 9 (future) = **24 neural networks**

---

## Dataset 1: Nordbyen Heat Consumption

**Training Date:** 2026-02-15 21:47:42  
**Dataset:** `processing/nordbyen_processing/nordbyen_features_engineered.csv`  
**Target Variable:** `heat_consumption`  
**Benchmark Job ID:** 1531802

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 0.1503 | Mean Absolute Error |
| **RMSE** | 0.2101 | Root Mean Squared Error |
| **MAPE** | 8.96% | Mean Absolute Percentage Error |
| **sMAPE** | 9.12% | Symmetric MAPE |
| **WAPE** | 8.02% | Weighted Absolute Percentage Error |
| **MASE** | 0.8748 | Mean Absolute Scaled Error |

### Probabilistic Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Pinball Loss** | 0.0490 | Quantile loss |
| **PICP** | 70.58% | Prediction Interval Coverage Probability (target: 80%) |
| **MIW** | 0.3938 | Mean Interval Width |
| **Winkler Score** | 0.7192 | Combined coverage + width penalty |
| **CRPS** | 0.1090 | Continuous Ranked Probability Score |

### Model Files
- Model checkpoint: `models/nordbyen_heat/NAMLSS.pt`
- Preprocessing state: `models/nordbyen_heat/NAMLSS_preprocessing_state.pkl`
- Predictions: `nordbyen_heat_benchmark/results/NAMLSS_predictions_1531802.csv`
- Benchmark results: `nordbyen_heat_benchmark/results/benchmark_results_20260215_214922_nordbyen_heat_1531802.csv`

### Training Configuration
- **Epochs:** 30
- **Batch Size:** 128
- **Learning Rate:** 0.001
- **Device:** CUDA (GPU)
- **Dropout:** 0.1
- **Hidden (window):** 64
- **Hidden (future_cov):** 32
- **L (history):** 168 hours (7 days)
- **H (forecast):** 24 hours (1 day)

---

## Dataset 2: Water Centrum Consumption

**Training Date:** 2026-02-15 22:02:57  
**Dataset:** `processing/centrum_processing/centrum_features_engineered.csv`  
**Target Variable:** `water_consumption`  
**Benchmark Job ID:** 1531806

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 0.0045 | Mean Absolute Error |
| **RMSE** | 0.0055 | Root Mean Squared Error |
| **MAPE** | 35.07% | Mean Absolute Percentage Error |
| **sMAPE** | 26.23% | Symmetric MAPE |
| **WAPE** | 19.17% | Weighted Absolute Percentage Error |
| **MASE** | 0.2304 | Mean Absolute Scaled Error |

### Probabilistic Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Pinball Loss** | 0.0014 | Quantile loss |
| **PICP** | 66.16% | Prediction Interval Coverage Probability (target: 80%) |
| **MIW** | 0.0134 | Mean Interval Width |
| **Winkler Score** | 0.0202 | Combined coverage + width penalty |
| **CRPS** | 0.0032 | Continuous Ranked Probability Score |

### Model Files
- Model checkpoint: `models/water_centrum/NAMLSS.pt`
- Preprocessing state: `models/water_centrum/NAMLSS_preprocessing_state.pkl`
- Benchmark results: `water_centrum_benchmark/results/benchmark_results_20260215_220427_Water_Centrum_1531806.csv`

### Training Configuration
- **Epochs:** 30
- **Batch Size:** 128
- **Learning Rate:** 0.001
- **Device:** CUDA (GPU)
- **Dropout:** 0.1
- **Hidden (window):** 64
- **Hidden (future_cov):** 32
- **L (history):** 168 hours (7 days)
- **H (forecast):** 24 hours (1 day)

---

## Dataset 3: Water Tommerby Consumption

**Status:** ⚠️ **EVALUATION PENDING**

**Training Date:** 2026-02-15 22:18 (model file exists)  
**Dataset:** `processing/tommerby_processing/tommerby_features_engineered.csv`  
**Target Variable:** `water_consumption`

### Model Files
- Model checkpoint: `models/water_tommerby/NAMLSS.pt` ✓
- Benchmark results: **NOT FOUND** - Needs to be run

### Notes
The model has been trained but benchmark evaluation has not been completed yet. Need to run benchmarker for this dataset to generate metrics.

---

## Observations & Analysis

### Heat Consumption (Nordbyen)
- **Strong Performance:** MASE = 0.8748 (below 1.0 indicates better than naive baseline)
- **Point Forecast:** MAE = 0.15 MWh, RMSE = 0.21 MWh
- **Calibration Issue:** PICP = 70.58% (target is 80%) - prediction intervals are too narrow
- **MAPE:** ~9% error, which is reasonable for energy forecasting

### Water Consumption (Centrum)
- **Excellent Point Forecast:** MASE = 0.2304 (substantially better than baseline)
- **Small Absolute Errors:** MAE = 0.0045 m³/h, RMSE = 0.0055 m³/h
- **High Relative Error:** MAPE = 35% (likely due to very small consumption values)
- **Calibration Issue:** PICP = 66.16% (worse than heat) - intervals too narrow
- **Best CRPS:** 0.0032 (excellent probabilistic forecast quality)

### Cross-Dataset Comparison
1. **MASE Comparison:**
   - Water Centrum (0.2304) outperforms Heat (0.8748) relative to their baselines
   - Suggests NAMLSS particularly effective for water consumption patterns

2. **Calibration Problems:**
   - Both datasets show under-coverage (PICP < 80%)
   - Heat: 70.58% | Water: 66.16%
   - **Potential causes with 23 covariates:**
     - Model overfitting to training data (too many features)
     - Prediction intervals (μ ± 1.282σ) too narrow
     - Feature redundancy reducing uncertainty estimation quality

3. **MAPE vs WAPE:**
   - Water shows large MAPE (35%) but smaller WAPE (19%)
   - Indicates occasional very small consumption values inflate percentage errors
   - Heat shows consistent errors (MAPE ≈ WAPE ≈ 9%)

---

## Known Issues with 23 Covariates

### 1. Poor Interval Coverage (PICP)
- **Target:** 80% coverage
- **Actual:** 66-71% coverage
- **Impact:** Prediction intervals are not reliable for uncertainty quantification

### 2. Potential Overfitting
With 24 separate neural networks (1 per feature stream), the model may be:
- Learning noise in training data
- Producing overconfident predictions (narrow σ)
- Suffering from feature redundancy (correlated features)

### 3. Redundant Features
Likely redundant pairs in current configuration:
- `hour` ↔ `hour_sin/hour_cos` (cyclical encoding + raw value)
- `temp` ↔ `temp_squared` (polynomial feature)
- `temp` ↔ `temp_wind_interaction` ↔ `temp_weekend_interaction`
- Multiple lagged features may overlap in information

### 4. Computational Cost
- 24 neural networks → longer training time
- More parameters → higher memory requirements
- Diminishing returns from additional features

---

## Recommendation: Reduce to 12 Covariates

Based on prior experiments showing better results with fewer features, the next iteration should use:

### Proposed Reduced Configuration (12 covariates):

**Endogenous (2):**
- `{target}_lag_24h` - Daily seasonality
- `{target}_rolling_24h` - Smoothed recent trend

**Exogenous (6):**
- `temp` - Primary driver
- `wind_speed` - Weather impact
- `humidity` - Additional weather
- `temp_squared` - Non-linear temperature effect
- `is_weekend` - Weekly pattern (moved from future covs)
- `is_public_holiday` - Special days (moved from future covs)

**Future Covariates (4):**
- `hour_sin` - Time of day (cyclical)
- `hour_cos` - Time of day (cyclical)
- `day_of_week` - Day pattern
- `season` - Seasonal effects

**Total:** 1 (target) + 2 (endo) + 6 (exo) + 4 (future) = **13 feature streams**

### Expected Benefits:
1. **Better calibration:** Fewer parameters → less overfitting → better σ estimates → improved PICP
2. **Faster training:** 13 networks instead of 24
3. **Simpler interpretation:** Core features easier to analyze
4. **Maintained performance:** Remove redundant features without losing predictive power

---

## Next Steps

### Action Items:
1. ✅ **Document current results** (this file)
2. ⏳ **Update covariate configuration** in code:
   - Modify `step1_3_data_pipeline.py` TSConfig defaults
   - Modify `benchmarker.py` NAMLSSAdapter._get_tsconfig()
3. ⏳ **Re-train NAMLSS with 12 covariates:**
   - Heat: `run_namlss_myenv.sh --dataset nordbyen_heat --n_epochs 30 --device cuda`
   - Water Centrum: `run_namlss_myenv.sh --dataset water_centrum --n_epochs 30 --device cuda`
   - Water Tommerby: `run_namlss_myenv.sh --dataset water_tommerby --n_epochs 30 --device cuda`
4. ⏳ **Compare results:**
   - Document new metrics in `NAMLSS_RESULTS_12_COVARIATES.md`
   - Analyze PICP improvement
   - Compare CRPS, MAE, RMSE between configurations

### Hypothesis to Test:
**"Reducing from 23 to 12 covariates will improve PICP from 66-71% to 75-82% while maintaining or improving point forecast accuracy (MAE, RMSE, CRPS)."**

---

## File Structure

```
NAMLSS/
├── NAMLSS_RESULTS_23_COVARIATES.md  ← This file
├── NAMLSS_RESULTS_12_COVARIATES.md  ← To be created after re-training
├── step1_3_data_pipeline.py         ← Update TSConfig here
├── run_namlss_local.py
├── run_namlss_myenv.sh
├── namlss_benchmark.ipynb
├── namlss_water_centrum_benchmark.ipynb
└── namlss_water_tommerby_benchmark.ipynb

models/
├── nordbyen_heat/NAMLSS.pt          ← 23-covariate version
├── water_centrum/NAMLSS.pt          ← 23-covariate version
└── water_tommerby/NAMLSS.pt         ← 23-covariate version (not benchmarked)
```

---

**End of Documentation**
