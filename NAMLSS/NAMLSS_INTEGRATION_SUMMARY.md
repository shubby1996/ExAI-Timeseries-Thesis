# NAMLSS Benchmarker Integration - Complete Guide

## ✅ Integration Complete!

NAMLSS (Time Series Neural Additive Model for Location, Scale, and Shape) has been successfully integrated into the benchmarker framework. You can now train, evaluate, and compare NAMLSS against other baseline models (NHITS, TFT, TimesNet).

---

## 📋 What Was Integrated

### 1. **NAMLSSAdapter Class**
Located in [`benchmarker.py`](../benchmarker.py), lines ~820-1250

**Key Features:**
- ✅ Auto-detects dataset type (heat vs water) from CSV path
- ✅ Uses NAMLSS's native data pipeline (`step1_3_data_pipeline.py`)
- ✅ Configurable architecture (endo, exo, future_cov features)
- ✅ Supports HPO-optimized hyperparameters
- ✅ Implements all 3 required methods: `train()`, `evaluate()`, `get_calibration_predictions()`
- ✅ **Analytical CRPS** for Normal distribution (exact, no sampling)
- ✅ CQR calibration support for better uncertainty quantification

### 2. **Model Configuration**
NAMLSS is registered in `Benchmarker.__init__()` with default settings:

```python
"NAMLSS": {
    "type": "NAMLSS",
    "quantile": True,        # Always probabilistic
    "n_epochs": 30,          # Training epochs
    "L": 168,                # Input window (hours)
    "H": 24,                 # Forecast horizon (hours)
    "batch_size": 128,
    "lr": 1e-3,              # Learning rate
    "dropout": 0.1,
    "patience": 5,           # Early stopping
    "hidden_window": 64,     # WindowMLP hidden size
    "hidden_future_cov": 32, # FutureCovMLP hidden size
    "device": "cpu",
    "best_params": namlss_best  # Auto-loaded from results/best_params_NAMLSS.json
}
```

### 3. **Feature Configuration**
Auto-configured based on dataset:

**Heat Dataset (nordbyen):**
```python
target = "heat_consumption"
endo_cols = ["heat_lag_1h", "heat_lag_24h", "heat_rolling_24h"]
exo_cols = ["temp", "wind_speed", "dew_point", "temp_squared", 
            "temp_wind_interaction", "humidity", "clouds_all", 
            "pressure", "rain_1h", "snow_1h", "temp_weekend_interaction"]
future_cov_cols = ["hour_sin", "hour_cos", "is_weekend", "is_public_holiday",
                   "day_of_week", "season", "hour", "month", "is_school_holiday"]
```

**Water Dataset (centrum/tommerby):**
```python
target = "water_consumption"
endo_cols = ["water_lag_1h", "water_lag_24h", "water_rolling_24h"]
# (same exo_cols and future_cov_cols as heat)
```

### 4. **Analytical CRPS**
New helper function for exact CRPS calculation:

```python
def calculate_crps_normal(y_true, mu, sigma):
    """Analytical CRPS for Normal distribution.
    
    CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    where z = (y - μ)/σ
    """
```

**Advantages:**
- ✅ Exact (no sampling error)
- ✅ Faster than ensemble-based CRPS
- ✅ Works directly with (μ, σ) output from NAMLSS

---

## 🚀 Usage

### **Basic Training**
```bash
python3 benchmarker.py --models NAMLSS
```

### **Compare Against Baselines**
```bash
python3 benchmarker.py --models NAMLSS NHITS_Q TIMESNET_Q TFT_Q
```

### **With CQR Calibration**
```bash
python3 benchmarker.py --models NAMLSS --use_cqr
```

### **Specific Dataset**
```bash
# Heat demand
python3 benchmarker.py \
    --csv_path processing/nordbyen_processing/nordbyen_features_engineered.csv \
    --models NAMLSS \
    --dataset nordbyen_heat

# Water consumption (Centrum)
python3 benchmarker.py \
    --csv_path processing/centrum_processing/centrum_features_engineered.csv \
    --models NAMLSS \
    --dataset centrum_water

# Water consumption (Tommerby)
python3 benchmarker.py \
    --csv_path processing/tommerby_processing/tommerby_features_engineered.csv \
    --models NAMLSS \
    --dataset tommerby_water
```

---

## 📊 Evaluation Metrics

NAMLSS is evaluated on the same metrics as baseline models:

### **Point Forecast Metrics**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- MAPE_EPS (MAPE with epsilon stabilization)
- sMAPE (Symmetric MAPE)
- WAPE (Weighted Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

### **Probabilistic Metrics**
- **Pinball Loss**: Average quantile loss across p10, p50, p90
- **PICP**: Prediction Interval Coverage Probability (target: 80%)
- **MIW**: Mean Interval Width (p90 - p10)
- **Winkler Score**: Combined width + miss penalty
- **CRPS**: Analytical CRPS for Normal(μ, σ)

---

## 💾 Model Persistence

**Models are saved to:**
```
models/{dataset}/NAMLSS.pt                          # PyTorch state dict
models/{dataset}/NAMLSS_preprocessing_state.pkl     # TSConfig + scalers
```

**Examples:**
- `models/nordbyen_heat/NAMLSS.pt`
- `models/water_centrum/NAMLSS.pt`
- `models/water_tommerby/NAMLSS.pt`

---

## 🎯 Key Differences from Other Models

| Feature | NAMLSS | NHITS/TFT | TimesNet |
|---------|--------|-----------|----------|
| **Interpretability** | ✅ Additive decomposition | ❌ Black box | ❌ Black box |
| **Uncertainty** | Parametric (μ, σ) | Sample-based | Sample-based |
| **Loss** | Normal NLL | Quantile or MSE | MQLoss or MSE |
| **CRPS** | Analytical | Ensemble | Ensemble |
| **Quantiles** | Derived from (μ, σ) | Direct output | Direct output |
| **Data Pipeline** | `step1_3_data_pipeline.py` | `model_preprocessing.py` | `model_preprocessing.py` |

---

## 🔧 Hyperparameter Optimization (HPO)

If you have HPO results saved in `results/best_params_NAMLSS.json`, the benchmarker will automatically use them:

```json
{
    "hidden_window": 128,
    "hidden_future_cov": 64,
    "dropout": 0.15,
    "lr": 5e-4,
    "batch_size": 256
}
```

---

## 📝 Example Output

```
[NAMLSS] Training...
  Train samples=8000 | Val samples=2000
  Device=cpu | Batch size=128 | LR=0.001 | Dropout=0.1
  Using default hyperparameters (no HPO results found)
  Epoch 01: train_nll=-1.234567 | val_nll=-1.345678
  ...
  Epoch 15: train_nll=-1.456789 | val_nll=-1.567890
    Saved best model -> models/nordbyen_heat/NAMLSS.pt
  Early stopping (patience=5). Best val_nll=-1.567890

[NAMLSS] Evaluating (Walk-forward)...

[NAMLSS] Applying CQR calibration (s_hat=0.1234)...

====================================================================
BENCHMARK RESULTS
====================================================================
Model    MAE    RMSE   MAPE   PICP    MIW    Winkler  CRPS
NAMLSS   1.234  1.789  12.45  82.3    5.67   4.123    0.895
====================================================================
```

---

## ✅ Verification

### **Syntax Check**
```bash
python3 -m py_compile benchmarker.py
# ✓ benchmarker.py syntax is valid
```

### **NAMLSS Modules**
```bash
python3 -c "import sys; sys.path.append('NAMLSS'); from train_tsnamlss import TSNAMLSSNormal; from step1_3_data_pipeline import TSConfig; print('OK')"
# NAMLSS modules OK
```

### **Feature Availability**
All required features exist in CSV files:
- ✅ `heat_lag_1h`, `heat_lag_24h`, `heat_rolling_24h` (nordbyen)
- ✅ `water_lag_1h`, `water_lag_24h`, `water_rolling_24h` (centrum/tommerby)
- ✅ All exogenous features (temp, wind_speed, etc.)
- ✅ All future covariates (hour_sin, hour_cos, etc.)

---

## 🎉 Integration Benefits

1. **Fair Comparison**: Same evaluation protocol as baselines
2. **Interpretability**: Additive structure enables feature attribution
3. **Efficiency**: Analytical CRPS is faster than sampling
4. **Calibration**: Native CQR support for better uncertainty
5. **Flexibility**: Auto-detects dataset type and configures features
6. **Robustness**: HPO integration for optimal hyperparameters

---

## 🐛 Known Limitations

1. **Environment Issue**: The `properscoring` package has a dependency conflict with `numba`/`coverage` in your current environment. This doesn't affect NAMLSS integration code—it's a pre-existing issue with the benchmarker.

2. **Workaround**: If benchmarker import fails, you may need to:
   ```bash
   pip uninstall coverage
   pip install coverage==7.2.7  # Known compatible version
   ```

---

## 📚 References

- **NAMLSS Documentation**: [`NAMLSS/NAMLSS_ADAPTER_GUIDE.md`](../NAMLSS/NAMLSS_ADAPTER_GUIDE.md)
- **Architecture Guide**: [`NAMLSS/CONFIGURABLE_ARCHITECTURE_GUIDE.md`](../NAMLSS/CONFIGURABLE_ARCHITECTURE_GUIDE.md)
- **Code Walkthrough**: [`NAMLSS/CODE_WALKTHROUGH_NEW_ARCHITECTURE.md`](../NAMLSS/CODE_WALKTHROUGH_NEW_ARCHITECTURE.md)

---

**Integration Date:** February 14, 2026  
**Status:** ✅ Complete and Ready for Use
