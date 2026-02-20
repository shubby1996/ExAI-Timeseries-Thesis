# NAMLSS Adapter Guide

## Overview

The `NAMLSSAdapter` integrates the TSNAMLSS (Time Series Neural Additive Model for Location, Scale, and Shape) model into the benchmarker framework. This allows NAMLSS to be evaluated alongside other baseline models (N-HiTS, TFT, TimesNet) using the same infrastructure.

## Architecture

NAMLSS is a probabilistic additive model that outputs both mean (μ) and uncertainty (σ) predictions. The adapter:
- Uses **Normal NLL** loss for training
- Generates **quantile predictions** (p10, p50, p90) from the learned distribution
- Supports **walk-forward evaluation** for fair comparison with baselines
- Integrates with **Conformal Quantile Regression (CQR)** for calibrated uncertainty

## Key Implementation Details

### 1. Data Pipeline
Unlike Darts/NeuralForecast adapters that use `model_preprocessing.py`, NAMLSS uses:
- `step1_3_data_pipeline.py` - Same pipeline as standalone NAMLSS training
- `TSConfig` - Configures feature columns (target, endo, exo, future_cov)
- `WindowDataset` - Sliding window data loader

### 2. Model Properties (via `__init__`)
```python
config = {
    "type": "NAMLSS",
    "quantile": True,          # Always True (probabilistic model)
    "n_epochs": 30,            # Training epochs
    "L": 168,                  # Input window length (hours)
    "H": 24,                   # Forecast horizon (hours)
    "batch_size": 128,
    "lr": 1e-3,                # Learning rate
    "dropout": 0.1,
    "patience": 5,             # Early stopping patience
    "hidden_window": 64,       # WindowMLP hidden size
    "hidden_future_cov": 32,   # FutureCovMLP hidden size
    "device": "cpu",           # "cpu" or "cuda"
    "best_params": None        # Optional: HPO-optimized hyperparameters
}
```

### 3. Required Methods

#### `train(csv_path, train_end_str, val_end_str)`
**Implementation:**
1. Loads data using `load_and_prepare()` from `step1_3_data_pipeline.py`
2. Chronologically splits into train/val using date strings
3. Fits `StandardScaler` on train set only
4. Initializes `TSNAMLSSNormal` model
5. Trains with early stopping based on validation NLL
6. Saves best model checkpoint with scalers to `models/{name}.pt`

**Saved Checkpoint Contains:**
- `model_state`: PyTorch state dict
- `cfg`: TSConfig parameters (target, endo_cols, exo_cols, future_cov_cols, L, H)
- `scalers`: Dict of StandardScaler parameters (mean, std) for each feature

#### `evaluate(csv_path, test_start_str, n_predictions) -> (metrics, predictions_df)`
**Implementation:**
1. Loads trained model checkpoint
2. Performs **walk-forward evaluation** for `n_predictions` days
3. For each 24-hour prediction window:
   - Extracts past 168 hours (L=168) of history
   - Extracts 24 hours (H=24) of future covariates
   - Predicts μ (mean) and σ (scale) in scaled space
   - Inverse transforms to original units
   - Generates quantiles: p10 = μ - 1.282σ, p90 = μ + 1.282σ
4. Applies CQR calibration if `self.cqr_s_hat` is set
5. Calculates all required metrics

**Quantile Generation:**
Unlike models that directly output quantiles, NAMLSS outputs (μ, σ) from a Normal distribution:
```python
p10 = mu - 1.282 * sigma  # 10th percentile (z=-1.282)
p50 = mu                  # 50th percentile (median)
p90 = mu + 1.282 * sigma  # 90th percentile (z=+1.282)
```

**Returns:**
```python
metrics = {
    "MAE", "RMSE", "MAPE", "MAPE_EPS", "sMAPE", "WAPE", "MASE",
    "Pinball",  # Quantile loss
    "PICP",     # Prediction Interval Coverage Probability
    "MIW",      # Mean Interval Width
    "Winkler",  # Interval score
    "CRPS"      # Continuous Ranked Probability Score
}

predictions_df = pd.DataFrame({
    "timestamp": [...],
    "actual": [...],
    "p10": [...],
    "p50": [...],
    "p90": [...]
})
```

#### `get_calibration_predictions(csv_path, cal_start_str, cal_end_str) -> predictions_df`
**Implementation:**
- Same as `evaluate()` but runs on calibration period
- Used for CQR to compute conformity scores
- Returns predictions DataFrame with same structure

### 4. Integration with Benchmarker

The adapter is registered in `Benchmarker._get_model_configs()`:

```python
self.configs = {
    ...
    "NAMLSS": {
        "type": "NAMLSS", 
        "quantile": True, 
        "n_epochs": 30, 
        "best_params": namlss_best,
        "L": 168, "H": 24, 
        "batch_size": 128, 
        "lr": 1e-3, 
        "dropout": 0.1, 
        "patience": 5,
        "hidden_window": 64, 
        "hidden_future_cov": 32, 
        "device": "cpu"
    },
    ...
}
```

And instantiated in `run_benchmark()`:
```python
if cfg["type"].upper() == "NAMLSS":
    adapter = NAMLSSAdapter(mk, cfg, models_folder=self.dataset_models_folder)
```

## Usage

### Training NAMLSS via Benchmarker

```bash
python benchmarker.py \
    --csv_path nordbyen_features_engineered.csv \
    --models NAMLSS \
    --dataset nordbyen_heat
```

### With CQR Calibration

```bash
python benchmarker.py \
    --csv_path nordbyen_features_engineered.csv \
    --models NAMLSS \
    --use_cqr \
    --alpha 0.2 \
    --dataset nordbyen_heat
```

### Comparing Multiple Models

```bash
python benchmarker.py \
    --csv_path centrum_features_engineered.csv \
    --models NAMLSS NHITS_Q TFT_Q TIMESNET_Q \
    --dataset centrum_water
```

## Hyperparameter Optimization (HPO)

To use HPO-optimized hyperparameters:

1. Run HPO script to generate `results/best_params_NAMLSS.json`
2. Benchmarker automatically loads and applies these parameters
3. Example `best_params_NAMLSS.json`:

```json
{
    "hidden_window": 128,
    "hidden_future_cov": 64,
    "dropout": 0.15,
    "lr": 5e-4,
    "batch_size": 256
}
```

If HPO params exist, they override the defaults in the config.

## Model Persistence

**Trained models are saved to:**
```
models/{dataset}/NAMLSS.pt
```

Examples:
- `models/nordbyen_heat/NAMLSS.pt` - Heat demand model
- `models/water_centrum/NAMLSS.pt` - Centrum water model
- `models/water_tommerby/NAMLSS.pt` - Tommerby water model

**Checkpoint Structure:**
```python
{
    "model_state": {...},          # PyTorch state_dict
    "cfg": {                       # TSConfig parameters
        "L": 168, 
        "H": 24,
        "target": "heat_consumption",
        "endo_cols": ["heat_lag_1h", "heat_lag_24h"],
        "exo_cols": ["temp", "wind_speed"],
        "future_cov_cols": ["hour_sin", "hour_cos"],
        ...
    },
    "scalers": {                   # StandardScaler parameters
        "heat_consumption": {"mean": 5.2, "std": 3.1},
        "temp": {"mean": 8.5, "std": 7.2},
        ...
    }
}
```

## Differences from Other Adapters

| Feature | NAMLSS | Darts (NHITS/TFT) | NeuralForecast (TimesNet) |
|---------|--------|-------------------|---------------------------|
| **Data Pipeline** | `step1_3_data_pipeline.py` | `model_preprocessing.py` | `model_preprocessing.py` |
| **Uncertainty** | Parametric (μ, σ) | Sample-based quantiles | Sample-based quantiles |
| **Loss Function** | Normal NLL | Quantile loss or MSE | MQLoss or MSE |
| **Framework** | PyTorch | Darts (PyTorch Lightning) | NeuralForecast (PyTorch Lightning) |
| **Quantile Generation** | Derived from (μ, σ) | Direct output | Direct output |

## Feature Configuration

NAMLSS uses `TSConfig` from `step1_3_data_pipeline.py`:

**Default Config (Heat Demand):**
```python
target = "heat_consumption"
endo_cols = ["heat_lag_1h", "heat_lag_24h"]  # AR features
exo_cols = ["temp"]                          # Weather/exogenous
future_cov_cols = ["hour_sin", "hour_cos"]   # Calendar/future-known
```

**Water Consumption Config:**
```python
target = "water_consumption"
endo_cols = ["water_lag_1h", "water_lag_24h"]
exo_cols = ["temp"]
future_cov_cols = ["hour_sin", "hour_cos"]
```

To customize, modify `TSConfig` defaults in `step1_3_data_pipeline.py`.

## Evaluation Metrics

NAMLSS is evaluated on the same metrics as baseline models:

**Point Forecast Metrics:**
- MAE, RMSE, MAPE, sMAPE, WAPE, MASE

**Probabilistic Metrics:**
- **Pinball Loss**: Average quantile loss across p10, p50, p90
- **PICP**: Prediction Interval Coverage Probability (target: 80%)
- **MIW**: Mean Interval Width (p90 - p10)
- **Winkler Score**: Combined width + miss penalty
- **CRPS**: Continuous Ranked Probability Score (samples from Normal(μ, σ))

## Advantages of NAMLSS Adapter

1. **Interpretability**: Additive structure allows decomposing predictions into feature contributions
2. **Calibration**: Parametric uncertainty (σ) can be better calibrated than sample-based quantiles
3. **Efficiency**: Direct (μ, σ) output is faster than generating 100+ samples
4. **Integration**: Seamlessly works with existing benchmarker infrastructure (CQR, metrics, comparison)

## Example Output

```
[NAMLSS] Training...
  Train samples=8000 | Val samples=2000
  Device=cpu | Batch size=128 | LR=0.001 | Dropout=0.1
  Epoch 01: train_nll=-1.234567 | val_nll=-1.345678
  ...
  Epoch 15: train_nll=-1.456789 | val_nll=-1.567890
    Saved best model -> models/nordbyen_heat/NAMLSS.pt
  Early stopping (patience=5). Best val_nll=-1.567890

[NAMLSS] Evaluating (Walk-forward)...
  [CRPS Debug] y_true length: 1200
  [CRPS Debug] samples is list with length: 1200

┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Model  ┃ MAE     ┃ RMSE    ┃ MAPE    ┃ PICP (%)┃ MIW     ┃ CRPS     ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ NAMLSS │ 1.234   │ 1.789   │ 12.45   │ 82.3    │ 5.67    │ 0.895    │
└────────┴─────────┴─────────┴─────────┴──────────┴─────────┴──────────┘
```

## Troubleshooting

**Issue:** `FileNotFoundError: model checkpoint not found`
- **Solution:** Run training first or check `models_folder` path

**Issue:** `RuntimeError: No predictions generated`
- **Solution:** Ensure sufficient history (≥168 hours) before test_start

**Issue:** `KeyError: missing feature column`
- **Solution:** Verify CSV has all required features from `TSConfig`

**Issue:** CUDA OOM
- **Solution:** Reduce `batch_size` or use `device="cpu"`

## References

- **NAMLSS Training**: `train_tsnamlss.py`
- **NAMLSS Evaluation**: `eval_tsnamlss.py`
- **Data Pipeline**: `step1_3_data_pipeline.py`
- **Benchmarker**: `benchmarker.py`

---

**Created:** 2025-01-14  
**Author:** NAMLSS Integration Team
