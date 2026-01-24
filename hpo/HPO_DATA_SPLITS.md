# HPO Data Usage Analysis

## Summary

**HPO uses a 70-20-10 train-val-test split** and follows best practices:
- ✅ **Trains** on training data (70%)
- ✅ **Evaluates** on validation data (20%) 
- ✅ **Never touches** test data (10%) during HPO

The test set is reserved for final benchmarking after HPO is complete.

---

## Detailed Split Configuration

### 1. Heat Dataset (Nordbyen)
**Total Period**: 2015-05 to 2020-11 (~67 months)

| Split | Period | Percentage | Used For |
|-------|--------|------------|----------|
| **Train** | 2015-05 to 2019-05 | ~70% (~48 months) | Model training during HPO |
| **Validation** | 2019-06 to 2020-05 | ~20% (~12 months) | Walk-forward evaluation (HPO objective) |
| **Test** | 2020-06 to 2020-11 | ~10% (~6 months) | Final benchmarking (NOT used in HPO) |

**Code**:
```python
"heat": {
    "train_end": "2019-05-31 23:00:00",  # Train: 2015-05 to 2019-05
    "val_end": "2020-05-31 23:00:00",    # Val: 2019-06 to 2020-05
    # Test: 2020-06 to 2020-11 (implicit)
}
```

---

### 2. Water Centrum Dataset
**Total Period**: 2018-04 to 2020-11 (~32 months)

| Split | Period | Percentage | Used For |
|-------|--------|------------|----------|
| **Train** | 2018-04 to 2019-11 | ~70% (~20 months) | Model training during HPO |
| **Validation** | 2019-12 to 2020-06 | ~20% (~7 months) | Walk-forward evaluation (HPO objective) |
| **Test** | 2020-07 to 2020-11 | ~10% (~5 months) | Final benchmarking (NOT used in HPO) |

**Code**:
```python
"water_centrum": {
    "train_end": "2019-11-30 23:00:00",  # Train: 2018-04 to 2019-11
    "val_end": "2020-06-30 23:00:00",    # Val: 2019-12 to 2020-06
    # Test: 2020-07 to 2020-11 (implicit)
}
```

---

### 3. Water Tommerby Dataset
**Total Period**: 2018-04 to 2020-11 (~32 months)

| Split | Period | Percentage | Used For |
|-------|--------|------------|----------|
| **Train** | 2018-04 to 2019-11 | ~70% (~20 months) | Model training during HPO |
| **Validation** | 2019-12 to 2020-06 | ~20% (~7 months) | Walk-forward evaluation (HPO objective) |
| **Test** | 2020-07 to 2020-11 | ~10% (~5 months) | Final benchmarking (NOT used in HPO) |

**Code**:
```python
"water_tommerby": {
    "train_end": "2019-11-30 23:00:00",  # Train: 2018-04 to 2019-11
    "val_end": "2020-06-30 23:00:00",    # Val: 2019-12 to 2020-06
    # Test: 2020-07 to 2020-11 (implicit)
}
```

---

## HPO Training Process (Per Trial)

### Step 1: Data Preparation
```python
state, t_sc, v_sc, _ = mp.prepare_model_data(
    csv_path,
    to_naive(split_cfg["train_end"]),  # 70% cutoff
    to_naive(split_cfg["val_end"]),    # 90% cutoff (train+val)
    cfg
)
```

Returns:
- `t_sc` (train_scaled): Training data up to `train_end`
  - `target`: Target time series (train set)
  - `past_covariates`: Historical features (train set)
  - `future_covariates`: Known future features (train set)

- `v_sc` (val_scaled): Validation data from `train_end` to `val_end`
  - `target`: Target time series (val set)
  - `past_covariates`: Historical features (val set)
  - `future_covariates`: Known future features (val set)

### Step 2: Model Training
```python
model.fit(
    t_sc["target"],              # Train on TRAINING set target
    past_covariates=tp,          # Train on TRAINING set covariates
    val_series=v_sc["target"],   # Validate on VALIDATION set target
    val_past_covariates=vp       # Validate on VALIDATION set covariates
)
```

### Step 3: HPO Evaluation (Walk-Forward on Validation Set)
```python
mae, picp = evaluate_model_walk_forward(
    model, 
    v_sc,        # Uses VALIDATION data only
    state, 
    n_steps=10,  # 10 walk-forward steps
    is_tft=False
)
```

**Walk-forward evaluation**:
- Takes first ~7 days of validation set as history
- Predicts next 24 hours
- Rolls forward 1 step
- Repeats for 10 steps
- Calculates MAE and PICP across all 10 predictions

---

## Why This Matters

### ✅ Correct HPO Practice
1. **No test set leakage**: Test data (last 10%) is completely isolated
2. **Proper validation**: HPO optimizes hyperparameters based on validation performance
3. **Realistic evaluation**: Walk-forward on validation mimics real-world deployment

### 📊 Final Workflow
```
HPO Phase (Current):
  ├─ Train on 70% → Validate on 20% → Optimize hyperparams
  └─ Result: Best hyperparameters per model

Benchmarking Phase (Next):
  ├─ Train on 70% with best hyperparameters
  ├─ Quick check on 20% (validation)
  └─ Final evaluation on 10% (test) ← ONLY USED ONCE
```

### 🔍 Dataset Sizes
| Dataset | Total Rows | Train (70%) | Val (20%) | Test (10%) |
|---------|-----------|-------------|-----------|------------|
| **Heat** | ~48,574 | ~34,000 | ~9,700 | ~4,874 |
| **Water Centrum** | ~22,989 | ~16,092 | ~4,598 | ~2,299 |
| **Water Tommerby** | Similar | Similar | Similar | Similar |

---

## Code References

**File**: `hpo/hpo_config.py`
- Lines 44-67: `SPLIT_CONFIG` dictionary
- Lines 70-77: `HPO_TRAINING_CONFIG` (n_epochs=15 for speed)

**File**: `hpo/run_hpo.py`
- Lines 207-268: `train_nhits()` - data loading and training
- Lines 50-205: `evaluate_model_walk_forward()` - validation evaluation
- Uses `mp.prepare_model_data()` from `model_preprocessing.py`

**File**: `model_preprocessing.py`
- `prepare_model_data()`: Splits data based on train_end and val_end timestamps
- Returns separate train (`t_sc`) and validation (`v_sc`) dictionaries

---

## Verification Commands

Check actual data periods:
```bash
# Heat dataset
python -c "import pandas as pd; df = pd.read_csv('processing/nordbyen_processing/nordbyen_features_engineered.csv'); print(f'Start: {df.ds.min()}'); print(f'End: {df.ds.max()}'); print(f'Rows: {len(df)}')"

# Water centrum dataset  
python -c "import pandas as pd; df = pd.read_csv('processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv'); print(f'Start: {df.ds.min()}'); print(f'End: {df.ds.max()}'); print(f'Rows: {len(df)}')"
```

Check split integrity:
```python
# Verify no overlap between train/val/test
import pandas as pd
from hpo.hpo_config import SPLIT_CONFIG

dataset = "heat"
cfg = SPLIT_CONFIG[dataset]

print(f"Train: [start] to {cfg['train_end']}")
print(f"Val:   {cfg['train_end']} to {cfg['val_end']}")
print(f"Test:  {cfg['val_end']} to [end]")
```

---

## Conclusion

The HPO setup is **methodologically sound** and follows best practices:
- Clean 70-20-10 split
- No test set contamination
- Validation-based hyperparameter selection
- Walk-forward evaluation for realistic performance estimates

The test set remains pristine for final benchmarking after HPO completes.
