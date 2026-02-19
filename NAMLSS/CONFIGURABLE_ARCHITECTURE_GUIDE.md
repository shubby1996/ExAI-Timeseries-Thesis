# Configurable Architecture Guide

Complete guide to using the new list-based configurable architecture for TSNAMLSSNormal time series forecasting.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Dataset Format](#dataset-format)
4. [Model Architecture](#model-architecture)
5. [Usage Examples](#usage-examples)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Default Configuration (Heat Demand Forecasting)

```bash
# 1. Prepare data
python3 step1_3_data_pipeline.py \
    --csv_path nordbyen_features_engineered.csv \
    --L 168 --H 24

# 2. Train model
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 50 \
    --save_path my_model.pt

# 3. Evaluate
python3 eval_tsnamlss.py \
    --ckpt my_model.pt \
    --csv_path nordbyen_features_engineered.csv

# 4. Interpret
python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model.pt \
    --do_effects
```

---

## Configuration

### TSConfig Structure

Edit in [step1_3_data_pipeline.py](step1_3_data_pipeline.py#L37):

```python
@dataclass
class TSConfig:
    # Window sizes
    L: int = 168        # History length (hours)
    H: int = 24         # Forecast horizon (hours)
    
    # Columns
    timestamp_col: str = "timestamp"
    target: str = None               # Singular target column to forecast
    endo_cols: list = None           # Optional AR features (lagged/transformed target)
    exo_cols: list = None            # Exogenous (historical feature) columns
    future_cov_cols: list = None     # Future covariate columns
    
    # Data settings
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    reindex_hourly: bool = True

    def __post_init__(self):
        # Defaults
        if self.target is None:
            self.target = "heat_consumption"
        if self.endo_cols is None:
            self.endo_cols = []  # No AR features by default
        if self.exo_cols is None:
            self.exo_cols = ["temp", "wind_speed", "dew_point", "temp_squared"]
        if self.future_cov_cols is None:
            self.future_cov_cols = ["hour_sin", "hour_cos"]
```

### Column Types Explained

| Type | Description | Examples | Used For |
|------|-------------|----------|----------|
| **target** | Singular target column | `"heat_consumption"` | What you're predicting |
| **endo_cols** | Optional AR features from target | `["heat_lag_1", "heat_roll_24"]` | Extra autoregressive signal |
| **exo_cols** | Historical features (past values only) | `["temp", "wind_speed"]` | External influences |
| **future_cov_cols** | Known future values | `["hour_sin", "hour_cos", "holiday"]` | Time features, calendars |

---

## Dataset Format

### Input CSV Requirements

Your CSV must have:
- A timestamp column (default: `"timestamp"`)
- The `target` column
- All columns specified in `endo_cols`, `exo_cols`
- Optionally `future_cov_cols` (auto-generated if time-based)

Example:
```csv
timestamp,heat_consumption,temp,wind_speed,dew_point,temp_squared
2020-01-01 00:00:00,45.2,5.3,3.1,2.1,28.09
2020-01-01 01:00:00,43.8,5.1,3.4,2.0,26.01
...
```

### WindowDataset Output

Each sample from `WindowDataset.__getitem__()`:

```python
{
    # Target
    'target_hist': Tensor[L],    # Historical target values
    'target_future': Tensor[H],  # Ground truth targets
    
    # Optional AR features (1 per endo_col)
    'heat_lag_1_hist': Tensor[L],
    'heat_roll_24_hist': Tensor[L],
    
    # Exogenous columns (1 per exo_col)
    'temp': Tensor[L],           # Historical temperature
    'wind_speed': Tensor[L],     # Historical wind speed
    'dew_point': Tensor[L],      # Historical dew point
    'temp_squared': Tensor[L],   # Historical temp^2
    
    # Future covariates (stacked)
    'future_cov': Tensor[H, D]   # D = len(future_cov_cols)
}
```

**Key Pattern**: Target keys are fixed (`target_hist`, `target_future`); AR features use `{endo_col}_hist`; exogenous use `{exo_col}`.

**Note**: `future_cov` is stacked in the dataset but split into one stream per feature inside the model.

---

## Model Architecture

### TSNAMLSSNormal Structure

```python
TSNAMLSSNormal(
    L=168,                                    # History window
    H=24,                                     # Forecast horizon
    target="heat_consumption",               # Singular target
    endo_cols=["heat_lag_1", "heat_roll_24"], # Optional AR features
    exo_cols=["temp", "wind_speed", ...],     # Exogenous inputs
    future_cov_cols=["hour_sin", "hour_cos"], # Future-known features
    hidden=128,                               # Hidden size in MLPs
    activation="elu",                         # Activation function
    dropout=0.0                               # Dropout rate
)
```

### Internal Components

1. **Target Stream**: `WindowMLP` for target history
    - Input: `(B, L)` target history
    - Output: `(B, H, 2)` contribution to μ and rawσ

2. **Endo Streams**: `ModuleDict` with one `WindowMLP` per endo_col
    - Input: `(B, L)` per AR feature
    - Output: `(B, H, 2)` per feature → summed

3. **Exo Streams**: `ModuleDict` with one `WindowMLP` per exo column
    - Input: `(B, L)` per exogenous feature
    - Output: `(B, H, 2)` per feature → summed

4. **Future Cov Streams**: `ModuleDict` with one `FutureCovMLP` per future feature
    - Input: `(B, H, 1)` per future covariate
    - Output: `(B, H, 2)` per feature → summed

5. **Head**: Additive assembly + constraints
    - μ and rawσ are additive sums of all streams + β
    - σ = softplus(rawσ) + eps

### Forward Pass

```python
def forward(self, target_hist, endo_hists, exo_hists, future_cov):
    """
    target_hist: (B, L) - historical target
    endo_hists: dict of {endo_col: (B, L)} - optional AR features
    exo_hists: dict of {exo_col: (B, L)} - historical exogenous
    future_cov: (B, H, D) - future covariates
    
    Returns: mu (B, H), sigma (B, H)
    """
```

**Contribution Outputs**:
- Summed streams: `contrib_target`, `contrib_endo_sum`, `contrib_exo_sum`, `contrib_future_sum`
- Per-feature: `contrib_endo_<col>`, `contrib_<exo_col>`, `contrib_future_<col>`

---

## Usage Examples

### Example 1: Target-Only (No Exogenous, No AR Features)

```python
# In step1_3_data_pipeline.py
@dataclass
class TSConfig:
    L: int = 168
    H: int = 24
    target: str = "heat_consumption"
    endo_cols: list = field(default_factory=list)                  # Empty!
    exo_cols: list = field(default_factory=list)                   # Empty!
    future_cov_cols: list = field(default_factory=lambda: ["hour_sin", "hour_cos"])
```

Model will only use historical target + future time features.

### Example 2: Single Exogenous Feature

```python
@dataclass
class TSConfig:
    L: int = 168
    H: int = 24
    target: str = "water_consumption"
    endo_cols: list = field(default_factory=list)
    exo_cols: list = field(default_factory=lambda: ["temp"])        # Only temperature
    future_cov_cols: list = field(default_factory=lambda: ["hour_sin", "hour_cos"])
```

### Example 3: Add AR Features (Lagged Target)

```python
@dataclass
class TSConfig:
    L: int = 168
    H: int = 24
    target: str = "heat_consumption"
    endo_cols: list = field(default_factory=lambda: ["heat_lag_1", "heat_roll_24"])
    exo_cols: list = field(default_factory=lambda: ["temp", "wind_speed"])
    future_cov_cols: list = field(default_factory=lambda: ["hour_sin", "hour_cos"])
```

### Example 4: No Future Covariates

```python
@dataclass
class TSConfig:
    L: int = 168
    H: int = 24
    target: str = "demand"
    endo_cols: list = field(default_factory=list)
    exo_cols: list = field(default_factory=lambda: ["price", "volume"])
    future_cov_cols: list = field(default_factory=lambda: [])       # Empty!
```

Model will create `future_cov` as `(H, 0)` tensor.

### Example 5: Extended Feature Set

```python
@dataclass
class TSConfig:
    L: int = 168
    H: int = 24
    target: str = "heat_consumption"
    endo_cols: list = field(default_factory=list)
    exo_cols: list = field(default_factory=lambda: [
        "temp", "temp_squared", "temp_cubed",
        "wind_speed", "wind_speed_squared",
        "dew_point", "humidity", "pressure",
        "solar_radiation", "cloud_cover"
    ])
    future_cov_cols: list = field(default_factory=lambda: [
        "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos",
        "is_weekend", "is_holiday"
    ])
```

---

## Advanced Configuration

### Custom Feature Engineering

Add custom features to your CSV or compute them in `load_and_prepare()`:

```python
def load_and_prepare(csv_path: Path, cfg: TSConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])
    df = df.sort_values(cfg.timestamp_col).set_index(cfg.timestamp_col)
    
    # Custom features
    df['temp_squared'] = df['temp'] ** 2
    df['temp_cubed'] = df['temp'] ** 3
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(float)
    
    # Auto-generate time features
    for cov_col in cfg.future_cov_cols:
        if cov_col == "hour_sin":
            hs, _ = compute_hour_sin_cos(df.index)
            df[cov_col] = hs
        # ... etc
    
    return df
```

### Model Hyperparameters

Modify in [train_tsnamlss.py](train_tsnamlss.py#L250):

```python
model = TSNAMLSSNormal(
    L=cfg.L,
    H=cfg.H,
    target=cfg.target,
    endo_cols=cfg.endo_cols,
    exo_cols=cfg.exo_cols,
    future_cov_cols=cfg.future_cov_cols,
    hidden=256,              # Increase capacity
    activation="relu",       # Try different activations
    dropout=0.1             # Add regularization
)
```

### Training Settings

```bash
python3 train_tsnamlss.py \
    --csv_path data.csv \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001 \
    --device cuda \
    --save_path model.pt
```

---

## Troubleshooting

### Error: "Missing required column: X"

**Cause**: `target` or a column listed in `endo_cols`/`exo_cols` is not in the CSV.

**Solution**: 
1. Check CSV columns: `pd.read_csv("data.csv").columns`
2. Update TSConfig or add column to CSV

### Error: "Future covariate 'X' not found and cannot be auto-generated"

**Cause**: Future covariate not in CSV and not time-based (hour_sin/hour_cos).

**Solution**:
1. Add column to CSV, OR
2. Remove from `future_cov_cols`, OR
3. Extend `load_and_prepare()` to auto-generate it

### Error: "No train samples found"

**Cause**: Not enough data or too many NaNs.

**Solution**:
1. Check data length: `len(df) >= L + H`
2. Check for NaNs: `df[[cfg.target] + cfg.endo_cols + cfg.exo_cols].isna().sum()`
3. Reduce L or H if needed

### Error: "Legacy model architecture no longer supported"

**Cause**: Loading old checkpoint with deprecated architecture.

**Solution**: Retrain model using current scripts.

### Warning: "Model initialized with random weights"

**Cause**: Checkpoint not found or path incorrect.

**Solution**: Verify `--ckpt` path is correct.

---

## Testing Configuration

### Quick Dry Run (2 epochs)

```bash
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 2 \
    --save_path test.pt
```

Expected output:
```
epoch 01: train_nll=... | val_nll=...
epoch 02: train_nll=... | val_nll=...
✓ Saved model to test.pt
```

### Verify Dataset

```bash
python3 step1_3_data_pipeline.py \
    --csv_path nordbyen_features_engineered.csv
```

Expected output:
```
=== Loaded ===
rows: 52560
date range: 2015-01-01 -> 2021-01-01

=== Window dataset sizes ===
train samples: 19745
val samples: 4234
test samples: 7096

=== Sample[0] sanity ===
target_hist: (168,)
target_future: (24,)
temp: (168,)
wind_speed: (168,)
...
future_cov: (24, 2)
```

---

## Performance Tips

1. **Batch Size**: Start with 128-256, increase if GPU memory allows
2. **Hidden Size**: 128 works well, try 256 for complex patterns
3. **Window Length L**: Longer captures seasonality, but increases compute
4. **Feature Selection**: More isn't always better—test incrementally
5. **Regularization**: Add dropout (0.1-0.2) if overfitting

---

## Summary

| Configuration | Purpose | Example |
|---------------|---------|---------|
| `target` | What to forecast | `"heat_consumption"` |
| `endo_cols` | Optional AR features | `["heat_lag_1", "heat_roll_24"]` |
| `exo_cols` | External drivers | `["temp", "wind_speed"]` |
| `future_cov_cols` | Known future info | `["hour_sin", "hour_cos"]` |
| `L` | History window | `168` (7 days) |
| `H` | Forecast horizon | `24` (1 day) |

**Key Principle**: Set `target`, then configure lists for `endo_cols`, `exo_cols`, `future_cov_cols`.

---

For migration from old architecture, see [ARCHITECTURE_DEPRECATION_NOTICE.md](ARCHITECTURE_DEPRECATION_NOTICE.md).
