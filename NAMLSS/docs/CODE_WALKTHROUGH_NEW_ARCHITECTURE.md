# CODE_WALKTHROUGH_NEW_ARCHITECTURE.md — TS‑NAMLSS with Configurable Lists

This document is a **code-first walkthrough** of the refactored **new architecture** that replaces the legacy single-exogenous-feature model with a **fully configurable list-based system**.

It covers:
- What each script does with the new architecture
- How to configure `target`, `endo_cols`, `exo_cols`, `future_cov_cols`
- Exact tensor flows through the pipeline
- Complete commands for data → train → eval → interpret
- What outputs to expect

> This is a **modern, clean** architecture. Legacy code has been completely removed. No backward compatibility.

---

## 0) New Architecture Mental Model

Instead of:
```python
# OLD (deprecated)
target_col = "heat_consumption"
exo_col = "temp"
hour_sin_col = "hour_sin"
```

We now separate a **singular target** from optional AR features:
```python
# NEW
target = "heat_consumption"                 # What to forecast (singular)
endo_cols = ["heat_lag_1", "heat_roll_24"]  # Optional AR features
exo_cols = ["temp", "wind_speed", ...]      # Historical features (past only)
future_cov_cols = ["hour_sin", "hour_cos"] # Future-known features  
```
```
If hour_cos matters more than hour_sin, what does that mean?
It usually means the target’s daily pattern is more aligned with the cosine axis than the sine axis.
 - A very practical way to interpret it:
  - hour_cos mostly separates “midnight vs noon” (because cos is +1 at 00:00, -1 at 12:00).
  - hour_sin mostly separates “6am vs 6pm” (because sin is +1 at 06:00, -1 at 18:00).
So if cosine is more important, your target is likely driven more by a day vs night contrast (or a pattern symmetric around midnight/noon) than by a morning vs evening contrast.

Example intuition:
If the target is generally higher during the daytime (roughly centered around noon) and lower at night, cos will often dominate. If the target is higher in the morning and lower in the evening (or vice versa), sin tends to matter more.
```
**Result**: 
- One model handles any combination of columns
- No code changes needed to add/remove features
- Clean, predictable data flow

---

## 1) Repository Structure (New)

### Main Scripts (All Updated)

| Script | Purpose | New Key Changes |
|--------|---------|-----------------|
| `step1_3_data_pipeline.py` | Data loading, scaling, windowing | TSConfig uses `target` + lists; WindowDataset keys are `target_hist`, `target_future`, `{endo_col}_hist` |
| `train_tsnamlss.py` | Training loop | Model initialized with `target`, `endo_cols`, `exo_cols`, `future_cov_cols` |
| `eval_tsnamlss.py` | Evaluation metrics | Auto-detects config from checkpoint; no legacy flag check |
| `interpret_tsnamlss.py` | Single-sample interpretability | Uses `target` + optional `endo_cols`; dataset keys match new format |
| `global_calibration_tsnamlss.py` | Calibration diagnostics | Works with new architecture automatically |


## 1.5) NAMLSS Theory & Mathematical Foundation

This section explains the theoretical foundation from the original NAMLSS paper (arXiv:2301.11862) and how it applies to our time-series implementation.

### Paper Concept

The core idea of NAMLSS is to model distribution parameters additively:

```
theta_k(x) = a_k( beta_k + Σ_j f_{k,j}(x_j) )
```

Where:
- `theta_k` is a distribution parameter (e.g., mu, sigma)
- `k` indexes parameters (for Normal: k ∈ {mu, sigma})
- `beta_k` is an intercept
- `f_{k,j}(x_j)` is a learned effect function for feature j
- `a_k` is a constraint function (e.g., softplus for sigma > 0)

**Key insight**: Additivity enables interpretability. Each component's contribution is isolated and explainable.

### Time-Series Adaptation

In time series, "a feature" is often a whole sequence. We preserve additivity at the **feature-stream level**:

**Streams**:
- `target_hist`: Historical target (L,)
- `endo_{i}`: Historical AR features (L,) per feature
- `exo_{i}`: Historical exogenous features (L,) per feature
- `future_{j}`: Future-known features (H,) per feature

**Additive decomposition**:
```
mu[h]      = β_mu + target_mu[h] + Σ_i endo_i_mu[h] + Σ_i exo_i_mu[h] + Σ_j future_j_mu[h]
rawsig[h]  = β_σ  + target_σ[h]  + Σ_i endo_i_σ[h]  + Σ_i exo_i_σ[h]  + Σ_j future_j_σ[h]
sigma[h]   = softplus(rawsig[h]) + eps     # Ensure sigma > 0
```

**Why use `rawsig` instead of `sigma`?**
- Additivity holds exactly for rawsig
- sigma = softplus(rawsig) is nonlinear, so sigma is not a clean sum
- For interpretability, we decompose rawsig, then apply softplus

### Dataset-Level Effect/Importance (CGA²M+-Style)

Beyond sample-level decomposition, we compute **stream-level importances** over the test set.

**Definition for a term T**:
```
effect(T) = Σ_{b,h} |T[b,h] - mean(T)| / Σ_{b,h} |y[b,h] - mean(y)|
```

This measures how much T varies, normalized by target variation.

**Global importance**:
```
importance_T = effect(T) / sum(effects for all streams)
```

**Computed for**:
- Mean contributions: contrib_target[...,0], contrib_endo_sum[...,0], contrib_exo_sum[...,0], contrib_future_sum[...,0]
- Uncertainty drivers: contrib_target[...,1], contrib_endo_sum[...,1], contrib_exo_sum[...,1], contrib_future_sum[...,1] (these are rawsig)

**Output**: 
- Global importance (single number per stream)
- Horizon-wise importance curves (how importance varies over h=1..H)

**Why both streams and features?**
- We report stream-level effects (target/endo/exo/future sums) for stability
- Per-feature contributions are also available for fine-grained attribution

---

## 2) Configuration System (The Core Change)

### TSConfig in step1_3_data_pipeline.py

```python
@dataclass
class TSConfig:
    # Window sizes (same as before)
    L: int = 168                    # History window (hours)
    H: int = 24                     # Forecast horizon (hours)

    # COLUMNS (NEW - target + lists)
    timestamp_col: str = "timestamp"
    target: str = None              # Singular target column
    endo_cols: list = None          # Optional AR features (lagged/transformed target)
    exo_cols: list = None           # Exogenous columns (historical features)
    future_cov_cols: list = None    # Future covariate columns (known future)

    # Splits and preprocessing (unchanged)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    reindex_hourly: bool = True

    def __post_init__(self):
        # Defaults (override these for custom configurations)
        if self.target is None:
          self.target = "heat_consumption"
        if self.endo_cols is None:
          self.endo_cols = []
        if self.exo_cols is None:
            self.exo_cols = ["temp", "wind_speed", "dew_point", "temp_squared"]
        if self.future_cov_cols is None:
            self.future_cov_cols = ["hour_sin", "hour_cos"]
```

### How to Customize

**Option 1: Edit TSConfig directly in the file** (lines 37-67)
```python
self.target = "heat_consumption"       # Change target
self.endo_cols = ["heat_lag_1"]         # Optional AR features
self.exo_cols = ["temp"]               # Fewer features
self.future_cov_cols = ["hour_sin", "hour_cos"]  # Add/remove
```

**Option 2: Pass via command line** (future enhancement)
```bash
# Not yet implemented, but can be added to argparse
```

### Column Types Explained

| Column Type | Input | Purpose | Example |
|-------------|-------|---------|---------|
| **target** | From CSV (required) | What you forecast | `"heat_consumption"` |
| **endo_cols** | From CSV (optional) | AR features (lags/transforms) | `["heat_lag_1", "heat_roll_24"]` |
| **exo_cols** | From CSV (required) | Historical context | `["temp", "wind_speed"]` |
| **future_cov_cols** | From CSV or auto-generated | Known future info | `["hour_sin", "hour_cos"]` |

---

## 3) New Dataset Format

### Old Format (Deprecated)
```python
{
    'y_hist': Tensor[L],        # Fixed name
    'y_future': Tensor[H],
    'temp_hist': Tensor[L],     # Only one exo fixed
    'future_cov': Tensor[H, 2]
}
```

### New Format (Current)
```python
{
  # Target (fixed keys)
  'target_hist': Tensor[L],                     # history
  'target_future': Tensor[H],                   # ground truth targets
    
  # Optional AR features (one entry per endo_col)
  'heat_lag_1_hist': Tensor[L],
  'heat_roll_24_hist': Tensor[L],
    
  # Exogenous (one entry per exo_col)
  'temp': Tensor[L],                            # historical
  'wind_speed': Tensor[L],
  'dew_point': Tensor[L],
  'temp_squared': Tensor[L],
    
  # Future covariates (stacked into one tensor)
  'future_cov': Tensor[H, D]                    # D = len(future_cov_cols)
}
```

**Key advantage**: Keys are dynamically named based on column lists. Add/remove features = automatic key generation.

**Note**: `future_cov` remains stacked in the dataset. The model splits it into per-feature streams internally.

---

## 4) Model Architecture (Updated)

### TSNAMLSSNormal Initialization

```python
model = TSNAMLSSNormal(
  L=168,
  H=24,
  target="heat_consumption",                       # NEW parameter
  endo_cols=["heat_lag_1", "heat_roll_24"],         # Optional AR features
  exo_cols=["temp", "wind_speed", "dew_point", "temp_squared"],
  future_cov_cols=["hour_sin", "hour_cos"],
  hidden=128,
  activation="elu",
  dropout=0.0
)
```

### Internal Structure

```
Input: {target_hist, endo_hists, exo_hists, future_cov}

├─ Target Block (one WindowMLP)
│  └─ target_hist (B, L) → (B, hidden) → target_mus, target_rawsigs
│
├─ Endo Block (ModuleDict, one WindowMLP per endo_col)
│  ├─ heat_lag_1 (B, L) → (B, hidden) → lag1_mus, lag1_rawsigs
│  ├─ heat_roll_24 (B, L) → (B, hidden) → roll24_mus, roll24_rawsigs
│  └─ Sum of all endo outputs
│
├─ Exogenous Block (ModuleDict, one WindowMLP per exo_col)
│  ├─ temp (B, L) → (B, hidden) → temp_mus, temp_rawsigs
│  ├─ wind_speed (B, L) → (B, hidden) → wind_mus, wind_rawsigs
│  ├─ dew_point (B, L) → (B, hidden) → dew_mus, dew_rawsigs
│  └─ temp_squared (B, L) → (B, hidden) → temp_sq_mus, temp_sq_rawsigs
│     └─ Sum of all exo outputs
│
├─ Future Covariate Block (ModuleDict, one FutureCovMLP per future_cov_col)
│  ├─ hour_sin (B, H, 1) → (B, hidden) → hour_sin_mus, hour_sin_rawsigs
│  ├─ hour_cos (B, H, 1) → (B, hidden) → hour_cos_mus, hour_cos_rawsigs
│  └─ Sum of all future outputs
│
└─ Head (NormalHead)
  ├─ mu = beta_mu + target_mus + endo_mus_sum + exo_mus_sum + future_mus_sum
  ├─ rawsig = beta_rawsig + target_rawsigs + endo_rawsigs_sum + exo_rawsigs_sum + future_rawsigs_sum
   ├─ sigma = softplus(rawsig) + eps
   └─ Output: (mu, sigma) shapes (B, H)
```

**Key change from old**:
- Target, endo, exo, and future covariates all use per-feature streams
- Summed stream outputs enable both coarse and fine-grained attribution
- Future covariates are split per feature (each sees (H, 1))

---

## 5) Complete Commands (Data → Train → Eval → Interpret)

### 5.1 Data Pipeline

```bash
python3 step1_3_data_pipeline.py \
    --csv_path nordbyen_features_engineered.csv \
    --L 168 --H 24
```

**Expected Output**:
```text
=== Loaded ===
rows: 52560
date range: 2015-01-01 00:00:00 -> 2021-01-01 00:00:00

=== Splits (inclusive index ranges) ===
train: (0, 36792) | rows: 36793
val  : (36793, 44486) | rows: 7694
test : (44487, 52559) | rows: 8073

=== Scaler summary (train only) ===
heat_consumption mean/std: 45.2341 12.6543
temp mean/std: 5.3456 7.8901
wind_speed mean/std: 3.1234 2.1234
dew_point mean/std: 2.0123 6.5432
temp_squared mean/std: 95.3421 98.1234

=== Window dataset sizes ===
train samples: 19745
val samples  : 4234
test samples : 7096

=== Sample[0] sanity ===
Tensor shapes:
target_hist: (168,)
target_future: (24,)
temp: (168,)
wind_speed: (168,)
dew_point: (168,)
temp_squared: (168,)
future_cov: (24, 2)  (H x 2 future features)

First 3 target_future values (scaled): [0.1234, 0.5678, -0.2341]
First 3 future_cov rows:
[[0.5000, 0.8660],
 [0.5176, 0.8554],
 [0.5350, 0.8447]]
```

### 5.2 Training

**Basic command** (uses defaults):
```bash
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 50 \
    --save_path my_model.pt
```

**Full command** (with all options):
```bash
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --save_path my_model.pt \
    --device cuda \
    --early_stopping_patience 5
```

**Expected Output**:
```text
✓ Detected configurable model (new architecture)
  target: heat_consumption
  endo_cols: []
  exo_cols: ['temp', 'wind_speed', 'dew_point', 'temp_squared']
  future_cov_cols: ['hour_sin', 'hour_cos']

Train samples: 19745
Val samples: 4234

epoch 01: train_nll=0.328475 | val_nll=-0.307646
  saved best -> my_model.pt

epoch 02: train_nll=0.298623 | val_nll=-0.308921
  saved best -> my_model.pt

...

epoch 50: train_nll=-0.405123 | val_nll=-0.408942

Early stopping triggered (patience=5). 
Best validation NLL: -0.408942
✓ Final model saved: my_model.pt
```

### 5.3 Evaluation

```bash
python3 eval_tsnamlss.py \
    --ckpt my_model.pt \
    --csv_path nordbyen_features_engineered.csv
```

**Expected Output**:
```text
✓ Detected configurable model:
  target: heat_consumption
  endo_cols: []
  exo_cols: ['temp', 'wind_speed', 'dew_point', 'temp_squared']
  future_cov_cols: ['hour_sin', 'hour_cos']

=== Test Metrics ===
test_nll_scaled: -0.332259
mae_orig: 0.181367
rmse_orig: 0.212774
picp_(1-alpha=0.2): 0.759234
miw_orig: 0.441234
winkler_orig: 0.701234
test samples: 7096
```

**Interpretation**:
- `test_nll_scaled`: Negative OK (depends on scaling)
- `mae_orig`: ~ 0.18 in original units (good if within domain expectations)
- `picp_(1-alpha=0.2)`: ~76% for nominal 80% PI (slight undercover)

### 5.4 Interpretation (Single Sample)

```bash
python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model.pt \
    --sample_idx 100
```

**Expected Output**:
```text
✓ Detected configurable model:
  target: heat_consumption
  endo_cols: []
  exo_cols: ['temp', 'wind_speed', 'dew_point', 'temp_squared']
  future_cov_cols: ['hour_sin', 'hour_cos']

Loading model from my_model.pt...

Processing sample 100...
Sample timestamps:
  history end  : 2016-03-15 22:00:00
  forecast start: 2016-03-16 00:00:00
  forecast end  : 2016-03-16 23:00:00

Additive decomposition check:
  reconstruct mu:      max|sum-total| = 0.000e+00 ✓
  reconstruct rawsig:  max|sum-total| = 0.000e+00 ✓

Endogenous effects:
  mean effect (μ): 0.0542
  scale effect (σ_raw): 0.0234

Exogenous effects:
  temp               : μ → +0.0234, σ_raw → +0.0156
  wind_speed        : μ → -0.0123, σ_raw → -0.0089
  dew_point         : μ → +0.0034, σ_raw → +0.0012
  temp_squared      : μ → +0.0145, σ_raw → +0.0067

Future covariate effects:
  hour_sin/cos      : μ → +0.0456, σ_raw → +0.0234

Computing occlusion maps for lags 0-167 (this runs ~336 forward passes)...

Occlusion significance (top lags by |ΔμΔ|):
  lag 0   (now)       : |Δμ|_mean = 0.0845
  lag 24  (1 day ago) : |Δμ|_mean = 0.0523
  lag 168 (7 days ago): |Δμ|_mean = 0.0234

Saved plots to: interp_out/
  - decomp_mu_sample100.png
  - decomp_rawsig_sample100.png
  - stack_mu_sample100.png
  - stack_rawsig_sample100.png
  - occ_abs_mu_sample100.png
  - occ_abs_rawsig_sample100.png
  - forecast_pi_sample100.png
  - history168_forecast24_pi_sample100.png
  - zscore_sample100.png
Done.
```

### 5.5 Full Dataset Forecast Visualization

**Purpose**: Generate comprehensive timeline plots showing the entire dataset (train + validation + test) with forecasts, ground truth, prediction intervals, and split boundaries.

```bash
module load python/3.12-conda
conda activate env-nam

python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model_cov12.pt \
    --plot_full_dataset \
    --out_dir interp_out_cov12_full
```

**What it does**:
- Generates 1-step ahead forecasts for all 48,574 samples (train, val, test)
- Uses efficient batched inference (batch_size=512) for speed
- Creates two interactive HTML plots with dates, units (kW), and MAE metrics

**Generated files**:
1. **full_dataset_forecasts_1step.html** (12 MB):
   - Complete timeline from 2015-05-02 to 2020-11-14
   - Ground truth (black line) vs forecast (red line)
   - 80% prediction interval (shaded red region)
   - Vertical markers showing Train/Val and Val/Test split boundaries
   - MAE calculated per split displayed in title
   - Interactive: zoom into any date range, hover for exact values
   - Shows all 33,810 train + 7,095 val + 7,096 test forecasts

2. **full_dataset_forecasts_multistep_test.html** (5.1 MB):
   - Test set only, showing 24-hour forecast trajectories
   - ~100 sampled forecast paths to visualize multi-step predictions
   - Shows forecast degradation over 24h horizon
   - Each trajectory shows ground truth vs forecast with 80% PI
   - Useful for understanding multi-step forecast behavior

**Output example**:
```text
=== Generating full dataset forecasts ===
Processing Train set (33810 samples)...
Processing Val set (7095 samples)...
Processing Test set (7096 samples)...
Total forecasts generated: 48001
✓ Saved full dataset forecast plot to: interp_out_cov12_full/full_dataset_forecasts_1step.html

Generating multi-step horizon plot for test set...
Saved multi-step forecast plot to: interp_out_cov12_full/full_dataset_forecasts_multistep_test.html

✓ Full dataset plots generated successfully!
```

**Use cases**:
- Understand forecast performance over entire time period with proper date context
- Identify periods where model performance degrades
- Visualize seasonal patterns and how model captures them
- Compare train/val/test performance visually
- Share comprehensive forecast visualization with stakeholders

**Performance**: ~1 minute to process 48k+ forecasts using batched inference

### 5.6 Interpretation with Dataset-Level Effects

```bash
python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model.pt \
    --sample_idx 100 \
    --do_effects \
    --effects_batch_size 512
```

**Additional Output** (saved to `interp_out/`):
```text
Computing dataset-level effect/importance over test set (batch 512)...

Stream-level effect/importance:
  endo_cols (μ-stream):      importance = 0.358
  exo_cols (μ-stream):       importance = 0.412
  future_cov_cols (μ-stream): importance = 0.230

  endo_cols (σ-stream):      importance = 0.295
  exo_cols (σ-stream):       importance = 0.487
  future_cov_cols (σ-stream): importance = 0.218

Saved:
  - effect_importance_stream_level.json
  - mu_importance_by_horizon.png
  - rawsig_importance_by_horizon_norm_y.png
  - rawsig_importance_by_horizon_norm_rawsig.png
```

### 5.7 Global Calibration

```bash
python3 global_calibration_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model.pt \
    --out_dir global_diag_out
```

**Expected Output**:
```text
✓ Detected configurable model...

Computing calibration metrics over 7096 test samples...

Average calibration metrics:
  PICP (target 0.80): 0.7592
  MIW: 0.4412
  z_mean: 0.0234
  z_std: 1.0234

Saved plots to global_diag_out/:
  - picp_by_horizon.png
  - miw_by_horizon.png
  - zmean_by_horizon.png
  - zstd_by_horizon.png
  - pit_hist.png
```

---

## 6) Deep File-by-File Walkthrough

### 6.1 `step1_3_data_pipeline.py`

**Key functions**:
- `load_and_prepare(csv_path, cfg)` - Load CSV, parse dates, auto-generate time features
- `fit_scalers_on_train(df, cfg, train_range)` - Fit StandardScaler for each column
- `apply_scalers(df, cfg, scalers)` - Transform all columns
- `WindowDataset.__init__()` - Validate windowing
- `WindowDataset.__getitem__()` - Return one sample dict

**Tensor Flow**:
```
Raw CSV
  ↓ (parse timestamp, sort)
DataFrame with DatetimeIndex
  ↓ (reindex hourly, fill missing)
DataFrame (regular 1h grid)
  ↓ (split chronologically)
train_df, val_df, test_df
  ↓ (fit scalers on train)
scalers = {col: StandardScaler for col in [target] + endo_cols + exo_cols}
  ↓ (transform)
scaled_df
  ↓ (windowing)
WindowDataset → dict with {target_hist, target_future, endo_col_hist, exo_col, future_cov}
```

**Important parameters**:
- `L=168`: 7 days of hourly history
- `H=24`: 24-hour forecast
- `train_frac=0.70`: 70% train split

**New in this version**:
- TSConfig uses `target` + lists for `endo_cols`, `exo_cols`, `future_cov_cols`
- Loop over target/endo/exo columns when building scalers/applying transforms
- WindowDataset keys use `target_hist`, `target_future`, and `f"{endo_col}_hist"`

---

### 6.2 `train_tsnamlss.py`

**Key functions**:
- `TSNAMLSSNormal.__init__()` - Initialize model with configurable parameters
- `TSNAMLSSNormal.forward()` - Run forward pass
- `collate_tensor_only()` - Batch collation (custom)
- `train_one_epoch()` - One training epoch
- `evaluate()` - Validation/test evaluation
- `main()` - Full training loop with early stopping

**Model Initialization**:
```python
model = TSNAMLSSNormal(
    L=cfg.L,
    H=cfg.H,
  target=cfg.target,               # NEW
  endo_cols=cfg.endo_cols,         # NEW
  exo_cols=cfg.exo_cols,           # NEW
  future_cov_cols=cfg.future_cov_cols,  # NEW
    hidden=128,
    activation="elu",
    dropout=0.0
)
```

**Forward Pass**:
```python
def forward(self, target_hist, endo_hists, exo_hists, future_cov):
  # target_hist: (B, L)
  # endo_hists: dict of {endo_col: (B, L)}
  # exo_hists: dict of {exo_col: (B, L)}
  # future_cov: (B, H, D)
  # Returns: mu (B, H), sigma (B, H), contributions dict
```

**Loss Computation** (Normal NLL):
```python
# Negative Log-Likelihood for Normal distribution:
# For each (b, h):
nll[b,h] = log(sigma[b,h]) + (y[b,h] - mu[b,h])^2 / (2 * sigma[b,h]^2)

# Aggregate over entire batch and horizon:
loss = mean(nll) over all (b, h) pairs
```

**Why horizon is averaged, not summed**:
We predict H steps ahead. We assume **independence across steps** (diagonal covariance):

```
p(y[1:H] | inputs) = Π_{h=1}^H p(y[h] | μ[h], σ[h])
```

So total NLL = sum of per-step NLL. We average to keep loss magnitude independent of H (enables fair comparison across different horizons).

**Why use rawsig + softplus + eps**:
```python
sigma = softplus(rawsig) + eps
```
- Ensures sigma > 0 (required for log(sigma))
- Softplus smooth gradients (better than ReLU or hardcoded bounds)
- eps prevents sigma from becoming exactly zero

loss = mean(nll)
```

**Batch collation** (custom):
```python
def collate_tensor_only(batch):
  # batch: list of dicts from WindowDataset
  # Returns: {target_hist, target_future, endo_hists, exo_hists, future_cov}
```

**Early stopping**:
- Track best validation NLL
- Stop if no improvement after 5 epochs

**New in this version**:
- Model accepts `target`, `endo_cols`, `exo_cols`, `future_cov_cols` at init
- Target/endo/exo/future are per-feature streams with summed outputs
- Future covariates use one FutureCovMLP per feature (each sees (H, 1))
- Collate function handles new key naming
- Output contributions: `contrib_target`, `contrib_endo_sum`, `contrib_exo_sum`, `contrib_future_sum`, plus per-feature `contrib_endo_<col>`, `contrib_<exo_col>`, `contrib_future_<col>`

---

### 6.3 `eval_tsnamlss.py`

**Key functions**:
- `load_checkpoint_and_config()` - Extract config from saved checkpoint
- `evaluate_test()` - Compute test metrics
- Main loop: load data, forward, compute metrics

**Config Detection**:
```python
ckpt = torch.load(path)
cfg_dict = ckpt['cfg']

# Extract config from checkpoint
target = cfg_dict.get('target', 'heat_consumption')
endo_cols = cfg_dict.get('endo_cols', [])
exo_cols = cfg_dict.get('exo_cols', [...])
future_cov_cols = cfg_dict.get('future_cov_cols', [...])
```

**Metrics Computed**:
1. **NLL** (scaled space)
2. **Point forecast** (mu):
   - MAE/RMSE in original units
3. **Prediction Intervals** (80%):
   - PICP (coverage), MIW (width), Winkler score

**New in this version**:
- Auto-detects config from checkpoint
- Errors on legacy models (raises clear message)
- Uses `target` plus optional `endo_cols` throughout
- Effect/importance uses target + summed endo/exo/future streams

---

### 6.4 `interpret_tsnamlss.py`

**Key functions**:
- `forward_single(model, sample, target)` - Forward pass one sample, collect decomposition
- `occlusion_maps()` - Compute lag sensitivity
- `compute_effects_over_testset()` - Optional: dataset-level importance
- Plotting functions (decomp, occlusion, forecast, z-score)

**Sample-Level Workflow**:
```
1. Load checkpoint + auto-detect config
2. Get one sample from dataset
3. Forward through model, capture internals
4. Extract components: target, endo, exo, future_cov
5. Verify additivity: sum(parts) ≈ total
6. Occlusion: for each lag, perturb → forward → |Δoutput|
7. Plots: decomposition, stack, occlusion heatmaps, forecast+PI, z-score
```

**Key signature change**:
```python
# OLD (deprecated)
forward_single(model, sample)

# NEW
forward_single(model, sample, target)
```

**Dataset-level effect/importance** (optional, `--do_effects`):
```python
def compute_effects_over_testset(loader, model, target, device):
    # Collect over entire test set:
    # - y values
    # - mu contributions per stream
    # - rawsig contributions per stream
    
    # Compute effect = total absolute deviation from mean
    # Compute importance = effect / sum_of_all_effects
    
    # Output:
    # - Global importance (single number per stream)
    # - Horizon-wise importance (curve)
    # - Save as JSON + plots
```

**New in this version**:
- Uses `target` (singular) plus optional `endo_cols`
- Dataset keys match new naming: `target_hist`, `target_future`, `f"{endo_col}_hist"`, `{exo_col}`
- Modern, list-aware decomposition logic

---

### 6.5 `global_calibration_tsnamlss.py`

**Key diagnostics**:
1. **PICP by horizon**: Coverage at each step h
2. **MIW by horizon**: PI width at each step
3. **z-score statistics**: Standardized residuals
4. **PIT histogram**: Uniformity check

**No changes needed** - Works automatically with new architecture.

---

## 7) Quick Reference: Old vs New

### Configuration

| Aspect | Old (Deprecated) | New (Current) |
|--------|-----------------|---------------|
| Target column | `target_col: str` | `target: str` |
| Exo feature | `exo_col: str` | `exo_cols: list` |
| Time features | `hour_sin_col`, `hour_cos_col` | `future_cov_cols: list` |
| Architecture flag | `use_legacy_exo: bool` | (no flag - always modern) |

### Dataset Keys

| Aspect | Old | New |
|--------|-----|-----|
| Target history | `'y_hist'` | `'target_hist'` |
| Target future | `'y_future'` | `'target_future'` |
| Exo history | `'temp_hist'` (hard-coded) | `'{exo_col}'` (any exo) |
| Future cov | `'future_cov'` (stacked) | `'future_cov'` (stacked) |

### Model Parameters

| Aspect | Old | New |
|--------|-----|-----|
| Init signature | `model = TSNAMLSSNormal(L, H, hidden, ...)` | `model = TSNAMLSSNormal(L, H, target, endo_cols, exo_cols, future_cov_cols, hidden, ...)` |
| Forward signature | `forward(endo_hist, exo_hist, future_cov)` | `forward(target_hist, endo_hists, exo_hists, future_cov)` where exo_hists is a dict |
| Exo networks | Single or flag-switched | Always ModuleDict |

---

## 8) Testing the New Architecture

### Quick Dry-Run (No Training)
```bash
python3 step1_3_data_pipeline.py --csv_path nordbyen_features_engineered.csv
```
Verifies data pipeline works and prints sample shapes.

### 2-Epoch Train Test
```bash
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 2 \
    --save_path test_arch.pt
```
Verifies training loop works. Expected: loss converges slightly in 2 steps.

### Full Cycle
```bash
# 1. Data
python3 step1_3_data_pipeline.py --csv_path nordbyen_features_engineered.csv

# 2. Train
python3 train_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --epochs 50 \
    --save_path prod_model.pt

# 3. Eval
python3 eval_tsnamlss.py \
    --ckpt prod_model.pt \
    --csv_path nordbyen_features_engineered.csv

# 4. Interpret
python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt prod_model.pt \
    --do_effects \
    --effects_batch_size 512

# 5. Calibration
python3 global_calibration_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt prod_model.pt
```

Expected output: All commands succeed, plots saved to `interp_out/` and `global_diag_out/`.

---

## 9) Configuration Examples

### Example 1: Auto-Regressive Only (No External Features)
Edit `step1_3_data_pipeline.py`:
```python
self.target = "heat_consumption"
self.endo_cols = []                         # No AR features
self.exo_cols = []                          # Empty!
self.future_cov_cols = ["hour_sin", "hour_cos"]
```
Model uses only historical targets + time features.

### Example 2: Single External Feature
```python
self.target = "heat_consumption"
self.endo_cols = []
self.exo_cols = ["temp"]                    # Only temperature
self.future_cov_cols = ["hour_sin", "hour_cos"]
```

### Example 3: Extended Feature Set
```python
self.target = "heat_consumption"
self.endo_cols = []
self.exo_cols = [
    "temp", "temp_squared", "temp_cubed",
    "wind_speed", "humidity",
    "solar_radiation"
]
self.future_cov_cols = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "is_weekend", "is_holiday"
]
```

Then retrain:
```bash
python3 train_tsnamlss.py --csv_path data.csv --epochs 50 --save_path model_extended.pt
```

No code changes needed.

---

## 10) Migration Checklist (From Old→New)

If you have **old checkpoints** or **legacy configurations**:

- [ ] Archive old checkpoints: `./archive_deprecated_checkpoints.sh`
- [ ] Delete any cached `__pycache__` directories (old pickle artifacts)
- [ ] Remove any custom scripts that reference `target_col`, `exo_col`, `use_legacy_exo`
- [ ] Update your config to use `target` plus `endo_cols`, `exo_cols`, `future_cov_cols`
- [ ] Retrain from scratch with new configuration
- [ ] Verify new checkpoint works: `python3 eval_tsnamlss.py --ckpt new_model.pt --csv_path data.csv`

---

## 11) Diagnostics: Understanding Outputs

### Training Output - What Each Line Means

```text
epoch 01: train_nll=0.328475 | val_nll=-0.307646
```
- `train_nll`: Average NLL over training batch (can be any value, negative OK)
- `val_nll`: Average NLL over validation set (use for early stopping)
- Negative NLL is normal (no absolute reference for log-likelihood)

```text
epoch 02: train_nll=0.298623 | val_nll=-0.308921
  saved best -> my_model.pt
```
- `val_nll` improved (less negative = better)
- Checkpoint updated

```text
Early stopping triggered (patience=5).
Best validation NLL: -0.408942
```
- No improvement for 5 consecutive epochs
- Training stopped, best checkpoint kept

### Evaluation Output - What Each Metric Means

```text
test_nll_scaled: -0.332259
```
- NLL in scaled space. Lower is better, but compare to baseline/null model.

```text
mae_orig: 0.181367
rmse_orig: 0.212774
```
- Point forecast accuracy in **original units** (heat consumption MW/h)
- Use for domain interpretability

```text
picp_(1-alpha=0.2): 0.759234
```
- Prediction interval coverage probability for 80% PI
- Target: 0.80. If < 0.80 → overconfident, if > 0.80 → underconfident

```text
miw_orig: 0.441234
```
- Mean prediction interval width (original units)
- Narrow = confident, wide = uncertain

### Interpretation Output - Decomposition Check

```text
Additive decomposition check:
  reconstruct mu:      max|sum-total| = 0.000e+00 ✓
  reconstruct rawsig:  max|sum-total| = 0.000e+00 ✓
```
- Confirms additivity holds numerically
- Should always be ~0 (machine precision errors only)

---

## 12) Common Issues & Solutions

### Issue: "Missing required column X"
**Cause**: `target` or a column listed in `exo_cols`/`endo_cols` not in CSV

**Solution**:
1. Check CSV header: `head -1 data.csv`
2. Update TSConfig or add column to CSV
3. Retrain

### Issue: "No train samples found"
**Cause**: Not enough data or too many NaNs

**Solution**:
1. Check row count: `wc -l data.csv`
2. Check NaNs: `pandas.read_csv("data.csv")[cols].isna().sum()`
3. Reduce L or H if needed

### Issue: "Legacy architecture not supported"
**Cause**: Trying to load old checkpoint

**Solution**: 
- Archive it: `mv old_model.pt archive/`
- Retrain: `python3 train_tsnamlss.py --csv_path data.csv --save_path new_model.pt`

### Issue: Training loss is NaN
**Cause**: Sigma became negative or zero, or numerical instability

**Solution**:
1. Check if `softplus + eps` is applied to rawsig (it is in new code)
2. Reduce learning rate: `--lr 0.0001`
3. Check for extreme feature values in CSV

---

## 13) Diagnostics: Interpreting Results & Understanding Failures

### Training Metrics

**Loss can be negative** - This is normal! NLL depends on data scaling and sigma values. Never assume negative = good.

**What to watch**:
- If `val_nll` keeps decreasing steeply, training is improving
- If `val_nll` plateaus or increases, model may be overfitting
- If loss becomes NaN, check for extreme feature values or sigma < eps

### Evaluation Metrics: Probabilistic Quality

**NLL (Negative Log-Likelihood)**
```
nll[b,h] = log(sigma[b,h]) + (y[b,h] - mu[b,h])^2 / (2 * sigma[b,h]^2)
```
- Lower NLL is better (but only meaningful when compared to baselines)
- NLL penalizes both bad point forecasts AND overconfident uncertainty

**PICP (Prediction Interval Coverage Probability)**
```
PICP = mean( lo[b,h] <= y[b,h] <= hi[b,h] )
```
- For 80% nominal PI, target PICP ≈ 0.80
- **If PICP < 0.80**: Model is **overconfident** (too narrow intervals)  
  → Solution: Increase sigma or use recalibration
- **If PICP > 0.80**: Model is **underconfident** (too wide intervals)  
  → Solution: Reduce sigma or improve mean forecast

**MIW (Mean Interval Width)**
- Average width of prediction intervals
- Narrower = more confident; wider = more uncertain
- Use with PICP: narrow intervals with PICP=0.80 is ideal

**Winkler Score**
```
Winkler = width + penalty_if_outside
```
- Combines width (narrowness) and coverage
- Lower is better
- Penalizes both under-coverage and over-width

### Evaluation Metrics: Point Forecast Quality

**MAE (Mean Absolute Error)** in original units
- Easier to interpret than scaled metrics
- Compare to domain baseline (e.g., persistence forecast)

**RMSE (Root Mean Squared Error)** in original units
- Penalizes large errors more than MAE
- More sensitive to outliers

### Calibration Diagnostics (Z-Scores)

```
z[b,h] = (y[b,h] - mu[b,h]) / sigma[b,h]
```

**zmean by horizon**:
- Should be ≈ 0 (mean forecast is unbiased)
- **If zmean > 0**: Predictions systematically too low (under-predict)
- **If zmean < 0**: Predictions systematically too high (over-predict)

**zstd by horizon**:
- Should be ≈ 1 (uncertainty well-calibrated)
- **If zstd > 1**: Sigma too small (over-confident)  
  → Predictions miss more than expected
- **If zstd < 1**: Sigma too large (under-confident)  
  → Giving up confidence unnecessarily

**PIT Histogram** (Probability Integral Transform):
```
u[b,h] = CDF_Normal(y[b,h]; mu[b,h], sigma[b,h])
```
- For well-calibrated forecasts, u should be **uniform** on [0,1]
- Non-uniform = mismatch between model and data
- Left-skewed: predictions often too optimistic
- Right-skewed: predictions often too pessimistic

### Common Pattern Failures

| Pattern | Cause | Solution |
|---------|-------|----------|
| PICP ≈ 0.76, zmean > 0, PIT left-skewed | Under-predicting bias + overconfidence | Increase bias in mu; increase sigma |
| PICP ≈ 0.95, zmean ≈ 0, zstd > 1 | Good mean, but sigma too small | Increase sigma scaling or use raw_sig post-hoc calibration |
| PICP ≈ 0.80, MAE high, zstd ≈ 1 | Well-calibrated but inaccurate mean | Better features or model capacity needed |
| Loss = NaN | Numerical instability | Reduce LR; check features for extremes; verify softplus applied |

---

## 14) Reproducibility Checklist (Common Silent Bugs)

Before submitting results or debugging failures, verify:

### Data Leakage
- [ ] **Chronological split**: No future data in train/val
- [ ] **Scaler fit on train only**: Never fit scalers on val/test
- [ ] **No future knowledge**: Exogenous features have no future info
- [ ] **Window alignment**: History strictly before future (t_end+1 gap)

### Numerical Stability
- [ ] **Sigma positivity**: `sigma = softplus(rawsig) + eps` always applied
- [ ] **Feature normalization**: Features scaled to reasonable range (e.g., [-5, 5])
- [ ] **Batch normalization**: Not used (can cause train/test mismatch in forecasting)
- [ ] **Loss monitoring**: Loss tracked per epoch; NaN detection enabled

### Model Architecture
- [ ] **ModuleDict for exogenous**: Each exogenous column has its own network
- [ ] **Future covariate dimension**: Matches `len(future_cov_cols)`
- [ ] **Batch size consistency**: Same collate function for train/val/test
- [ ] **Device consistency**: All tensors on same device (cuda or cpu)

### Evaluation Correctness
- [ ] **Inverse transform applied**: Metrics in original units
- [ ] **Horizon-wise aggregation**: NLL averaged, not summed
- [ ] **PI bounds**: lo < hi always; boundaries use quantiles correctly
- [ ] **Checkpoint loaded properly**: Config extracted; no architecture mismatch

### Preprocessing Details
- [ ] **Hourly alignment**: Gaps filled/fixed before windowing
- [ ] **Timestamp parsing**: No timezone confusion
- [ ] **Missing values**: Explicit NaN handling (drop vs. interpolate)
- [ ] **Sorting**: Data sorted by time; no random permutation

### Output Paths
- [ ] **Saved paths printed**: Always print absolute path of saved plots/checkpoints
- [ ] **Output directory created**: No silent failures due to missing dirs
- [ ] **Plots readable**: PNG files actually written; no zero-byte files
- [ ] **JSON valid**: effect_importance.json is parseable

---

## 15) Loss Function Details (Deep Dive)

### Negative Log-Likelihood for Normal Distribution

For a Normal distribution with mean μ and std σ:

```
p(y | μ, σ) = (1 / (σ√(2π))) * exp(-(y-μ)² / (2σ²))
```

Taking negative log:
```
-log p(y | μ, σ) = log(σ) + (y-μ)² / (2σ²) + const
```

This is the **NLL for one observation**.

### Horizon Aggregation (Diagonal Covariance)

We predict for H steps ahead. We assume **independence across steps** (diagonal covariance):

```
p(y[1:H] | inputs) = Π_{h=1}^H p(y[h] | μ[h], σ[h])
```

So total NLL:
```
NLL_total = Σ_{h=1}^H NLL[h]
```

And for a batch:
```
loss = mean over (b, h) of NLL[b,h]
```

**Why this assumption?**
- Simplifies computation (no covariance matrix inversion)
- In practice, decent approximation for many time series
- Can be relaxed later (e.g., with AR covariances)

### Softplus Constraint on Sigma

Raw sigma can be unconstrained (can go negative). We apply:

```
sigma_final = softplus(raw_sigma) + eps
```

Where:
```
softplus(x) = log(1 + exp(x))
```

**Properties**:
- Always > 0
- Smooth gradients (unlike ReLU)
- Approaches x for large x (linear for large values)
- Approaches log(2) for x=0

**Why not exp(raw_sigma)?**
- Exp can explode or vanish with small changes
- Softplus is gentler on gradients

---

## 16) File Organization After Running All Steps

```
repository/
├── README.md
├── CONFIGURABLE_ARCHITECTURE_GUIDE.md
├── ARCHITECTURE_DEPRECATION_NOTICE.md
├── CLEANUP_STATUS.md
├── CODE_WALKTHROUGH_NEW_ARCHITECTURE.md (this file)
│
├── step1_3_data_pipeline.py
├── train_tsnamlss.py
├── eval_tsnamlss.py
├── interpret_tsnamlss.py
├── global_calibration_tsnamlss.py
│
├── prod_model.pt               # Your trained checkpoint
├── test_arch.pt                # (Optional) test checkpoint
│
├── archive/
│   ├── best_model_4exo.pt      # Old (deprecated)
│   ├── best_tsnamlss.pt        # Old (deprecated)
│   └── test_new_arch.pt        # New arch reference
│
├── interp_out/                 # Single sample interpretability
│   ├── decomp_mu_sample100.png
│   ├── occ_abs_mu_sample100.png
│   ├── forecast_pi_sample100.png
│   ├── effect_importance_stream_level.json
│   ├── mu_importance_by_horizon.png
│   └── ... (more plots)
│
├── global_diag_out/            # Calibration diagnostics
│   ├── picp_by_horizon.png
│   ├── zmean_by_horizon.png
│   ├── pit_hist.png
│   └── ... (more plots)
│
└── processing/
    ├── nordbyen_processing/
    ├── centrum_processing/
    └── tommerby_processing/
```

---

## 17) Summary

**What Changed**:
- Configuration: `target` + Lists (`endo_cols`, `exo_cols`, `future_cov_cols`)
- Architecture: Single-exo flag → Always ModuleDict (any number of features)
- Dataset keys: Fixed names → Dynamic names (based on column lists)
- Checkpoints: Old archived, new format only

**What Stayed**:**
- Training pipeline
- Interpretability analysis
- Evaluation metrics
- Calibration diagnostics

**Benefit**: Clean, maintainable, configurable system. Add/remove features = no code changes, just edit the lists.

---

For the **original code walkthrough** (legacy), see `CODE_WALKTHROUGH.md`.  
For **complete configuration examples**, see `CONFIGURABLE_ARCHITECTURE_GUIDE.md`.
