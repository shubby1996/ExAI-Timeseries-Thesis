# Model Configuration and Prediction Flow Guide

**Purpose:** This document explains each model architecture, configuration, and the complete flow from input data to predictions. Use this to understand and explain how each model processes data and generates forecasts.

**Last Updated:** January 11, 2026

---

## Table of Contents
1. [Overview](#overview)
2. [Common Pipeline Components](#common-pipeline-components)
3. [NHiTS Model](#nhits-model)
4. [TFT Model (Temporal Fusion Transformer)](#tft-model)
5. [TimesNet Model](#timesnet-model)
6. [Model Comparison Table](#model-comparison-table)
7. [Configuration Files](#configuration-files)

---

## Overview

We use **three primary models** for time series forecasting, each available in two variants:
- **Quantile Regression (Q)**: Produces probabilistic predictions with uncertainty intervals (p10, p50, p90)
- **Point Forecast (MSE)**: Produces deterministic single-point predictions

### Model Portfolio
| Model ID | Type | Framework | Loss Function | Use Case |
|----------|------|-----------|---------------|----------|
| NHITS_Q | NHiTS | Darts | Quantile Regression | Probabilistic heat/water forecasting |
| NHITS_MSE | NHiTS | Darts | Mean Squared Error | Deterministic forecasting |
| TIMESNET_Q | TimesNet | NeuralForecast | Multi-Quantile Loss | Probabilistic forecasting |
| TIMESNET_MSE | TimesNet | NeuralForecast | MSE | Deterministic forecasting |
| TFT_Q | TFT | Darts | Quantile Regression | Probabilistic with attention |
| TFT_MSE | TFT | Darts | MSE | Deterministic with attention |

---

## Common Pipeline Components

All models share the same preprocessing pipeline before model-specific processing begins.

### 1. Feature Engineering (Layer 1)
**Input:** Raw CSV with timestamps and raw measurements
- Heat: `heat_consumption` (kWh)
- Water: `water_consumption` (m³)

**Processing:**
- Weather data integration (temperature, humidity, wind, etc.)
- Lag features: 1-hour, 24-hour lags
- Rolling statistics: 24-hour rolling mean
- Interaction features: temperature × wind, temperature × weekend
- Calendar features: hour, day_of_week, month, season
- Holiday indicators: public holidays, school holidays

**Output:** `nordbyen_features_engineered.csv` or `centrum_features_engineered_from_2018-04-01.csv`

### 2. Feature Configuration (Layer 2)

Features are categorized into three roles:

#### Target Variable
- **Heat:** `heat_consumption`
- **Water:** `water_consumption`

#### Past Covariates (Observable Only Up to Now)
Known only for historical periods, must be forecasted or unavailable for future:
```python
past_covariates = [
    # Weather observations
    "temp", "dew_point", "humidity", "clouds_all",
    "wind_speed", "rain_1h", "snow_1h", "pressure",
    
    # Lagged target features
    "heat_lag_1h" / "water_lag_1h",
    "heat_lag_24h" / "water_lag_24h", 
    "heat_rolling_24h" / "water_rolling_24h",
    
    # Engineered interactions
    "temp_squared", "temp_wind_interaction", 
    "temp_weekend_interaction"
]
```

#### Future Covariates (Known for Future Periods)
Available for both past and future (calendar features):
```python
future_covariates = [
    # Time encodings
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "month", "season",
    
    # Calendar flags
    "is_weekend", "is_public_holiday", 
    "is_school_holiday"
]
```

### 3. Data Splitting

Time-based splits (no shuffling to maintain temporal order):
```
Train:  [start] ──────────→ [train_end]
Val:                         [train_end+1h] ─→ [val_end]
Test:                                            [val_end+1h] ─→ [end]
```

**Typical Configuration:**
- **Train end:** 2023-10-31
- **Val end:** 2024-03-31
- **Test period:** 2024-04-01 onwards

### 4. Scaling (Normalization)

Each feature type is scaled independently using **robust scaling** (median and IQR):

```python
# Fit scalers on TRAINING DATA ONLY (prevents data leakage)
target_scaler = Scaler()
past_covariates_scaler = Scaler()
future_covariates_scaler = Scaler()

# Fit on train, transform train/val/test
target_scaled = target_scaler.fit_transform(train_target)
val_target_scaled = target_scaler.transform(val_target)
test_target_scaled = target_scaler.transform(test_target)
```

**Why Scale?**
- Neural networks train faster with normalized inputs
- Prevents features with large magnitudes from dominating
- Improves gradient descent convergence

**Why Fit on Train Only?**
- Prevents **data leakage** (test statistics bleeding into training)
- Ensures model sees only training distribution during learning
- Makes validation metrics realistic

### 5. TimeSeries Conversion

Convert pandas DataFrames to Darts/NeuralForecast TimeSeries objects:
```python
# Darts format (for NHiTS, TFT)
target_ts = TimeSeries.from_dataframe(df, value_cols=['heat_consumption'], freq='H')
past_cov_ts = TimeSeries.from_dataframe(df, value_cols=past_covariates_cols, freq='H')
future_cov_ts = TimeSeries.from_dataframe(df, value_cols=future_covariates_cols, freq='H')

# NeuralForecast format (for TimesNet)
nf_df = df.reset_index().rename(columns={'timestamp': 'ds', 'heat_consumption': 'y'})
nf_df['unique_id'] = 'nordbyen'  # Series identifier
```

---

## NHiTS Model

### Architecture Overview

**NHiTS** = **N**eural **Hi**erarchical Interpolation for **T**ime **S**eries

**Key Innovation:** Multi-rate hierarchical architecture that models time series at multiple temporal scales simultaneously.

### Architecture Diagram

```
Input (168h history)
       ↓
┌──────────────────────────────┐
│   Stack 1: High Resolution   │
│   (captures daily patterns)  │
│   - Multiple blocks          │
│   - Small pooling kernel     │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│   Stack 2: Medium Resolution │
│   (captures weekly patterns) │
│   - Intermediate pooling     │
└──────────────────────────────┘
       ↓
┌──────────────────────────────┐
│   Stack 3: Low Resolution    │
│   (captures seasonal trends) │
│   - Large pooling kernel     │
└──────────────────────────────┘
       ↓
   Sum all outputs
       ↓
Prediction (24h forecast)
```

### Configuration Parameters

```python
nhits_config = {
    # === Data Shape ===
    "input_chunk_length": 168,      # Look back 7 days (168 hours)
    "output_chunk_length": 24,      # Forecast 1 day ahead (24 hours)
    
    # === Architecture ===
    "num_stacks": 3,                # Number of hierarchical stacks (default/HPO)
    "num_blocks": 1,                # Blocks per stack (default/HPO)
    "num_layers": 2,                # MLP layers in each block (default/HPO)
    "layer_widths": 512,            # Neurons per layer (default/HPO)
    "dropout": 0.1,                 # Dropout rate for regularization (default/HPO)
    
    # === Training ===
    "batch_size": 32,               # Mini-batch size
    "n_epochs": 100,                # Training epochs (heat/water)
    "optimizer_kwargs": {
        "lr": 1e-3,                 # Learning rate (default/HPO)
        "weight_decay": 1e-5        # L2 regularization (default/HPO)
    },
    
    # === Loss Function ===
    "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9])  # For NHITS_Q
    # OR: No likelihood (MSE default) for NHITS_MSE
    
    # === Other ===
    "random_state": 42,             # Reproducibility seed
    "force_reset": True,            # Reinitialize model each training
    "pl_trainer_kwargs": {
        "logger": True,             # Enable TensorBoard logging
        "enable_checkpointing": False
    }
}
```

### How NHiTS Processes Data

#### Step 1: Input Preparation
```python
# Historical window (168 hours = 7 days)
historical_target = target_ts[:prediction_start - 1h]  # Shape: (168, 1)

# Combine past and future covariates
historical_covariates = past_cov_ts.stack(future_cov_ts)  # Shape: (168, num_features)
```

#### Step 2: Forward Pass Through Stacks

Each **stack** operates at a different temporal resolution:

**Stack 1 (High Resolution):**
- Captures **fine-grained patterns** (daily cycles, hourly spikes)
- Small max pooling → retains detailed information
- Outputs partial forecast

**Stack 2 (Medium Resolution):**
- Captures **weekly patterns** (weekday vs weekend)
- Medium pooling → smooths noise
- Outputs partial forecast

**Stack 3 (Low Resolution):**
- Captures **seasonal trends** (long-term patterns)
- Large pooling → extreme smoothing
- Outputs partial forecast

**Final Output:** Sum of all stack predictions
```
final_forecast = stack1_output + stack2_output + stack3_output
```

#### Step 3: Generate Predictions

**For Quantile Models (NHITS_Q):**
```python
# Generate 100 samples from learned distribution
predictions = model.predict(
    n=24,                           # Forecast 24 hours
    series=historical_target,
    past_covariates=historical_covariates,
    num_samples=100                 # Monte Carlo samples
)

# Extract quantiles from samples
p10 = predictions.quantile(0.1)    # 10th percentile (lower bound)
p50 = predictions.quantile(0.5)    # Median (central forecast)
p90 = predictions.quantile(0.9)    # 90th percentile (upper bound)
```

**For Point Models (NHITS_MSE):**
```python
# Single deterministic prediction
predictions = model.predict(
    n=24,
    series=historical_target,
    past_covariates=historical_covariates,
    num_samples=1
)
p50 = predictions.values()          # Only median available
p10 = p50                          # No uncertainty
p90 = p50
```

#### Step 4: Inverse Scaling
```python
# Transform back to original scale
p10_original = target_scaler.inverse_transform(p10)
p50_original = target_scaler.inverse_transform(p50)
p90_original = target_scaler.inverse_transform(p90)
```

### Strengths of NHiTS
✅ **Multi-scale modeling** captures both short and long patterns  
✅ **Efficient** - hierarchical design reduces parameters  
✅ **Interpretable** - each stack models specific temporal scale  
✅ **Fast training** compared to transformer models  
✅ **Good with covariates** - naturally handles exogenous features

### Limitations of NHiTS
❌ **No explicit attention** mechanism (unlike TFT)  
❌ **Less flexible** than transformers for complex interactions  
❌ **Fixed hierarchical structure** may not suit all data

---

## TFT Model

### Architecture Overview

**TFT** = **T**emporal **F**usion **T**ransformer

**Key Innovation:** Attention-based architecture with gating mechanisms and variable selection, designed specifically for multi-horizon forecasting with heterogeneous inputs.

### Architecture Diagram

```
Past Covariates          Future Covariates
(168h history)           (24h known future)
       ↓                         ↓
┌──────────────────┐    ┌──────────────────┐
│ Variable         │    │ Variable         │
│ Selection        │    │ Selection        │
│ Network (VSN)    │    │ Network (VSN)    │
└──────────────────┘    └──────────────────┘
       ↓                         ↓
┌──────────────────┐    ┌──────────────────┐
│ LSTM Encoder     │    │ LSTM Decoder     │
│ (Past Context)   │    │ (Future Context) │
└──────────────────┘    └──────────────────┘
       ↓                         ↓
       └─────────┬───────────────┘
                 ↓
        ┌─────────────────┐
        │ Multi-Head      │
        │ Self-Attention  │
        │ (Temporal       │
        │  Dependencies)  │
        └─────────────────┘
                 ↓
        ┌─────────────────┐
        │ Gated Residual  │
        │ Network (GRN)   │
        └─────────────────┘
                 ↓
        ┌─────────────────┐
        │ Quantile Output │
        │ Layer           │
        └─────────────────┘
                 ↓
        Prediction (24h forecast)
```

### Configuration Parameters

```python
tft_config = {
    # === Data Shape ===
    "input_chunk_length": 168,      # Look back 7 days
    "output_chunk_length": 24,      # Forecast 1 day ahead
    
    # === Architecture ===
    "hidden_size": 64,              # LSTM/attention hidden dimension (default/HPO)
    "lstm_layers": 1,               # Number of LSTM layers (default/HPO)
    "num_attention_heads": 4,       # Multi-head attention heads (default/HPO)
    "dropout": 0.1,                 # Dropout rate (default/HPO)
    
    # === Training ===
    "batch_size": 32,
    "n_epochs": 100,
    "optimizer_kwargs": {
        "lr": 1e-3                  # Learning rate (default/HPO)
    },
    
    # === Loss Function ===
    "likelihood": QuantileRegression(quantiles=[0.1, 0.5, 0.9])  # For TFT_Q
    # OR: No likelihood (MSE default) for TFT_MSE
    
    # === Other ===
    "random_state": 42,
    "force_reset": True,
    "pl_trainer_kwargs": {
        "logger": True,
        "enable_checkpointing": False
    }
}
```

### How TFT Processes Data

#### Step 1: Input Preparation

**Unlike NHiTS, TFT uses past and future covariates SEPARATELY:**

```python
# Historical window
historical_target = target_ts[:prediction_start - 1h]
historical_past_cov = past_cov_ts[:prediction_start - 1h]

# Known future features (calendar for next 24 hours)
future_cov = future_cov_ts[prediction_start : prediction_start + 23h]
```

**Why Separate?**
- TFT explicitly models the difference between "what we observed" vs "what we know about the future"
- Past covariates: weather observations (only available historically)
- Future covariates: calendar features (known for both past and future)

#### Step 2: Variable Selection Network (VSN)

TFT's first innovation - **learns which features matter most:**

```python
# For each input feature, compute importance weight
for feature in all_features:
    importance_weight = sigmoid(linear_layer(feature))
    selected_feature = feature * importance_weight

# Output: weighted feature set (suppresses irrelevant features)
```

**Benefit:** 
- Reduces noise from irrelevant features
- Improves interpretability (can visualize feature importance)
- Automatic feature engineering

#### Step 3: LSTM Encoding

**Past Encoder:**
```python
# Process historical context
past_context = LSTM(
    inputs=[historical_target, historical_past_cov, historical_future_cov],
    hidden_size=64,
    num_layers=1
)
# Output: compressed historical representation
```

**Future Decoder:**
```python
# Process known future features
future_context = LSTM(
    inputs=[future_cov],
    hidden_size=64,
    num_layers=1,
    initial_state=past_context  # Initialize with past state
)
# Output: contextual representation for each forecast step
```

#### Step 4: Multi-Head Self-Attention

**Purpose:** Capture long-range temporal dependencies

```python
# Attention allows each time step to "attend to" other relevant time steps
for head in range(num_attention_heads):
    # Query, Key, Value projections
    Q = linear_Q(future_context)
    K = linear_K(future_context)
    V = linear_V(future_context)
    
    # Attention scores (which past steps are relevant?)
    attention_weights = softmax(Q @ K.T / sqrt(hidden_size))
    
    # Weighted sum of values
    attention_output = attention_weights @ V

# Concatenate all heads
multi_head_output = concat([head1, head2, head3, head4])
```

**Example:** 
- When predicting hour 18 (6 PM peak), attention mechanism can focus on:
  - Yesterday's hour 18 (daily pattern)
  - Last week's hour 18 (weekly pattern)
  - Recent hours 16-17 (momentum/trend)

#### Step 5: Gated Residual Network (GRN)

**Purpose:** Adaptive non-linear processing with skip connections

```python
# Gating mechanism controls information flow
gate = sigmoid(linear_gate(attention_output))
transformed = ELU(linear_transform(attention_output))
output = gate * transformed + (1 - gate) * attention_output  # Skip connection
```

#### Step 6: Generate Predictions

**For Quantile Models (TFT_Q):**
```python
predictions = model.predict(
    n=24,
    series=historical_target,
    past_covariates=historical_past_cov,
    future_covariates=future_cov,
    num_samples=100
)

p10 = predictions.quantile(0.1)
p50 = predictions.quantile(0.5)
p90 = predictions.quantile(0.9)
```

**For Point Models (TFT_MSE):**
```python
predictions = model.predict(
    n=24,
    series=historical_target,
    past_covariates=historical_past_cov,
    future_covariates=future_cov,
    num_samples=1
)
p50 = predictions.values()
```

### Strengths of TFT
✅ **Attention mechanism** explicitly models temporal dependencies  
✅ **Variable selection** automatically identifies important features  
✅ **Interpretable** - can visualize attention weights and feature importance  
✅ **Handles heterogeneous inputs** (past/future/static)  
✅ **State-of-the-art accuracy** on many benchmarks

### Limitations of TFT
❌ **Computationally expensive** (attention is O(n²))  
❌ **Slower training** than NHiTS  
❌ **More parameters** → requires more data  
❌ **Complex architecture** → harder to debug

---

## TimesNet Model

### Architecture Overview

**TimesNet** = Time Series Network with 2D Vision Convolutions

**Key Innovation:** Transforms 1D time series into 2D tensors (temporal + periodic dimensions) and applies 2D convolutions (like image processing) to capture both temporal and periodic patterns.

### Architecture Diagram

```
Input (168h history + 24h future features)
       ↓
┌────────────────────────────────────┐
│ 1D → 2D Transformation             │
│ Convert time series to 2D "image"  │
│ Rows: Temporal dimension           │
│ Cols: Periodic dimension (e.g., 24)│
└────────────────────────────────────┘
       ↓
┌────────────────────────────────────┐
│ TimesBlock 1                       │
│ ┌────────────────┐                 │
│ │ 2D Convolution │ → Captures      │
│ │ (Inception)    │    spatial +    │
│ │                │    temporal     │
│ └────────────────┘    patterns     │
└────────────────────────────────────┘
       ↓
┌────────────────────────────────────┐
│ TimesBlock 2                       │
│ (Deeper hierarchical features)     │
└────────────────────────────────────┘
       ↓
┌────────────────────────────────────┐
│ 2D → 1D Transformation             │
│ Flatten back to time series        │
└────────────────────────────────────┘
       ↓
┌────────────────────────────────────┐
│ Linear Projection                  │
│ Map to forecast horizon (24h)      │
└────────────────────────────────────┘
       ↓
Prediction (24h forecast)
```

### Configuration Parameters

```python
timesnet_config = {
    # === Data Shape ===
    "h": 24,                        # Forecast horizon (24 hours)
    "input_size": 168,              # Look back window (168 hours)
    
    # === Features ===
    "futr_exog_list": [             # All exogenous as future covariates
        "temp", "humidity", "wind_speed", ...,  # Weather
        "hour_sin", "hour_cos", "day_of_week", ... # Calendar
    ],
    
    # === Architecture ===
    "hidden_size": 64,              # Hidden layer dimension (default/HPO)
    "conv_hidden_size": 64,         # 2D convolution channels (default/HPO)
    "top_k": 2,                     # Number of dominant frequencies (default/HPO)
    "dropout": 0.1,                 # Dropout rate (default/HPO)
    
    # === Training ===
    "batch_size": 32,
    "max_steps": n_epochs * (n_samples // batch_size),  # Total gradient updates
    "learning_rate": 1e-3,          # Learning rate (default/HPO)
    
    # === Loss Function ===
    "loss": MQLoss(quantiles=[0.1, 0.5, 0.9])  # For TIMESNET_Q
    # OR: MSE() for TIMESNET_MSE
    
    # === Scaling ===
    "scaler_type": "robust",        # Robust scaling for exogenous features
    
    # === PyTorch Lightning ===
    "logger": True,
    "enable_checkpointing": False
}
```

### How TimesNet Processes Data

#### Step 1: Input Preparation (NeuralForecast Format)

**Unlike Darts models, NeuralForecast uses a different data format:**

```python
# Convert to NeuralForecast format
nf_df = pd.DataFrame({
    'unique_id': 'nordbyen',           # Series identifier
    'ds': timestamps,                  # DatetimeIndex
    'y': target_values,                # Target variable
    'temp': temperature,               # Exogenous feature 1
    'humidity': humidity,              # Exogenous feature 2
    ...                                # All other features
})

# Historical window
hist_df = nf_df[(ds >= pred_start - 168h) & (ds < pred_start)]

# Future features (known for next 24 hours)
fut_df = nf_df[(ds >= pred_start) & (ds <= pred_start + 23h)]
```

**Important:** TimesNet treats ALL exogenous features as **future-known** (assumes weather is forecasted).

#### Step 2: 1D → 2D Transformation

**Core innovation:** Reshape 1D time series into 2D "image":

```python
# Detect dominant periods via FFT (Fast Fourier Transform)
fft_output = FFT(input_sequence)
dominant_periods = top_k_frequencies(fft_output, k=2)  # e.g., [24, 168]

# For each dominant period p, create 2D representation
for period in dominant_periods:
    # Reshape: (seq_length,) → (seq_length // period, period)
    # Example: 168 hours with period=24 → (7, 24) "image"
    #   Rows = days (7 days)
    #   Cols = hours within day (24 hours)
    tensor_2d = reshape(input_sequence, (-1, period))
```

**Intuition:**
- **Period = 24:** Daily pattern (captures hourly cycles)
- **Period = 168:** Weekly pattern (captures day-of-week cycles)
- 2D tensor allows convolutions to capture both dimensions simultaneously

**Example for Heat Consumption:**
```
Period = 24 (daily pattern)
           Hour 0  1  2  3  ... 22 23
Day 1      [150  140 135 130 ... 200 180]
Day 2      [155  145 138 132 ... 205 185]
Day 3      [148  142 136 130 ... 198 178]
...
Day 7      [152  144 137 131 ... 202 182]

Now 2D convolution can capture:
- Vertical patterns (consistency across days at same hour)
- Horizontal patterns (within-day progression)
- Diagonal patterns (phase shifts)
```

#### Step 3: 2D Inception Convolutions

**Apply Inception-style multi-scale convolutions:**

```python
# Multiple parallel convolutions with different kernel sizes
conv1x1 = Conv2D(kernel_size=1, filters=64)(tensor_2d)    # Point-wise
conv3x3 = Conv2D(kernel_size=3, filters=64)(tensor_2d)    # Local patterns
conv5x5 = Conv2D(kernel_size=5, filters=64)(tensor_2d)    # Broader patterns

# Concatenate all scales
multi_scale_features = concat([conv1x1, conv3x3, conv5x5])
```

**Why Inception?**
- **Different kernel sizes** capture patterns at different scales
- **1×1:** Point-wise interactions (e.g., feature combinations)
- **3×3:** Local patterns (e.g., 3-hour windows within 3-day spans)
- **5×5:** Broader patterns (e.g., weekly cycles)

#### Step 4: Hierarchical TimesBlocks

**Stack multiple TimesBlocks for deeper representations:**

```python
# Block 1: Extract low-level periodic patterns
block1_output = TimesBlock(
    2d_conv(tensor_2d),
    residual_connection,
    layer_norm
)

# Block 2: Extract higher-level patterns
block2_output = TimesBlock(
    2d_conv(block1_output),
    residual_connection,
    layer_norm
)
```

Each block refines the representation, capturing increasingly abstract patterns.

#### Step 5: 2D → 1D Transformation

**Flatten back to 1D time series:**

```python
# Reshape 2D tensor back to 1D sequence
output_1d = flatten(block2_output)  # Shape: (seq_length, features)
```

#### Step 6: Projection to Forecast Horizon

**Linear layer maps to output length:**

```python
# Project to 24-hour forecast
forecast = Linear(hidden_size → 24)(output_1d)
```

#### Step 7: Generate Predictions

**For Quantile Models (TIMESNET_Q):**
```python
fcst = model.predict(df=hist_df, futr_df=fut_df)

# Extract quantiles from output columns
p10 = fcst['TimesNet-lo-80.0']     # 10th percentile
p50 = fcst['TimesNet-median']      # Median
p90 = fcst['TimesNet-hi-80.0']     # 90th percentile
```

**For Point Models (TIMESNET_MSE):**
```python
fcst = model.predict(df=hist_df, futr_df=fut_df)
p50 = fcst['TimesNet']              # Point forecast only
p10 = p50                          # No uncertainty
p90 = p50
```

### Strengths of TimesNet
✅ **Novel 2D approach** captures periodic patterns naturally  
✅ **Inception architecture** models multi-scale patterns  
✅ **Fast inference** - convolutions are efficient  
✅ **Handles irregular patterns** better than LSTM-based models  
✅ **Recent SOTA** on multiple benchmarks

### Limitations of TimesNet
❌ **Assumes periodic patterns exist** (less effective for aperiodic data)  
❌ **FFT overhead** for period detection  
❌ **Less interpretable** than attention-based models  
❌ **Treats all exogenous as future-known** (may not suit all scenarios)

---

## Model Comparison Table

| Aspect | NHiTS | TFT | TimesNet |
|--------|-------|-----|----------|
| **Framework** | Darts | Darts | NeuralForecast |
| **Core Mechanism** | Multi-scale hierarchical stacks | Attention + Variable Selection | 2D Convolutions on periodic tensors |
| **Input Handling** | Past + Future covariates (stacked) | Past & Future separate | All as Future covariates |
| **Architecture Style** | Feed-forward MLP stacks | Transformer (LSTM + Attention) | CNN (Inception-style) |
| **Temporal Modeling** | Hierarchical multi-rate sampling | Self-attention across time | 2D convolutions (time × period) |
| **Interpretability** | Medium (stack contributions) | High (attention weights, VSN) | Low (2D conv feature maps) |
| **Training Speed** | ⚡ Fast | 🐢 Slow | ⚡⚡ Very Fast |
| **Parameter Count** | Medium | High | Medium |
| **Memory Usage** | Low | High (attention O(n²)) | Medium |
| **Best For** | Efficient multi-scale forecasting | Complex dependencies, heterogeneous features | Strong periodic patterns |
| **Typical Epochs** | 100 | 100 | 150 (as max_steps) |
| **HPO Params** | num_stacks, num_blocks, layer_widths | hidden_size, lstm_layers, attention_heads | hidden_size, conv_hidden_size, top_k |

---

## Configuration Files

### Benchmarker Configuration

All models are configured in [benchmarker.py](../benchmarker.py):

```python
self.configs = {
    "NHITS_Q": {
        "type": "NHITS",
        "quantile": True,
        "n_epochs": 100,
        "best_params": nhits_best  # From HPO if available
    },
    "NHITS_MSE": {
        "type": "NHITS",
        "quantile": False,
        "n_epochs": 100,
        "best_params": None
    },
    "TIMESNET_Q": {
        "type": "TIMESNET",
        "quantile": True,
        "n_epochs": 150,
        "best_params": timesnet_best
    },
    "TIMESNET_MSE": {
        "type": "TIMESNET",
        "quantile": False,
        "n_epochs": 150,
        "best_params": None
    },
    "TFT_Q": {
        "type": "TFT",
        "quantile": True,
        "n_epochs": 100,
        "best_params": tft_best
    },
    "TFT_MSE": {
        "type": "TFT",
        "quantile": False,
        "n_epochs": 100,
        "best_params": None
    }
}
```

### HPO (Hyperparameter Optimization) Results

Optimized parameters are loaded from:
- `results/best_params_NHITS.json`
- `results/best_params_TIMESNET.json`
- `results/best_params_TFT.json`

If HPO results exist, they override default parameters:

```python
if "best_params" in config and config["best_params"] is not None:
    print("  Using optimized hyperparameters from HPO")
    # Override defaults with HPO results
    model_params.update(best_params)
else:
    print("  Using default hyperparameters (no HPO results found)")
```

### Walk-Forward Evaluation

All models use **walk-forward validation** for realistic evaluation:

```python
for day in range(n_predictions):  # e.g., 50 days
    prediction_start = test_start + day * 24h
    
    # Step 1: Use all data up to prediction_start
    historical_data = full_data[:prediction_start - 1h]
    
    # Step 2: Generate 24-hour forecast
    forecast = model.predict(n=24, series=historical_data, ...)
    
    # Step 3: Compare against actual observations
    actuals = full_data[prediction_start : prediction_start + 23h]
    
    # Step 4: Calculate errors
    errors[day] = actuals - forecast
    
    # Step 5: Move to next day (no retraining)
```

**Why Walk-Forward?**
- Simulates real-world deployment (predict one day, observe, repeat)
- More realistic than single long-horizon forecast
- Prevents look-ahead bias

---

## Prediction Output Format

All models produce the same output structure:

```python
predictions_df = pd.DataFrame({
    'timestamp': [...],     # Hourly timestamps
    'actual': [...],        # Ground truth consumption
    'p10': [...],          # 10th percentile (lower bound) [Q models only]
    'p50': [...],          # Median prediction (central forecast)
    'p90': [...]           # 90th percentile (upper bound) [Q models only]
})
```

**For MSE models:** `p10 = p50 = p90` (no uncertainty intervals)

### Evaluation Metrics

Calculated from predictions:

**Point Forecast Metrics:**
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **sMAPE:** Symmetric MAPE
- **WAPE:** Weighted Absolute Percentage Error
- **MASE:** Mean Absolute Scaled Error

**Probabilistic Forecast Metrics (Q models only):**
- **Pinball Loss:** Average quantile loss across p10, p50, p90
- **CRPS:** Continuous Ranked Probability Score
- **PICP:** Prediction Interval Coverage Probability (should be ~80%)
- **MIW:** Mean Interval Width (p90 - p10)
- **Winkler Score:** Penalizes both width and coverage violations

---

## Conformalized Quantile Regression (CQR)

Optional post-processing step for **calibrating prediction intervals**:

### Problem
Raw model quantiles (p10, p90) may not achieve nominal coverage (e.g., should contain 80% of actuals but only contains 65%).

### Solution
CQR adjusts intervals using a **calibration set**:

1. **Calibration Phase:**
   ```python
   # Get predictions on held-out calibration set
   cal_predictions = model.predict(calibration_data)
   
   # Compute non-conformity scores (how far off were the intervals?)
   scores = max(cal_predictions['p10'] - actuals, actuals - cal_predictions['p90'])
   
   # Find adjustment factor (quantile of scores)
   s_hat = np.quantile(scores, 1 - alpha)  # alpha = 0.2 for 80% coverage
   ```

2. **Adjustment Phase:**
   ```python
   # Widen test intervals by s_hat
   test_predictions['p10_adjusted'] = test_predictions['p10'] - s_hat
   test_predictions['p90_adjusted'] = test_predictions['p90'] + s_hat
   ```

**Result:** Intervals now have **guaranteed coverage** on exchangeable data.

**Usage in Benchmarker:**
```python
benchmarker.run(use_cqr=True, alpha=0.2)  # Enable CQR with 80% target coverage
```

---

## Summary: When to Use Each Model

### Use NHiTS When:
- ✅ Need fast training/inference
- ✅ Data has clear multi-scale patterns (hourly, daily, weekly)
- ✅ Limited computational resources
- ✅ Want efficient deployment

### Use TFT When:
- ✅ Have complex heterogeneous features
- ✅ Need interpretability (attention visualization)
- ✅ Long-range dependencies are important
- ✅ Accuracy is more important than speed
- ✅ Have sufficient data (TFT needs more data than NHiTS)

### Use TimesNet When:
- ✅ Strong periodic patterns exist (daily/weekly cycles)
- ✅ Need fast inference (production deployment)
- ✅ Want to try novel architecture
- ✅ Have forecasted weather/exogenous features

### Quantile vs MSE Variants:
- **Choose _Q (Quantile):** When uncertainty quantification is needed (risk assessment, capacity planning)
- **Choose _MSE:** When only point forecasts are needed (deterministic predictions, simpler models)

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| [benchmarker.py](../benchmarker.py) | Model training, evaluation, configuration |
| [model_preprocessing.py](../model_preprocessing.py) | Feature configuration, scaling, TimeSeries conversion |
| [conformal_calibration.py](../conformal_calibration.py) | CQR calibration implementation |
| `results/best_params_*.json` | HPO results (if available) |
| `models/{MODEL_NAME}.pt` | Trained model checkpoints |
| `models/{MODEL_NAME}_preprocessing_state.pkl` | Scaler states for inference |

---

**Document Version:** 1.0  
**Created:** January 11, 2026  
**Author:** GitHub Copilot

For questions or updates, refer to the codebase or update this document as models evolve.
