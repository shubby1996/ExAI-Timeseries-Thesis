# Interactive Plots - Quick Reference Guide

## How to View the Plots

1. **Open in Browser**:
   ```bash
   # From HPC shell
   firefox interp_out/mu_importance_by_horizon_per_cov.html &
   
   # Or download and open locally
   scp user@hpc:interp_out/*.html ./local/folder/
   ```

2. **Interactive Features** (in any plot):
   - **Hover**: Mouse over any point to see exact values
   - **Zoom**: Click and drag to zoom into a region
   - **Reset**: Double-click to zoom back out
   - **Pan**: Hold Shift and drag while zoomed
   - **Toggle**: Click legend items to show/hide traces
   - **Export**: Camera icon in toolbar → "Download plot as png"

---

## Plot Categories

### 🌍 Category 0: Full Dataset Forecast Visualization

**Purpose**: See the complete forecast timeline across train, validation, and test sets with proper dates and units.

**Command to generate**:
```bash
module load python/3.12-conda
conda activate env-nam

python3 interpret_tsnamlss.py \
    --csv_path nordbyen_features_engineered.csv \
    --ckpt my_model_cov12.pt \
    --plot_full_dataset \
    --out_dir interp_out_cov12_full
```

| File | Size | What It Shows | Key Features |
|------|------|---------------|-------------|
| `full_dataset_forecasts_1step.html` | 12 MB | Complete timeline: 48,001 1-step ahead forecasts from 2015-05-02 to 2020-11-14 | • Ground truth (black) vs forecast (red)<br>• 80% prediction interval (shaded red)<br>• Train/Val/Test split markers<br>• MAE per split in title<br>• X-axis shows actual dates<br>• Y-axis in kW units |
| `full_dataset_forecasts_multistep_test.html` | 5.1 MB | Test set 24-hour forecast trajectories (~100 sampled paths) | • Shows forecast degradation over 24h<br>• Ground truth vs forecast trajectories<br>• 80% PI per trajectory<br>• Useful for multi-step behavior analysis |

**Interactive features**:
- **Zoom into dates**: Click and drag to zoom into specific time periods (e.g., winter 2018)
- **Hover for exact values**: See date, ground truth, forecast, and PI bounds
- **Toggle traces**: Click legend to hide/show ground truth, forecast, or PI
- **Split navigation**: Use vertical blue/orange markers to identify train/val/test boundaries
- **Export**: Camera icon to save zoomed view as PNG

**Interpretation guide**:

1. **full_dataset_forecasts_1step.html**:
   - **Black line**: Actual heat consumption (ground truth)
   - **Red line**: Model's 1-step ahead forecast (μ)
   - **Red shaded area**: 80% prediction interval (10th to 90th percentile)
   - **Vertical dashed lines**: Split boundaries
     - Blue: Train/Val boundary
     - Orange: Val/Test boundary
   - **Title metrics**: Shows MAE (Mean Absolute Error in kW) for each split
   
   **What to look for**:
   - Red line should closely follow black line (good tracking)
   - Most black line should stay within red shaded area (good calibration)
   - Check if performance degrades in certain seasons or periods
   - Compare visual fit across train/val/test splits

2. **full_dataset_forecasts_multistep_test.html**:
   - **Green lines**: Ground truth for each 24h forecast window
   - **Red lines**: Model forecasts for each 24h window
   - **Light red shading**: 80% PI for each forecast
   - Each trajectory shows how forecast evolves over 24 hours
   
   **What to look for**:
   - Forecast uncertainty (shaded area) typically widens at longer horizons
   - Check if model captures daily patterns over full 24h
   - Identify if certain forecast horizons have systematic errors
   - Compare early-horizon (h=1-6) vs late-horizon (h=18-24) accuracy

**Typical use cases**:
- "Show me the full forecast timeline with actual dates"
- "How does the model perform in winter vs summer?"
- "Identify time periods where forecast errors are large"
- "Visual verification of train/val/test split quality"
- "Share comprehensive forecast visualization with stakeholders"

**Performance**: Generates 48,001 forecasts in ~1 minute using batched inference (batch_size=512)

---

### 📊 Category 1: Per-Sample Decomposition (Sample #100)

**Purpose**: Understand what each covariate contributes to the forecast for a single observation.

| File | What It Shows | Key Insight |
|------|---------------|------------|
| `decomp_mu_sample100.html` | Mean predictions (μ) from each covariate over 24h | Which covariates drive the level of the forecast |
| `decomp_rawsig_sample100.html` | Uncertainty (rawσ) from each covariate over 24h | Which covariates drive prediction uncertainty |
| `stack_mu_sample100.html` | Stacked area chart of μ contributions | How the 8 covariates combine to make the prediction |
| `stack_rawsig_sample100.html` | Stacked area chart of rawσ contributions | How uncertainty compounds over horizon |

**Typical Use**: "Why did the model predict 5.2 kW for hour 23?" → Hover to decompose.

---

### 🎯 Category 2: Forecast with Prediction Intervals

**Purpose**: See the model's forecast and confidence bounds.

| File | What It Shows |
|------|---------------|
| `forecast_pi_sample100.html` | 24-hour ahead forecast with 80% prediction interval (shaded blue area) |
| `history168_forecast24_pi_sample100.html` | Full context: 168h history + 24h forecast with PI and actual values (if available) |

**Red line** = Actual (if available)  
**Blue line & shaded area** = Model forecast ± 80% PI

---

### 📈 Category 3: Diagnostic Plots

**Purpose**: Check model calibration and residual behavior.

| File | What It Shows |
|------|---------------|
| `sigma_sample100.html` | Predicted uncertainty (σ) over 168h + 24h window |
| `zscore_sample100.html` | Standardized residuals (should be ~Normal(0,1)) with ±1.96 reference lines |

**zscore_sample100**: If points stay within ±2 bands, model is well-calibrated.

---

### 🔍 Category 4: Occlusion Analysis (Single Sample)

**Purpose**: What happens if we "remove" each lag from the input?

**Format**: 84 lags (168h ÷ 2h per column) × 2 feature types = heatmaps

| File | Metric | Feature Type |
|------|--------|--------------|
| `occ_abs_dmu_endo_sample100.html` | Change in mean (Δμ) | Endogenous (past heat) |
| `occ_abs_dmu_exo_sample100.html` | Change in mean (Δμ) | Exogenous (weather) |
| `occ_abs_drawsig_endo_sample100.html` | Change in uncertainty (Δrawσ) | Endogenous |
| `occ_abs_drawsig_exo_sample100.html` | Change in uncertainty (Δrawσ) | Exogenous |

**Interpretation**:
- Bright colors = That lag is important
- Lag 1 (most recent) usually bright
- Lag 24 (daily cycle) often bright
- Lag 168 (weekly cycle) sometimes bright

---

### 🌟 Category 5: Test-Set Importance Analysis (All 7,096 Test Samples)

**Purpose**: Aggregate importance of each covariate across all test samples and all forecast hours.

**3 Normalizations Provided**:

#### Option A: μ Importance (Primary Recommendation)

| File | What It Shows |
|------|---------------|
| `mu_importance_by_horizon.html` | Stream-level: How important is each stream type? |
| `mu_importance_by_horizon_per_cov.html` | **Per-covariate**: Bar plot for each of 8 covariates (0-1 normalized) |

**Read this plot**: For each covariate (color), see how its importance changes across forecast hours (x-axis).

**Example**: "endo_heat_lag_1h" stays high across all hours, but "future_hour_sin" only matters in early hours.

#### Option B: rawσ Importance (Normalized by y)

| File | What It Shows |
|------|---------------|
| `rawsig_importance_by_horizon_norm_y.html` | Stream-level: How much uncertainty does each stream contribute? |
| `rawsig_importance_by_horizon_per_cov_norm_y.html` | Per-covariate: Uncertainty attribution per feature |

#### Option C: rawσ Importance (Internal Normalization)

| File | What It Shows |
|------|---------------|
| `rawsig_importance_by_horizon_norm_rawsig.html` | Stream-level: Alternative uncertainty normalization |
| `rawsig_importance_by_horizon_per_cov_norm_rawsig.html` | Per-covariate: Alternative uncertainty attribution |

**When to use each**:
- **Option A (μ normalized by y)**: Most interpretable; shows which features matter for accuracy
- **Option B (rawσ norm by y)**: How much does each feature contribute to prediction uncertainty?
- **Option C (rawσ norm rawσ)**: Techincal normalization; similar to Option B but different scale

---

## Example Workflow

### Q: "Is temperature more important than humidity?"

1. Open `mu_importance_by_horizon_per_cov.html`
2. Hover over bars for `exo_temp` (blue) vs `exo_dew_point` (purple)
3. Check values: If temp bar is ~0.13 and dew_point is ~0.08, temperature is more important
4. Compare across horizons (x-axis): Is this true at all forecast hours?

### Q: "When does the model become uncertain?"

1. Open `sigma_sample100.html` 
2. Look for peaks in the purple σ line
3. Compare with `stack_rawsig_sample100.html` - which covariate contributes to peaks?

### Q: "Does removing past heat values hurt predictions?"

1. Open `occ_abs_dmu_endo_sample100.html`
2. Look for bright horizontal stripes (high occlusion effect)
3. Check which lags are brightest: recent (lag 1-6)? daily (lag 24-26)? weekly (lag 168)?

### Q: "Can I see the full forecast timeline with dates and units?"

1. Generate full dataset plots:
   ```bash
   module load python/3.12-conda
   conda activate env-nam
   python3 interpret_tsnamlss.py \
       --csv_path nordbyen_features_engineered.csv \
       --ckpt my_model_cov12.pt \
       --plot_full_dataset \
       --out_dir interp_out_full
   ```
2. Open `interp_out_full/full_dataset_forecasts_1step.html`
3. Zoom into specific date ranges to inspect forecast quality
4. Check MAE metrics in title for each split
5. Open `full_dataset_forecasts_multistep_test.html` to see 24h forecast behavior

---

## Covariate Legend

**Endogenous** (feed in their own past):
- `endo_heat_lag_1h` - Heat consumption 1 hour ago
- `endo_heat_lag_24h` - Heat consumption 24 hours ago (daily cycle)

**Exogenous** (external inputs):
- `exo_temp` - Outdoor temperature (°C)
- `exo_wind_speed` - Wind speed (m/s)
- `exo_dew_point` - Dew point temperature (°C)
- `exo_temp_squared` - Temperature squared (captures nonlinearity)

**Future Covariates** (known 24h ahead):
- `future_hour_sin` - sin(2π × hour / 24) for hour-of-day seasonality
- `future_hour_cos` - cos(2π × hour / 24) for hour-of-day seasonality

---

## File Size Notes

- Each HTML file: **4-9 MB** (vs ~50-100 KB static PNG)
- Trade-off: Larger file size → Full interactivity (hover, zoom, legends)
- All plots open instantly in browser (no server needed)
- Suitable for local storage and emailing

---

## Tips for Exploration

1. **Start with**: `mu_importance_by_horizon_per_cov.html` - Best overview
2. **Then check**: `history168_forecast24_pi_sample100.html` - See model in action
3. **If curious**: `occ_abs_dmu_exo_sample100.html` - Which weather lags matter?
4. **For uncertainty**: `rawsig_importance_by_horizon_per_cov_norm_y.html` - Which features make model uncertain?

---

## Browser Tips

- **Chrome/Edge**: Fastest performance, camera icon export works natively
- **Firefox**: Full support, export via camera icon
- **Safari**: Full support
- **Mobile**: Works but touch gestures may differ; desktop recommended for detailed exploration

---

**Generated**: 18 interactive HTML plots  
**Total Size**: ~130 MB  
**Location**: `/home/hpc/iwi5/iwi5389h/ExAI-TimeSeries-Additive-Interpretability/interp_out/`
