# Interactive Plotly Visualizations Guide

## Overview

The benchmark notebooks now generate **interactive Plotly visualizations** in addition to static matplotlib plots. These interactive plots provide enhanced exploration capabilities with hover tooltips, zoom/pan, and more.

## How to Use the Interactive Plots

### Method 1: View in Jupyter Notebook (Recommended)

The interactive plots are **embedded directly in the notebooks** and will display automatically when you run the cells.

**Steps:**
1. Run cell "6. Generate Visualizations" to generate all plots
2. Scroll down to sections:
   - **6.4.1** - Interactive Time Series (Full Test Set)
   - **6.4.2** - Interactive Error Distribution
   - **6.4.3** - Interactive Scatter Plots
   - **6.4.4** - Interactive Metrics Comparison

3. The plots will render interactively within the notebook

**Notebook Features:**
- ✅ Hover tooltips showing exact values
- ✅ Zoom: Click and drag to zoom into regions
- ✅ Pan: Shift + drag to pan across the plot
- ✅ Reset: Double-click to reset zoom/pan
- ✅ Legend: Click legend items to toggle traces on/off
- ✅ Save: Use the camera icon in the toolbar to download as PNG

### Method 2: Download and Open in Browser

If the notebook embedding doesn't work or you prefer standalone files:

**Steps:**
1. Locate the `interactive_plots/` subdirectory in your results folder
   - Heat: `nordbyen_heat_benchmark/results/interactive_plots/`
   - Water (Centrum): `water_centrum_benchmark/results/interactive_plots/`
   - Water (Tommerby): `water_tommerby_benchmark/results/interactive_plots/`

2. Download the desired HTML files to your computer

3. Open them with any web browser (double-click or right-click → Open With)

**Available Files:**
```
interactive_plots/
├── {model_name}_timeseries.html         # Full test set time series
├── {model_name}_error_distribution.html # Error distributions
├── {model_name}_scatter.html            # Actual vs predicted
└── interactive_metrics_comparison.html  # All models' metrics
```

### Method 3: Command Line

Open plots directly from terminal:

```bash
# For Linux/WSL
xdg-open ~/path/to/interactive_plots/{model_name}_timeseries.html

# For macOS
open ~/path/to/interactive_plots/{model_name}_timeseries.html

# For Windows
start %USERPROFILE%\path\to\interactive_plots\{model_name}_timeseries.html
```

## Plot Types

### 1. Interactive Time Series (`*_timeseries.html`)
**What it shows:** Full test set forecast with confidence intervals

**Features:**
- Actual values (black line)
- Predicted median (blue line)
- 80% confidence interval (blue shaded area)
- Full hover information with timestamps and values

**Use case:** See overall forecast performance across entire test period

### 2. Interactive Error Distribution (`*_error_distribution.html`)
**What it shows:** Side-by-side histograms of prediction errors

**Features:**
- Absolute error distribution (left)
- Percentage error distribution (right)
- Interactive bin selection
- Summary statistics on hover

**Use case:** Understand error patterns and distribution

### 3. Interactive Scatter Plot (`*_scatter.html`)
**What it shows:** Actual vs predicted values with error magnitude

**Features:**
- Color scale representing absolute error (yellow=low, purple=high)
- Perfect prediction reference line (red dashed)
- Individual point details on hover
- Identify outliers easily

**Use case:** Detect systematic prediction biases and outliers

### 4. Interactive Metrics Comparison (`interactive_metrics_comparison.html`)
**What it shows:** Bar charts comparing 6 point forecast metrics across all models

**Features:**
- Side-by-side model comparison
- Metrics: MAE, RMSE, MAPE, sMAPE, WAPE, MASE
- Hover to see exact values
- Zoom into specific metrics

**Use case:** Quick visual comparison of model performance

## Common Interactions

| Action | How To |
|--------|--------|
| **Zoom in** | Click and drag on the plot area |
| **Zoom to region** | Click and drag to select a region |
| **Pan** | Hold Shift and drag |
| **Reset view** | Double-click anywhere on the plot |
| **Toggle legend** | Click on legend item name |
| **Show/hide traces** | Click legend items |
| **Save as PNG** | Click camera icon in toolbar |
| **View exact value** | Hover over data point |
| **Download data** | Use "Download plot as PNG" or save as HTML |

## File Locations

**Heat Benchmark Results:**
- All results: `nordbyen_heat_benchmark/results/`
- Static plots: `nordbyen_heat_benchmark/results/*.png`
- Interactive plots: `nordbyen_heat_benchmark/results/interactive_plots/*.html`

**Water Centrum Benchmark Results:**
- All results: `water_centrum_benchmark/results/`
- Static plots: `water_centrum_benchmark/results/*.png`
- Interactive plots: `water_centrum_benchmark/results/interactive_plots/*.html`

**Water Tommerby Benchmark Results:**
- All results: `water_tommerby_benchmark/results/`
- Static plots: `water_tommerby_benchmark/results/*.png`
- Interactive plots: `water_tommerby_benchmark/results/interactive_plots/*.html`

## Troubleshooting

### Plots not displaying in notebook?
- Try refreshing the page (Ctrl+Shift+R)
- Check browser console for JavaScript errors
- Ensure you have a recent browser version
- Try downloading the HTML file and opening it directly

### IFrame not loading?
- Check file path: `print(os.path.join(RESULTS_DIR, "interactive_plots"))`
- Ensure files exist in the directory
- Try absolute path instead of relative

### Performance issues with large datasets?
- Interactive plots use all data points - may be slow with very large datasets
- Try zooming into specific time ranges instead
- Download the PNG version if interactive exploration isn't needed

### Can't open HTML files?
- Ensure file has `.html` extension
- Right-click → Open With → Browser
- Or: `start filename.html` (Windows) or `open filename.html` (Mac)

## Additional Resources

- Plotly documentation: https://plotly.com/python/
- Interactive features: https://plotly.com/python/interactive-charts/
- Hover text customization: https://plotly.com/python/hover-text-and-formatting/

## Technical Details

**Generated With:**
- Plotly 5.x
- Python pandas for data handling
- Jupyter IFrame for notebook embedding

**File Size:**
- Typical plot: 100-500 KB (HTML)
- Full metrics comparison: 50-200 KB

**Browser Compatibility:**
- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅
- Internet Explorer ❌ (use modern browser)
