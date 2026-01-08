# Benchmark Comparison Tool

## Overview
`benchmark_runner.ipynb` is a comprehensive tool for comparing and analyzing benchmark results across different datasets and models.

## Features

### 1. **Data Loading & Summary**
- Loads all benchmark history from `results/benchmark_history.csv`
- Shows summary statistics by dataset and model
- Displays date ranges and entry counts

### 2. **Overall Performance Comparison**
- Latest results table for all model-dataset combinations
- 6-panel visualization comparing all metrics (MAE, RMSE, MAPE, CRPS, MIW, PICP)
- Side-by-side heat vs water performance

### 3. **Best Performers Analysis**
- Identifies best models by metric
- Overall winners and dataset-specific winners
- Special analysis for PICP calibration (target: 80%)

### 4. **Model Architecture Comparison**
- NHITS vs TIMESNET performance across datasets
- Quantitative comparison with percentage differences
- Dataset-dependent performance insights

### 5. **Loss Function Analysis**
- Quantile vs MSE comparison for each architecture
- Point forecast accuracy comparison (MAE, RMSE, MAPE)
- Insights on when to use each loss function

### 6. **Historical Trends**
- MAE trends over time for all models
- Improvement tracking from first to latest run
- Visual trend plots by dataset

### 7. **Detailed Comparison Table**
- Complete results table with all metrics
- Exported to CSV for external analysis
- Sortable by any metric

### 8. **Key Insights & Recommendations**
- Automated insight generation
- Performance summaries
- Actionable recommendations for future work

## Usage

### Running the Notebook
```bash
jupyter notebook benchmark_runner.ipynb
```

Or use VS Code's Jupyter extension.

### Cell Execution Order
Run cells sequentially from top to bottom:
1. Setup and imports
2. Data loading
3. Analysis sections (2-8)

### Output Files
- `results/benchmark_comparison_all.png` - Main comparison visualization
- `results/benchmark_trends_mae.png` - Historical trends plot
- `results/benchmark_comparison_latest.csv` - Latest results export

## Key Metrics Explained

### Point Forecast Metrics
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

### Probabilistic Forecast Metrics
- **PICP**: Prediction Interval Coverage Probability (target: 80%)
- **MIW**: Mean Interval Width (lower is better, but not at expense of PICP)
- **CRPS**: Continuous Ranked Probability Score (lower is better)

### Metadata
- **n_epochs**: Training epochs used
- **has_hpo**: Whether hyperparameter optimization was used

## Current Findings (Dec 26, 2025)

### Best Models
- **Heat (Nordbyen)**: NHITS_Q (MAE: 0.194)
- **Water (Centrum)**: TIMESNET_Q (MAE: 0.003)

### Key Insights
1. Quantile loss consistently outperforms MSE
2. TIMESNET excels on water data
3. NHITS provides great heat forecasts with faster training
4. PICP calibration needs improvement (currently ~38% vs target 80%)

### Recommendations
1. Use Quantile models for production
2. Focus on PICP calibration
3. Continue HPO for MSE variants
4. Monitor trends for sustained improvement

## Maintenance

### Adding New Benchmarks
New benchmark runs automatically appear in the notebook as they're saved to `benchmark_history.csv` with the dataset column.

### Customization
- Modify visualization styles in section 1 (matplotlib/seaborn settings)
- Add new metrics by updating the `metrics` list in relevant cells
- Customize insights logic in section 8

## Dependencies
- pandas, numpy
- matplotlib, seaborn
- IPython.display
- All installed via `requirements.txt`
