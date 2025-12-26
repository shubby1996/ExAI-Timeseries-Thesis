# Benchmark Organization Structure

This document describes the clean separation between **data processing** and **benchmarking**.

## Directory Structure

```
ExAI-Timeseries-Thesis/
│
├── processing/                          # ALL DATA PROCESSING
│   ├── centrum_processing/              # Water data processing
│   │   ├── centrum_water_weather_aligned.csv
│   │   ├── centrum_features_engineered_from_2018-04-01.csv
│   │   ├── align_data_centrum.py
│   │   ├── feature_engineering_centrum.py
│   │   └── CENTRUM_FEATURE_ENGINEERING_DOCUMENTATION.md
│   │
│   └── nordbyen_processing/             # Heat data processing
│       ├── nordbyen_heat_weather_aligned.csv
│       ├── nordbyen_features_engineered.csv
│       ├── align_data_nordbyen.py
│       ├── feature_engineering_nordbyen.py
│       └── NORDBYEN_FEATURE_ENGINEERING_DOCUMENTATION.md
│
├── water_centrum_benchmark/             # WATER BENCHMARKING ONLY
│   ├── scripts/
│   │   ├── run_benchmarker_water.py     # References ../../processing/centrum_processing/
│   │   └── benchmark_water_job.slurm
│   ├── notebooks/
│   │   └── water_data_exploration.ipynb
│   └── results/
│
├── nordbyen_heat_benchmark/             # HEAT BENCHMARKING ONLY
│   ├── scripts/
│   │   ├── run_benchmarker.py           # References ../../processing/nordbyen_processing/
│   │   └── benchmark_job.slurm
│   ├── notebooks/
│   └── results/
│
├── benchmarker.py                       # Shared benchmark infrastructure
└── model_preprocessing.py               # Shared preprocessing utilities
```

## Key Principles

### 1. Processing Folder (`processing/`)
**Purpose**: ALL data-related work from raw → engineered features

**Contains**:
- Raw data files
- Aligned data (water/weather merged)
- Engineered features (final datasets)
- Processing scripts (alignment, feature engineering)
- Data documentation

### 2. Benchmark Folders (`*_benchmark/`)
**Purpose**: Model training, evaluation, comparison

**Contains**:
- Benchmark runner scripts
- SLURM job files
- Exploration notebooks
- Results (metrics, predictions, plots)

**Does NOT contain**: Data files (references `processing/` instead)

## Usage

### Running Water Benchmark

**Option 1: SLURM (Recommended for full benchmarking)**
```bash
# From project root
cd water_centrum_benchmark/scripts
sbatch benchmark_water_job.slurm
```

**Option 2: Local execution (for testing)**
```bash
# From project root
cd water_centrum_benchmark/scripts
python run_benchmarker_water.py --models NHITS_Q TIMESNET_Q
```

**What it does:**
- Loads data from: `../../processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv`
- Trains models: NHITS_Q, NHITS_MSE, TIMESNET_Q, TIMESNET_MSE (or specify with --models)
- Saves results to: `../results/`
- Saves predictions to: `../results/*_predictions.csv`
- Training time: ~2-3 hours on GPU for all 4 models

### Running Heat Benchmark

**Option 1: SLURM (Recommended for full benchmarking)**
```bash
# From project root
cd nordbyen_heat_benchmark/scripts
sbatch benchmark_job.slurm
```

**Option 2: Local execution (for testing)**
```bash
# From project root
cd nordbyen_heat_benchmark/scripts
python run_benchmarker.py --models NHITS_Q TIMESNET_Q
```

**What it does:**
- Loads data from: `../../processing/nordbyen_processing/nordbyen_features_engineered.csv`
- Trains models: NHITS_Q, NHITS_MSE, TIMESNET_Q, TIMESNET_MSE (or specify with --models)
- Saves results to: `../results/`
- Saves predictions to: `../results/*_predictions.csv`
- Training time: ~2-3 hours on GPU for all 4 models

### Model Variants Available

Both benchmarks support these models:
- **NHITS_Q**: NHITS with Quantile Regression (probabilistic, 100 epochs)
- **NHITS_MSE**: NHITS with MSE loss (point forecast, 100 epochs)
- **TIMESNET_Q**: TimesNet with Multi-Quantile Loss (probabilistic, 150 epochs)
- **TIMESNET_MSE**: TimesNet with MSE loss (point forecast, 150 epochs)

### Checking Results

```bash
# View benchmark results
cat water_centrum_benchmark/results/benchmark_results.csv
cat nordbyen_heat_benchmark/results/benchmark_results.csv

# Generate visualizations
cd <project_root>
python visualize_benchmark.py
```

### Data Processing (if you need to regenerate features)

**Water data:**
```bash
cd processing/centrum_processing
python feature_engineering_centrum.py
```

**Heat data:**
```bash
cd processing/nordbyen_processing
python feature_engineering_nordbyen.py
```

## Key Differences Between Datasets

| Aspect | Water (Centrum) | Heat (Nordbyen) |
|--------|----------------|-----------------|
| **Data File** | centrum_features_engineered_from_2018-04-01.csv | nordbyen_features_engineered.csv |
| **Target Variable** | water_consumption | heat_consumption |
| **Records** | 22,989 (filtered from 2018-04-01) | 47,854 (full dataset) |
| **Date Range** | 2018-04-01 to 2020-11-14 | Full historical range |
| **Train/Val/Test** | 2018 / 2019 / 2020 | 2018 / 2019 / 2020 |
| **Features** | 26 features | Similar feature set |
| **Data Quality** | Early data filtered due to zeros | Full dataset used |

## Testing Before Full Benchmarking

**RECOMMENDED**: Test the benchmarker with minimal epochs before submitting full SLURM jobs:

### Test Water Benchmarker (8-10 minutes)
```bash
python test_benchmarker_quick_water.py
```

### Test Heat Benchmarker (10-12 minutes)
```bash
python test_benchmarker_quick.py
```

Both tests run all 4 model variants with reduced epochs to validate the setup works correctly.

## How Data Detection Works

The benchmarker **automatically detects** whether you're running water or heat data:
- Detects "water" or "centrum" in file path → uses `water_feature_config()` 
- Detects "heat" or "nordbyen" in file path → uses `default_feature_config()`
- Automatically adjusts target variable (water_consumption vs heat_consumption)
- Automatically adjusts lag features (water_lag_* vs heat_lag_*)

No manual configuration needed - just point to the correct CSV file!
