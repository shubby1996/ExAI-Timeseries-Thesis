# Water Centrum Benchmark

This directory contains all files related to water consumption benchmarking for the Centrum DMA.

## Directory Structure

- **data/**: Raw water consumption data files
- **processing/**: Data processing and feature engineering scripts and outputs
- **scripts/**: Benchmark execution scripts
- **results/**: Benchmark results and visualizations (generated after running)
- **docs/**: Documentation files
- **water_data_exploration.ipynb**: Interactive data exploration notebook

## Key Files

### Data Files
- `processing/centrum_features_engineered_from_2018-04-01.csv` - Main dataset for benchmarking (Apr 2018 - Nov 2020)

### Scripts
- `scripts/run_benchmarker_water.py` - Main benchmarking script
- `scripts/benchmark_water_job.slurm` - SLURM job submission script
- `scripts/check_water_split.py` - Analyze train/val/test split
- `scripts/engineer_filtered_water.py` - Feature engineering pipeline

## Usage

### Run Benchmark
From the main project directory:
```bash
sbatch water_centrum_benchmark/scripts/benchmark_water_job.slurm
```

Or run specific models:
```bash
sbatch water_centrum_benchmark/scripts/benchmark_water_job.slurm "NHITS_Q TIMESNET_Q"
```

### Data Exploration
```bash
jupyter notebook water_centrum_benchmark/water_data_exploration.ipynb
```

## Data Information

- **Period**: April 2, 2018 to November 14, 2020
- **Records**: 22,989 hourly observations
- **Features**: 26 (including weather, temporal, lags, interactions)
- **Target**: water_consumption (m³/h)
- **Train/Val/Test**: 2018 / 2019 / 2020

