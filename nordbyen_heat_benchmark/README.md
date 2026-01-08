# Nordbyen Heat Benchmarking

This directory contains all files related to heat consumption forecasting for the Nordbyen DMA.

## Directory Structure

- **data/** - Processed data files
- **scripts/** - Data processing and benchmarking scripts
- **notebooks/** - Jupyter notebooks for exploration
- **results/** - Benchmarking results (created during runs)

## Workflow

1. Data exploration: Use notebooks for analysis
2. Benchmarking: `run_benchmarker.py` or submit `benchmark_job.slurm`

## Data Info

- Source: nordbyen_processing/nordbyen_heat_weather_aligned.csv
- Features: 26 columns (weather, temporal, lags, interactions)
- Target: heat
- Train/Val/Test: 2018 / 2019 / 2020

## Usage

From the project root:
```bash
sbatch nordbyen_heat_benchmark/scripts/benchmark_job.slurm
```
