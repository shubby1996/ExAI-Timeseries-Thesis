# Water Tommerby Benchmark

This directory contains all files related to water consumption benchmarking for the Tommerby DMA.

## Directory Structure

- **scripts/**: Benchmark execution scripts
- **results/**: Benchmark results and visualizations (generated after running)
- **notebooks/**: Jupyter notebooks for data exploration and analysis

## Key Files

### Data Files
- `../processing/tommerby_processing/tommerby_features_engineered.csv` - Main dataset for benchmarking (processed from raw water data)

### Scripts
- `scripts/run_benchmarker_water.py` - Main benchmarking script
- `scripts/benchmark_water_job.slurm` - SLURM job submission script

## Usage

### Step 1: Generate Feature-Engineered Data
Before running the benchmark, you need to process the raw data:
```bash
cd processing/tommerby_processing
python build_features_tommerby.py
```

This will create `tommerby_features_engineered.csv` which includes:
- Water consumption data from raw_data/water_b_tommerby
- Weather features from raw_data/weather
- Time-based features (hour, day, month, weekend flags)
- Lag features and rolling averages
- Holiday features (public and school holidays)
- Interaction features (temperature, wind, weekend)

### Step 2: Run Benchmark
From the project root directory:
```bash
# Submit job with all default models (NHITS_Q, NHITS_MSE, TIMESNET_Q, TIMESNET_MSE, TFT_Q, TFT_MSE)
sbatch water_tommerby_benchmark/scripts/benchmark_water_job.slurm

# Or specify specific models
sbatch water_tommerby_benchmark/scripts/benchmark_water_job.slurm "NHITS_Q TIMESNET_Q"
```

### Step 3: Monitor Progress
Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f water_tommerby_benchmark/scripts/benchmark_water_tommerby_<JOB_ID>.log
```

### Step 4: Analyze Results
Results will be saved to `water_tommerby_benchmark/results/`:
- `benchmark_results_*_Water_Tommerby_<JOB_ID>.csv` - Model metrics
- `*_predictions_<JOB_ID>.csv` - Model predictions
- `benchmark_history.csv` - Historical results across runs

## Models

The benchmark supports the following models:
- **NHITS_Q**: NHITS with Quantile loss (probabilistic forecasting)
- **NHITS_MSE**: NHITS with MSE loss (point forecasting)
- **TIMESNET_Q**: TIMESNET with Quantile loss
- **TIMESNET_MSE**: TIMESNET with MSE loss
- **TFT_Q**: Temporal Fusion Transformer with Quantile loss
- **TFT_MSE**: Temporal Fusion Transformer with MSE loss

## Data Split

The data is split into:
- **Training**: 2018 data
- **Validation**: 2019 data
- **Test**: 2020 data

## Target Variable

- `water_consumption`: Hourly water consumption in the Tommerby DMA

## Features

### Weather Features
- Temperature, dew point, humidity
- Wind speed, cloud coverage
- Rain and snow precipitation
- Atmospheric pressure

### Time Features
- Hour (cyclical: sin/cos encoding)
- Day of week, month, season
- Weekend indicator

### Lag Features
- 1-hour lag
- 24-hour lag
- 24-hour rolling average

### Interaction Features
- Temperature squared
- Temperature × wind speed
- Temperature × weekend

### Holiday Features
- Danish public holidays
- School holidays (if available)
