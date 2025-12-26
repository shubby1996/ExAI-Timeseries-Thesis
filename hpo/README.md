# Hyperparameter Optimization (HPO)

This folder contains scripts for tuning hyperparameters of NHITS and TimesNet models using Optuna.

## Files

- **`hpo_tuner.py`** - Core HPO script that runs Optuna trials
- **`hpo_runner.ipynb`** - Jupyter notebook for submitting and monitoring HPO jobs
- **`hpo_job.slurm`** - SLURM job script for running HPO on HPC cluster
- **`hpo_current_jobs.json`** - Tracks active HPO jobs (auto-generated)

## Usage

### Option 1: Using Jupyter Notebook (Recommended)

Open `hpo_runner.ipynb` from the **project root** directory:

```bash
cd /path/to/ExAI-Timeseries-Thesis
jupyter notebook hpo/hpo_runner.ipynb
```

The notebook provides:
1. Configuration (models, trials, SLURM settings)
2. Job submission
3. Status monitoring
4. Results viewing

### Option 2: Direct SLURM Submission

From the **hpo/** directory:

```bash
cd hpo
sbatch hpo_job.slurm NHITS 50
sbatch hpo_job.slurm TIMESNET 50
```

From the **project root**:

```bash
sbatch hpo/hpo_job.slurm NHITS 50
```

### Option 3: Direct Python Execution (Local Testing)

From the **project root**:

```bash
python hpo/hpo_tuner.py NHITS 10
```

## How It Works

1. **HPO Process**:
   - Loads data from `processing/nordbyen_processing/nordbyen_features_engineered.csv`
   - Runs Optuna trials to optimize hyperparameters
   - Trains models with different hyperparameter combinations
   - Evaluates on validation set
   - Saves best parameters to `results/best_params_<MODEL>.json`

2. **Optimized Parameters**:
   - **NHITS**: `num_stacks`, `num_blocks`, `num_layers`, `layer_widths`, `dropout`, `learning_rate`
   - **TimesNet**: `top_k`, `d_model`, `d_ff`, `num_kernels`, `dropout`, `learning_rate`

3. **Results Location**:
   - Best params: `results/best_params_NHITS.json`, `results/best_params_TIMESNET.json`
   - These are automatically loaded by `benchmarker.py` during full benchmarking

## Configuration

Edit parameters in `hpo_runner.ipynb` or directly in `hpo_tuner.py`:

```python
N_TRIALS = 50          # Number of optimization trials
PARTITION = 'rtx3080' # GPU partition
TIME_LIMIT = '10:00:00' # 10 hours
```

## Expected Runtime

- **NHITS**: ~30-60 minutes for 50 trials (GPU)
- **TimesNet**: ~60-90 minutes for 50 trials (GPU)

## Integration with Benchmarking

After HPO completes, the benchmarker automatically uses optimized parameters:

```bash
cd ../water_centrum_benchmark/scripts
sbatch benchmark_water_job.slurm
```

The benchmarker checks for `results/best_params_*.json` and uses those parameters if available.
