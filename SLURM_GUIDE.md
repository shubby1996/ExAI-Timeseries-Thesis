# Benchmarker SLURM Integration

## Quick Start Guide

### Option 1: SLURM Execution (Recommended)

**Step 1: Submit the job**
```bash
sbatch benchmark_job.slurm
```

**Step 2: Monitor job status**
```bash
squeue -u $USER
```

**Step 3: Check logs**
```bash
tail -f benchmark_<JOB_ID>.log
```

**Step 4: Once complete, continue with notebook**
- Run cells 8+ in `benchmark_runner.ipynb` for visualization and analysis

### Option 2: Using the Notebook

1. **Open**: `benchmark_runner.ipynb`
2. **Run cells 1-6**: Setup and configuration
3. **Run cell 7A**: Submit SLURM job
4. **Run cell 7B**: Monitor job status (re-run until complete)
5. **Run cells 8+**: View results and visualizations

### Option 3: Local Execution

- Run cell 7C in the notebook (takes 10-15 minutes, may timeout)

## Files Created

- **`benchmark_job.slurm`**: SLURM batch script for HPC submission
- **`run_benchmarker.py`**: Standalone Python script that runs the benchmarker
- **`benchmark_runner.ipynb`**: Interactive notebook (updated with SLURM integration)

## SLURM Configuration

The job is configured with:
- **Time limit**: 2 hours
- **GPU**: 1x RTX 3080
- **CPUs**: 8 cores
- **Partition**: rtx3080

To modify, edit `benchmark_job.slurm` and adjust the `#SBATCH` directives.

## Output Files

- **`benchmark_<JOB_ID>.log`**: Standard output log
- **`benchmark_<JOB_ID>.err`**: Error log
- **`results/benchmark_results.csv`**: Final metrics comparison
- **`results/*.csv`**: Individual model predictions
- **`models/`**: Trained model checkpoints

## Troubleshooting

**Job failed?**
1. Check error log: `cat benchmark_<JOB_ID>.err`
2. Verify conda environment: `conda activate myenv`
3. Check GPU availability: `squeue -p rtx3080`

**Results not found?**
1. Ensure job completed: `squeue -j <JOB_ID>`
2. Check log for errors
3. Verify data file exists: `ls -lh nordbyen_features_engineered.csv`

**Need faster execution?**
- Reduce epochs in `benchmarker.py` line 271
- Or modify `benchmark_job.slurm` to request more resources
