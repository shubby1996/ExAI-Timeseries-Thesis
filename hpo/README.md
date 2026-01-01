# HPO Two-Stage System

Complete implementation of multi-objective hyperparameter optimization.

## Quick Start - NEW NOTEBOOK INTERFACE

**Recommended**: Use the interactive Jupyter notebook for easy job submission and monitoring:

```bash
jupyter lab hpo/hpo_runner_v2.ipynb
```

This notebook provides:
- **Section 4**: Submit all Stage 1 experiments with confirmation prompt
- **Section 5**: Monitor Stage 1 progress (run periodically)
- **Section 6**: Submit all Stage 2 experiments (after Stage 1 complete)
- **Section 7**: Monitor Stage 2 progress
- **Section 8**: View and compare final results in DataFrames

**Resources**: A100 GPU, 10-hour time limit, 32GB RAM, tinygpu partition

---

## Command-Line Interface (Alternative)

### Submit All Experiments (Priority Order)
```bash
python hpo/submit_experiment.py --stage all --priority
```

### Submit Individual Experiments
```bash
# Stage 1: Architecture optimization
python hpo/submit_experiment.py --stage 1 --model NHITS_Q --dataset water --trials 50

# Stage 2: Calibration optimization (after Stage 1 completes)
python hpo/submit_experiment.py --stage 2 --model NHITS_Q --dataset water --trials 20
```

### Check Status
```bash
python hpo/check_status.py
python hpo/check_status.py --model NHITS_Q --dataset water
```

### Local Testing

First, load the required module:
```bash
module load python/pytorch2.6py3.12
```

Then test:
```bash
# Test Stage 1 (1 trial, ~2 minutes)
python hpo/stage1_architecture.py --model NHITS_Q --dataset water --test

# Test Stage 2 (1 trial, requires Stage 1 complete)
python hpo/stage2_calibration.py --model NHITS_Q --dataset water --test
```

**Note**: Training runs on CPU. For faster execution, use SLURM with GPU.

## Files

### Core Scripts
- `hpo_tuner_v2.py` - Enhanced HPO tuner with dataset/loss support
- `hpo_job_v2.slurm` - SLURM job template
- `submit_hpo.sh` - Job submission helper
- `check_status.py` - Status tracking tool

### Configuration
- `HPO_STRATEGY.md` - Complete execution strategy
- `hpo_experiments_log.csv` - Experiment tracking log

### Results
- `results/<experiment>/` - Study databases and best params
- `results/<experiment>/best_params.json` - Best hyperparameters
## Directory Structure

```
hpo/
├── stage1_architecture.py      # Stage 1: Optimize architecture for MAE
├── stage2_calibration.py       # Stage 2: Calibrate quantiles for PICP
├── submit_experiment.py        # Submit jobs to SLURM
├── check_status.py             # Monitor experiment status
├── run_stage1.slurm           # SLURM script for Stage 1
├── run_stage2.slurm           # SLURM script for Stage 2
├── HPO_STRATEGY.md            # Complete strategy document
├── STRATEGY_SUMMARY.md        # Quick reference
├── README.md                  # This file
├── results/
│   ├── stage1/                # Stage 1 results
│   │   ├── water_nhits_q/
│   │   │   ├── best_params.json
│   │   │   └── study.db
│   │   └── ...
│   └── stage2/                # Stage 2 results
│       ├── water_nhits_q/
│       │   ├── calibrated_quantiles.json
│       │   └── study.db
│       └── ...
└── tracking/
    └── experiments.csv        # Experiment tracking
```

## Implementation Files

### 1. stage1_architecture.py
- **Purpose**: Optimize model architecture and learning parameters for MAE
- **Fixed**: Quantiles at [0.1, 0.5, 0.9]
- **Optimizes**: num_stacks, num_blocks, num_layers, layer_widths, hidden_size, lr, dropout, etc.
- **Output**: `results/stage1/<dataset>_<model>/best_params.json`

### 2. stage2_calibration.py
- **Purpose**: Calibrate quantile levels to achieve PICP ≈ 80%
- **Fixed**: Architecture from Stage 1
- **Optimizes**: q_low (0.01-0.25), q_high (0.75-0.99)
- **Objective**: Minimize |PICP - 80|
- **Output**: `results/stage2/<dataset>_<model>/calibrated_quantiles.json`

### 3. run_stage1.slurm
- SLURM job script for Stage 1
- GPU: A100, 12 hours, 32GB RAM
- Exports: MODEL, DATASET, TRIALS

### 4. run_stage2.slurm
- SLURM job script for Stage 2
- GPU: A100, 6 hours, 32GB RAM
- Exports: MODEL, DATASET, TRIALS

### 5. submit_experiment.py
- Submit experiments to SLURM
- Handles dependencies (Stage 2 requires Stage 1 complete)
- Tracks experiments in `tracking/experiments.csv`
- Priority order submission: Water TIMESNET_Q → Heat NHITS_Q → Water NHITS_Q → Heat TIMESNET_Q

### 6. check_status.py
- Monitor experiment progress
- Shows SLURM status, results, completion
- Next steps recommendations

## Experiment Priority

From [HPO_STRATEGY.md](HPO_STRATEGY.md):

1. **Water TIMESNET_Q** - Best overall performer (MAE: 0.003, PICP: 53%)
2. **Heat NHITS_Q** - Best heat model (MAE: 0.194, PICP: 40%)
3. **Water NHITS_Q** - Fast training, good baseline
4. **Heat TIMESNET_Q** - Architecture comparison

## Two-Stage Approach

### Stage 1: Architecture Optimization (50 trials)
- **Goal**: Find best model structure for MAE
- **Fixed**: Quantiles = [0.1, 0.5, 0.9]
- **Search Space**: Architecture and learning parameters
- **Duration**: 4-10 hours per experiment
- **Output**: Best hyperparameters, MAE achieved

### Stage 2: Calibration Optimization (20 trials)
- **Goal**: Achieve PICP ≈ 80% (±5%)
- **Fixed**: Architecture from Stage 1
- **Search Space**: Quantile levels (q_low, q_high)
- **Duration**: 2-3 hours per experiment
- **Output**: Calibrated quantiles, PICP achieved

## Success Criteria

### Stage 1
- ✓ MAE improves by 5-15% over baseline
- ✓ No catastrophic overfitting
- ✓ Training converges stably

### Stage 2
- ✓ PICP reaches 75-85%
- ✓ Interval width reasonable (not too wide)
- ✓ Calibration consistent across forecast horizon

### Overall
- ✓ Both deterministic (MAE) and probabilistic (PICP) performance improved
- ✓ Models deployable for production

## Typical Workflow

```bash
# 1. Submit all Stage 1 experiments
python hpo/submit_experiment.py --stage all --priority

# 2. Monitor progress
watch -n 60 'python hpo/check_status.py'

# 3. When Stage 1 completes for a model, submit Stage 2
python hpo/submit_experiment.py --stage 2 --model NHITS_Q --dataset water

# 4. After all experiments complete, run final benchmarks
# (Use optimized parameters from results/stage2/)

# 5. Compare with baseline using benchmark_runner.ipynb
```

## Results Format

### Stage 1 Output (best_params.json)
```json
{
  "stage": 1,
  "model": "NHITS_Q",
  "dataset": "water",
  "objective": "MAE",
  "best_mae": 0.0025,
  "best_params": {
    "num_stacks": 3,
    "num_blocks": 2,
    "num_layers": 3,
    "layer_widths": 512,
    "lr": 0.001,
    "dropout": 0.2,
    ...
  },
  "n_trials": 50,
  "timestamp": "2025-01-10 15:30:00"
}
```

### Stage 2 Output (calibrated_quantiles.json)
```json
{
  "stage": 2,
  "model": "NHITS_Q",
  "dataset": "water",
  "objective": "PICP_calibration",
  "target_picp": 80.0,
  "achieved_picp": 78.5,
  "calibration_error": 1.5,
  "calibrated_quantiles": [0.08, 0.5, 0.92],
  "architecture_params": {...},
  "stage1_mae": 0.0025,
  "n_trials": 20,
  "timestamp": "2025-01-10 20:15:00"
}
```

## Troubleshooting

### Stage 1 Not Finding Good Parameters
- Increase trials: `--trials 100`
- Check data quality and preprocessing
- Review learning curves in Optuna study

### Stage 2 Cannot Achieve PICP ≈ 80%
- Review Stage 1 architecture (may need better base model)
- Try wider search range: `q_low (0.01-0.3), q_high (0.7-0.99)`
- Check validation set representativeness

### SLURM Jobs Failing
- Check logs: `tail -f hpo/results/stage1/hpo_stage1_<jobid>.log`
- Verify environment: `module load Anaconda3 CUDA`
- Check GPU availability: `sinfo -p tinygpu`

### Restart failed experiment
```bash
# Optuna supports resuming from database

## References

- [HPO_STRATEGY.md](HPO_STRATEGY.md) - Complete strategy and implementation guide
- [STRATEGY_SUMMARY.md](STRATEGY_SUMMARY.md) - Quick reference
- [benchmark_runner.ipynb](../benchmark_runner.ipynb) - Baseline results comparison

## Notes

- **Multi-objective**: Both MAE and PICP must improve
- **Quantile models only**: NHITS_Q and TIMESNET_Q (4 experiments total)
- **Two-stage rationale**: Separates architecture search from calibration to avoid conflating objectives
- **Baseline PICP issue**: Currently 40-53%, target 80% - this is the critical problem to solve
