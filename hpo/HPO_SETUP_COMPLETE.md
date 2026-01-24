# HPO Setup Complete ✅

**Date**: January 16, 2025  
**Status**: Ready for Testing

---

## 🎯 What Was Implemented

Clean, single-stage multi-objective HPO system for:
- **Models**: NHITS_Q, TFT_Q, TIMESNET_Q
- **Datasets**: heat (nordbyen), water_centrum, water_tommerby  
- **Optimization**: MAE (point forecast) + PICP (probabilistic coverage, target 80%)
- **Framework**: Optuna with Pareto front generation

---

## 📁 Created Files

| File | Size | Purpose |
|------|------|---------|
| `hpo_config.py` | 2.3K | Search spaces for all 3 models, dataset paths, training config |
| `run_hpo.py` | 20K | Main HPO runner (multi-objective, walk-forward validation) |
| `submit_job.sh` | 1.5K | SLURM job submission for individual experiments |
| `analyze_results.py` | 2.6K | Load and analyze all HPO results |
| `test_quick.sh` | 316B | Quick test with 2 trials (NHITS_Q on heat) |
| `logs/` | - | Directory for SLURM stdout/stderr |

---

## 🚀 Quick Start

### 1. Quick Test (2 trials, ~10 minutes)
```bash
./hpo/test_quick.sh
```

### 2. Full Local Test (3 trials, ~30 minutes)
```bash
python hpo/run_hpo.py --model NHITS_Q --dataset heat --trials 3 --job-id test_nhits
```

### 3. Submit SLURM Job (50 trials, production)
```bash
./hpo/submit_job.sh NHITS_Q heat 50
```

Monitor with:
```bash
squeue -u $USER
tail -f hpo/logs/hpo_NHITS_Q_heat_<JOBID>.log
```

---

## 📊 Results Location

Results are saved in structured directories:
```
hpo/results/
├── NHITS_Q_heat/
│   ├── best_params_NHITS_Q_heat_<JOBID>.json
│   └── pareto_front_NHITS_Q_heat_<JOBID>.html
├── TFT_Q_heat/
│   └── best_params_TFT_Q_heat_<JOBID>.json
└── TIMESNET_Q_heat/
    └── best_params_TIMESNET_Q_heat_<JOBID>.json
```

Each result contains:
- **Pareto Front**: All non-dominated solutions (MAE vs PICP tradeoff)
- **Best MAE**: Hyperparameters with lowest MAE
- **Best PICP**: Hyperparameters closest to 80% coverage
- **Balanced Solution**: Point closest to ideal (0 MAE, 0 PICP penalty)

---

## 🔍 Search Spaces

### NHITS_Q (7 hyperparameters)
- `num_stacks`: 2-5 (stacked architectures)
- `num_blocks`: 1-3 (blocks per stack)
- `layer_widths`: [256, 512, 1024]
- `pooling_kernel_sizes`: 2x/4x/8x downsampling combinations
- `learning_rate`: 1e-5 to 1e-2 (log scale)
- `dropout`: 0.0 to 0.3
- `batch_size`: [32, 64, 128]

### TFT_Q (5 hyperparameters)
- `hidden_size`: [32, 64, 128, 256]
- `lstm_layers`: 1-3
- `attention_heads`: [2, 4, 8]
- `dropout`: 0.0 to 0.3
- `learning_rate`: 1e-5 to 1e-3 (log scale)

### TIMESNET_Q (5 hyperparameters)
- `hidden_size`: [64, 128, 256]
- `conv_hidden_size`: [32, 64, 128]
- `num_kernels`: 3-6 (temporal convolutions)
- `top_k`: 2-5 (frequency components)
- `learning_rate`: 1e-5 to 1e-3 (log scale)

---

## 📋 Execution Plan

### Priority 1: Heat Dataset (most critical)
```bash
./hpo/submit_job.sh NHITS_Q heat 50      # ~16-20 GPU hours
./hpo/submit_job.sh TFT_Q heat 50        # ~25-30 GPU hours  
./hpo/submit_job.sh TIMESNET_Q heat 50   # ~16-20 GPU hours
```

### Priority 2: Water Centrum
```bash
./hpo/submit_job.sh NHITS_Q water_centrum 50
./hpo/submit_job.sh TFT_Q water_centrum 50
./hpo/submit_job.sh TIMESNET_Q water_centrum 50
```

### Priority 3: Water Tommerby
```bash
./hpo/submit_job.sh NHITS_Q water_tommerby 50
./hpo/submit_job.sh TFT_Q water_tommerby 50
./hpo/submit_job.sh TIMESNET_Q water_tommerby 50
```

**Total**: 9 experiments × 50 trials = 450 trials (~80-100 GPU hours)

---

## 🔧 Analyze Results

After jobs complete:
```bash
python hpo/analyze_results.py
```

Output includes:
- Summary table of best MAE and PICP per model-dataset
- Pareto front statistics
- Recommendations for balanced hyperparameters

---

## 🔗 Integration with Benchmarker

Update `benchmarker.py` to load HPO results:

```python
def _load_hpo_params(self, model_name):
    """Load best hyperparameters from HPO results"""
    # Detect dataset from self.name
    dataset = self._detect_dataset()
    
    # Find latest result file
    result_dir = f"hpo/results/{model_name}_{dataset}"
    result_files = glob(f"{result_dir}/best_params_*.json")
    
    if result_files:
        latest = max(result_files, key=os.path.getctime)
        with open(latest) as f:
            hpo_results = json.load(f)
            return hpo_results["best_balanced"]["hyperparameters"]
    
    return None  # Fallback to default params
```

---

## ✅ Validation Checklist

- [x] Old files deleted (hpo_tuner.py, hpo_job.slurm, etc.)
- [x] New clean implementation created
- [x] Search spaces defined for all 3 models
- [x] Multi-objective optimization (MAE + PICP)
- [x] Walk-forward validation (10 steps on validation set)
- [x] Pareto front generation and visualization
- [x] Result saving with job_id tracking
- [x] SLURM submission script
- [x] Analysis script
- [x] Help command works: `python hpo/run_hpo.py --help`
- [ ] Local test with 2-3 trials
- [ ] SLURM job submission and monitoring
- [ ] Full 50-trial runs for 9 experiments
- [ ] Results analysis
- [ ] Benchmarker integration

---

## 🎓 Key Decisions

1. **Single-Stage Optimization**: Combined MAE and PICP in one multi-objective study (user rejected two-stage approach)

2. **Multi-Objective Framework**: Using Optuna with Pareto optimization to find tradeoffs between forecast accuracy (MAE) and probabilistic coverage (PICP)

3. **File Organization**: Clean per-model-dataset directories with job IDs for tracking multiple runs

4. **Individual Job Submission**: User prefers explicit control over each experiment rather than batch submission

5. **Walk-Forward Validation**: 10-step walk-forward on validation set to get robust MAE and PICP estimates

---

## 📚 References

- **Master Plan**: `hpo/HPO_REVIVAL_PLAN.md`
- **Model Documentation**: `docs/MODEL_CONFIGURATION_AND_PREDICTION_FLOW.md`
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Darts HPO Examples**: https://unit8co.github.io/darts/examples/15-hyperparameter-optimization.html

---

## 🐛 Troubleshooting

### Job fails immediately
- Check SLURM logs: `tail -f hpo/logs/hpo_<MODEL>_<DATASET>_<JOBID>.log`
- Verify dataset path exists in `hpo_config.py`
- Check GPU availability: `sinfo -p gpu`

### Out of memory
- Reduce `batch_size` in trial suggestions
- Reduce `hidden_size` search space
- Request more GPU memory in submit_job.sh

### No Pareto front generated
- Need at least 2 trials to create a front
- Check that both MAE and PICP are being computed
- Verify results directory was created

### Import errors
- Activate conda environment: `conda activate myenv`
- Check package versions: `pip list | grep -E "optuna|darts|neuralforecast"`

---

**Next Step**: Run `./hpo/test_quick.sh` to validate the setup! 🚀
