# HPO Strategy Summary - Two-Stage Multi-Objective Approach

## 🎯 Core Philosophy

**Problem**: Current models have good MAE but poor PICP (40-53% vs 80% target)
**Solution**: Two-stage optimization separating architecture from calibration

## 📊 What We're Optimizing

### 4 Quantile Models (MSE variants excluded):
1. Water + TIMESNET_Q (Best: MAE 0.003, PICP 53%)
2. Heat + NHITS_Q (Best: MAE 0.194, PICP 40%)
3. Water + NHITS_Q (Second: MAE 0.005, PICP 41%)
4. Heat + TIMESNET_Q (Second: MAE 0.217, PICP 20%)

## 🔬 Two-Stage Approach

### Stage 1: Architecture Optimization (~50 trials each)
**Goal**: Find best model structure for MAE
- **Optimize**: MAE only
- **Vary**: Architecture + learning params
- **Fixed**: Quantiles at [0.1, 0.5, 0.9]
- **Output**: best_params_<model>.json

### Stage 2: Calibration Optimization (~20 trials each)
**Goal**: Achieve PICP ≈ 80%
- **Optimize**: |PICP - 80|
- **Vary**: Quantile levels [q_low, 0.5, q_high]
- **Fixed**: Architecture from Stage 1
- **Output**: calibrated_quantiles_<model>.json

## ⏱️ Timeline

- **Stage 1**: 4 models × 6-10 hours = 24-40 hours (can parallelize)
- **Stage 2**: 4 models × 2-3 hours = 8-12 hours (can parallelize)
- **Total**: ~32-52 hours sequential, ~12-20 hours parallel

## ✅ Success Criteria

**Deterministic**:
- MAE: Improve 5-15% OR maintain current levels
- RMSE/MAPE: Proportional improvement

**Probabilistic** (Critical):
- PICP: 75-85% (currently 40-53%)
- MIW: Reasonable (not excessively wide)
- CRPS: Improve or maintain

## 📁 Implementation Plan

### Files to Create:
1. `stage1_architecture.py` - Architecture optimizer
2. `stage2_calibration.py` - Calibration optimizer
3. `run_stage1.slurm` - SLURM for Stage 1
4. `run_stage2.slurm` - SLURM for Stage 2
5. `tracking/experiments.csv` - Progress tracking

### Execution Order:
```
Priority 1: Water TIMESNET_Q (best performer)
Priority 2: Heat NHITS_Q (best heat model)
Priority 3: Water NHITS_Q (fast, good baseline)
Priority 4: Heat TIMESNET_Q (compare architectures)
```

## 🎬 Next Steps

1. Implement Stage 1 optimizer
2. Implement Stage 2 optimizer
3. Create SLURM scripts
4. Setup tracking system
5. Execute experiments in priority order

See `HPO_STRATEGY.md` for complete details.
