# Hyperparameter Optimization Strategy - Multi-Objective Approach

## Overview
This document outlines the **multi-objective HPO strategy** focusing on improving both:
1. **Deterministic Performance**: MAE, RMSE, MAPE
2. **Probabilistic Performance**: PICP calibration (target: 80%), MIW, CRPS

## Philosophy
**Focus on what matters**: Optimize only Quantile models (NHITS_Q, TIMESNET_Q) since:
- They consistently outperform MSE variants
- They provide uncertainty quantification (essential for production)
- Current benchmark issue: PICP ~40-53% vs target 80% (intervals too narrow)

## Scope
**Total HPO Experiments**: 4 focused runs (Quantile models only)

**Models to Optimize**:
1. Heat + NHITS_Q (Current: MAE 0.194, PICP 40%)
2. Heat + TIMESNET_Q (Current: MAE 0.217, PICP 20%)
3. Water + NHITS_Q (Current: MAE 0.005, PICP 41%)
4. Water + TIMESNET_Q (Current: MAE 0.003, PICP 53%)

**Why NOT optimizing MSE variants**:
- No uncertainty quantification capability
- Consistently worse MAE than Quantile
- Not production-ready (need intervals for decision-making)

---

## Multi-Objective Optimization Strategy

### Problem: Single-objective (MAE only) is insufficient
Current models show **narrow intervals** (PICP 40-53% vs 80% target). Optimizing only for MAE could:
- Further narrow intervals (worse calibration)
- Sacrifice probabilistic quality for deterministic gains
- Miss the real production requirement (reliable uncertainty)

### Solution: Balanced Multi-Objective Function

**Objective Function**:
```python
score = α × normalized_MAE + β × calibration_penalty

where:
  calibration_penalty = abs(PICP - 80) / 80
  α = 0.6  # Weight for deterministic performance
  β = 0.4  # Weight for probabilistic calibration
```

**Rationale**:
- Balances accuracy and calibration
- Penalizes both under-coverage (<80%) and over-coverage (>80%)
- Prevents optimizing one at expense of the other

### Alternative Approaches

**Option 1: Two-stage optimization** (Recommended)
1. **Stage 1**: Optimize for MAE (find best architecture/learning params)
2. **Stage 2**: Fix architecture, tune only quantile widths for PICP=80%

**Option 2: Pareto optimization**
- Use Optuna multi-objective studies
- Generate Pareto front of MAE vs PICP
- Select from frontier based on requirements

**Option 3: Constraint-based**
- Primary: Minimize MAE
- Constraint: PICP must be 75-85%
- Reject trials outside constraint

### Recommended: Two-Stage Approach

**Stage 1: Architecture + Learning**
```python
Optimize: MAE only
Hyperparameters:
  - Architecture (num_stacks, num_blocks, layers, widths, hidden_size)
  - Learning (lr, dropout, weight_decay)
  - Fixed: quantiles = [0.1, 0.5, 0.9]
Trials: 50
Goal: Find best model structure
```

**Stage 2: Calibration**
```python
Optimize: PICP calibration (minimize |PICP - 80|)
Hyperparameters:
  - Quantile levels only: [low, 0.5, high]
  - Fixed: Best architecture from Stage 1
Trials: 20
Goal: Achieve PICP ≈ 80%
```

**Why two-stage?**
- Stage 1: Focus on what architecture learns best
- Stage 2: Post-hoc calibration without retraining
- Separates concerns: model quality vs interval width
- Computationally efficient

---

## Execution Strategy

### Phase 1: Benchmark Analysis ✅ COMPLETED
**Status**: Dec 26, 2025 benchmarks completed

**Key Findings**:
- Quantile models superior to MSE across all metrics
- PICP severely under-calibrated (40-53% vs 80%)
- Heat: NHITS_Q best (MAE 0.194)
- Water: TIMESNET_Q best (MAE 0.003)

### Phase 2: Prioritization
Based on benchmark results, prioritize models by **impact potential**:

**Tier 1 - CRITICAL** (Both need improvement):
1. **Water + TIMESNET_Q**: Best MAE but moderate PICP (53%)
2. **Heat + NHITS_Q**: Best heat model but worst PICP (40%)

**Tier 2 - IMPORTANT** (Secondary performers):
3. **Water + NHITS_Q**: Good MAE, poor PICP (41%)
4. **Heat + TIMESNET_Q**: Moderate MAE, worst PICP (20%)

### Phase 3: Two-Stage Execution Plan

**Stage 1: Architecture Optimization (4 experiments)**
```bash
Priority Order:
1. Water + TIMESNET_Q  [Best overall, needs calibration]
2. Heat + NHITS_Q      [Best heat, needs calibration]
3. Water + NHITS_Q     [Fast training, good baseline]
4. Heat + TIMESNET_Q   [Compare architectures]

Each experiment:
- Optimize: MAE on validation set
- Trials: 50
- Time: ~4-8 hours per experiment
- Output: best_params_<model>.json
```

**Stage 2: Calibration Optimization (4 experiments)**
```bash
Same order as Stage 1, but:
- Fix architecture from Stage 1
- Tune quantile levels: [q_low, 0.5, q_high]
- Optimize: |PICP - 80|
- Trials: 20
- Time: ~2-3 hours per experiment
- Output: calibrated_quantiles_<model>.json
```

### Phase 4: Validation
After both stages complete:
1. Run full benchmark with Stage 1 params + Stage 2 quantiles
2. Compare deterministic metrics (MAE, RMSE, MAPE)
3. Verify calibration (PICP ≈ 80%, reasonable MIW)
4. Select production models

---

## Resource Considerations

### Compute Time Estimates
**Stage 1** (Architecture):
- NHITS: ~4-6 hours (50 trials)
- TIMESNET: ~6-10 hours (50 trials)
- Water dataset: Faster than heat (less data)

**Stage 2** (Calibration):
- Fast: ~2-3 hours (20 trials, no retraining)
- Just evaluates different quantile thresholds

**Total per model**: ~6-13 hours (both stages)
**Total for all 4 models**: ~30-40 hours sequential, ~10-15 hours parallel

### GPU Requirements
- Each experiment: 1 GPU (tinygpu partition)
- Can parallelize: 2-4 jobs if GPUs available
- Stage 2 can use results/ directory from Stage 1

---

## Implementation Requirements

### 1. Stage 1: Architecture Optimizer
**File**: `hpo/stage1_architecture.py`

Features needed:
- Load dataset (heat/water)
- Train NHITS_Q or TIMESNET_Q with trial params
- Evaluate MAE on validation set
- Return MAE as objective
- Save best params

Hyperparameter space:
```python
NHITS:
  num_stacks: 1-5
  num_blocks: 1-3  
  num_layers: 1-4
  layer_widths: [128, 256, 512, 1024]
  lr: 1e-5 to 1e-2 (log)
  dropout: 0.0 to 0.5
  weight_decay: 1e-8 to 1e-2 (log)
  quantiles: [0.1, 0.5, 0.9]  # FIXED

TIMESNET:
  hidden_size: [32, 64, 128, 256]
  conv_hidden_size: [32, 64, 128, 256]
  top_k: 1-5
  lr: 1e-5 to 1e-2 (log)
  dropout: 0.0 to 0.5
  quantiles: [0.1, 0.5, 0.9]  # FIXED
```

### 2. Stage 2: Calibration Optimizer
**File**: `hpo/stage2_calibration.py`

Features needed:
- Load best architecture from Stage 1
- Train with fixed architecture
- Vary ONLY quantile levels: [q_low, 0.5, q_high]
- Evaluate PICP on validation set
- Minimize: abs(PICP - 80)
- Save calibrated quantiles

Hyperparameter space:
```python
Quantile calibration:
  q_low: 0.01 to 0.25  # Lower quantile
  q_high: 0.75 to 0.99  # Upper quantile
  (middle always 0.5 for median)
```

### 3. Directory Structure
```
hpo/
  stage1_architecture.py      # Architecture optimizer
  stage2_calibration.py        # Calibration optimizer
  run_stage1.slurm            # SLURM template for stage 1
  run_stage2.slurm            # SLURM template for stage 2
  submit_hpo.sh               # Helper to submit experiments
  
  results/
    stage1/
      heat_nhits_q/
        study.db              # Optuna database
        best_params.json      # Best architecture
      water_timesnet_q/
        ...
    stage2/
      heat_nhits_q/
        study.db              # Calibration study
        calibrated_quantiles.json
      water_timesnet_q/
        ...
  
  tracking/
    experiments.csv           # Track all experiments
    stage1_summary.md         # Stage 1 results
    stage2_summary.md         # Stage 2 results
```

### 4. Experiment Tracking
**File**: `hpo/tracking/experiments.csv`
```csv
stage,model,dataset,status,job_id,mae,picp,start,end,notes
1,NHITS_Q,water,pending,,,,,,"Stage 1: Architecture"
1,NHITS_Q,heat,pending,,,,,,"Stage 1: Architecture"
1,TIMESNET_Q,water,pending,,,,,,"Stage 1: Architecture"
1,TIMESNET_Q,heat,pending,,,,,,"Stage 1: Architecture"
2,NHITS_Q,water,pending,,,,,,"Stage 2: Calibration"
...
```

---

## Decision Checkpoints

### After Stage 1 (Architecture Optimization)
**Review**:
- Did MAE improve vs baseline?
- What hyperparameters mattered most?
- Are patterns consistent across datasets?

**Decide**:
- Proceed to Stage 2 for all models
- Skip Stage 2 for models that don't improve
- Adjust Stage 2 quantile ranges based on current PICP

### After Stage 2 (Calibration)
**Review**:
- Is PICP now 75-85%?
- Did MIW increase reasonably?
- Did MAE stay similar (not degrade)?

**Decide**:
- Run final benchmark with optimized params
- Accept models for production
- Iterate if calibration failed

### Final Validation
**Review full benchmark**:
- Compare Stage1+Stage2 vs baseline
- Deterministic: MAE, RMSE, MAPE
- Probabilistic: PICP (target: 80%), MIW, CRPS

**Success criteria**:
- MAE: 5-15% improvement OR maintained
- PICP: 75-85% (currently 40-53%)
- MIW: Reasonable (not excessively wide)

---

## Next Steps

### Immediate Actions
1. ✅ Define multi-objective strategy (this document)
2. ⏳ Implement Stage 1 optimizer
3. ⏳ Implement Stage 2 optimizer
4. ⏳ Create SLURM scripts for both stages
5. ⏳ Setup tracking system

### Execution Sequence
1. Run Stage 1 for all 4 models (can parallelize)
2. Analyze Stage 1 results
3. Run Stage 2 for all 4 models (using Stage 1 params)
4. Analyze Stage 2 results
5. Run final benchmarks with optimized params
6. Compare and document improvements

---

**Last Updated**: 2025-12-27
**Status**: Strategy defined - Ready for implementation
**Next Action**: Implement Stage 1 architecture optimizer

---

## Implementation Notes (December 27, 2025)

### Quantile Symmetry Constraint
Darts QuantileRegression requires symmetric quantiles around the median (0.5). The implementation uses:
- **Approach**: q_offset parameter instead of separate q_low/q_high
- **Range**: q_offset ∈ [0.01, 0.49]
- **Computation**: q_low = 0.5 - q_offset, q_high = 0.5 + q_offset
- **Benefit**: Ensures systematic exploration while respecting library constraints

### Module Loading
Use `module load python/pytorch2.6py3.12` before running HPO scripts locally or in SLURM.

### Test Results
✅ Stage 1 tested successfully (Water NHITS_Q: MAE=0.1427)
✅ Stage 2 tested successfully (loads Stage 1 results, enforces constraints)
