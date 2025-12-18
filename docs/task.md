# Project Tasks

## ✅ Completed
- [x] **Project Restructuring**
    - [x] Create `TFT/` directory for TFT-specific code.
    - [x] Create `NHiTS/` directory for NHiTS-specific code.
    - [x] Refactor preprocessing to be generic (`model_preprocessing.py`).
    - [x] Update all scripts to use relative paths and generic preprocessing.
- [x] **TFT Pipeline Verification**
    - [x] Verify data alignment (`align_data.py`).
    - [x] Verify feature engineering (`feature_engineering.py`).
    - [x] Train TFT model (`TFT/train_tft_nordbyen.py`).
    - [x] Evaluate TFT model (`TFT/evaluate_tft_nordbyen.py`).
    - [x] Visualize TFT predictions (`TFT/visualize_predictions.py`).
- [x] **NHiTS Implementation**
    - [x] Implement NHiTS training script (`NHiTS/train_nhits_nordbyen.py`).
        - *Note: Handled NHiTS limitation (no future covariates) by merging into past covariates.*
    - [x] Implement NHiTS evaluation script (`NHiTS/evaluate_nhits_nordbyen.py`).
    - [x] Implement NHiTS prediction script (`NHiTS/predict_nhits_nordbyen.py`).
    - [x] Implement NHiTS visualization script (`NHiTS/visualize_nhits_predictions.py`).
    - [x] Verify full NHiTS pipeline execution.
- [x] **Documentation**
    - [x] Document project structure.
    - [x] Document TFT pipeline usage.
    - [x] Document NHiTS pipeline usage.
    - [x] Create changelog.

- [x] **TCN Implementation**
    - [x] Create `TCN/` directory.
    - [x] Implement TCN training script (`TCN/train_tcn_nordbyen.py`).
    - [x] Implement TCN evaluation script (`TCN/evaluate_tcn_nordbyen.py`).
    - [x] Implement TCN prediction script (`TCN/predict_tcn_nordbyen.py`).
    - [x] Implement TCN visualization script (`TCN/visualize_tcn_predictions.py`).
    - [x] Verify full TCN pipeline execution.
        - *Note: TCN evaluation has issues with quantile predictions (NaN errors).*
- [x] **Comparison & Analysis**
    - [x] Compare TFT vs NHiTS performance metrics (`docs/model_comparison.md`).
        - *Result: TFT outperforms NHiTS across all metrics (7.5% better MAE, 64% better R²).*
    - [x] Analyze model characteristics (size, architecture, uncertainty quantification).

## 📅 Upcoming
- [ ] **TCN Debugging & Refinement**
    - [ ] Fix TCN quantile prediction issues (NaN errors in evaluation).
    - [ ] Complete TCN evaluation and add to comparison report.
    - [ ] Document TCN pipeline (`docs/tcn_pipeline.md`).
- [ ] **Further Analysis**
    - [ ] Analyze computational efficiency (training time, inference speed).
    - [ ] Investigate `StatsForecast` warning (optional).
    - [ ] Hyperparameter tuning for models (if needed).
- [ ] **Thesis Writing**
    - [ ] Leverage existing documentation for methodology chapter.
    - [ ] Create publication-quality figures from visualizations.
    - [ ] Write results and discussion sections.
