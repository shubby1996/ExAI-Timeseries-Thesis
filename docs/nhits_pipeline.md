# NHiTS Pipeline Documentation

This document describes how to run the N-HiTS pipeline.

## 1. Data Preparation
Uses the same shared data as TFT. Ensure `nordbyen_features_engineered.csv` exists.
```bash
python align_data.py
python feature_engineering.py
```

## 2. Training
Train the NHiTS model.
```bash
python NHiTS/train_nhits_nordbyen.py
```
- **Special Note**: NHiTS does not support "future covariates" (known future inputs like time, holidays) natively in the same way TFT does. The training script automatically **merges future covariates into past covariates** to allow the model to access this information.
- **Output**: 
    - Model: `models/nhits_nordbyen.pt`
    - State: `models/nhits_nordbyen_preprocessing_state.pkl`

## 3. Evaluation
Evaluate the model on the test set.
```bash
python NHiTS/evaluate_nhits_nordbyen.py
```
- **Output**:
    - Metrics: `results/nhits_evaluation_metrics.csv`
    - Predictions: `results/nhits_evaluation_predictions.csv`

## 4. Visualization
Generate performance plots for NHiTS.
```bash
python NHiTS/visualize_nhits_predictions.py
```
- **Output**: PNG plots in `results/` directory (prefixed with `nhits_`).
    - `nhits_plot_actual_vs_predicted.png`
    - `nhits_plot_error_distribution.png`
    - ...

## 5. Future Prediction
Generate forecasts for the future.
```bash
python NHiTS/predict_nhits_nordbyen.py
```
- **Output**: `nhits_predictions_future_24h.csv`
