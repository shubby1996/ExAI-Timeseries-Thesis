# TFT Pipeline Documentation

This document describes how to run the Temporal Fusion Transformer (TFT) pipeline.

## 1. Data Preparation
Ensure `nordbyen_features_engineered.csv` exists in the root `data/` directory. If not, run:
```bash
python align_data.py
python feature_engineering.py
```

## 2. Training
Train the TFT model using the default configuration.
```bash
python TFT/train_tft_nordbyen.py
```
- **Input**: `nordbyen_features_engineered.csv`
- **Output**: 
    - Model: `models/tft_nordbyen.pt`
    - State: `models/tft_nordbyen_preprocessing_state.pkl`

## 3. Evaluation
Evaluate the model on the test set using walk-forward validation.
```bash
python TFT/evaluate_tft_nordbyen.py
```
- **Output**:
    - Metrics: `results/evaluation_metrics.csv`
    - Predictions: `results/evaluation_predictions.csv`

## 4. Visualization
Generate performance plots.
```bash
python TFT/visualize_predictions.py
```
- **Output**: PNG plots in `results/` directory.
    - `plot_actual_vs_predicted.png`
    - `plot_error_distribution.png`
    - `plot_scatter.png`
    - `plot_daily_pattern.png`
    - `plot_error_over_time.png`

## 5. Future Prediction
Generate forecasts for the future (beyond the dataset).
```bash
python TFT/predict_tft_nordbyen.py
```
- **Output**: `predictions_future_24h.csv`
