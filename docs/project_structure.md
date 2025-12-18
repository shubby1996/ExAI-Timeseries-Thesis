# Project Structure

This document outlines the organization of the codebase after restructuring to support multiple models (TFT and NHiTS).

## Root Directory (`data/`)

The root directory contains shared resources, data files, and generic utility scripts.

### Key Files
- **`model_preprocessing.py`**: The core preprocessing module used by ALL models. It handles:
    - Feature configuration (target, past/future covariates).
    - Data loading and validation.
    - Time-based splitting (Train/Val/Test).
    - Scaling (fitting on train, transforming all).
    - *Note: Contains aliases for backward compatibility with old `tft_preprocessing` pickles.*
- **`align_data.py`**: Script to align heat consumption data with weather data.
- **`feature_engineering.py`**: Script to generate features (lags, rolling means, calendar features) from aligned data.
- **`nordbyen_features_engineered.csv`**: The main dataset used for training models.

### Directories
- **`TFT/`**: Contains all scripts specific to the Temporal Fusion Transformer model.
- **`NHiTS/`**: Contains all scripts specific to the N-HiTS model.
- **`models/`**: Stores trained model artifacts (`.pt` files) and preprocessing states (`.pkl` files).
- **`results/`**: Stores evaluation metrics (`.csv`) and visualization plots (`.png`).
- **`docs/`**: Project documentation.

---

## Model Directories

### `TFT/`
- **`train_tft_nordbyen.py`**: Trains the TFT model.
- **`evaluate_tft_nordbyen.py`**: Evaluates the trained model using walk-forward validation.
- **`predict_tft_nordbyen.py`**: Generates future predictions.
- **`visualize_predictions.py`**: Generates plots for TFT performance.

### `NHiTS/`
- **`train_nhits_nordbyen.py`**: Trains the NHiTS model.
    - *Special Handling*: Merges future covariates into past covariates as NHiTS does not support future covariates.
- **`evaluate_nhits_nordbyen.py`**: Evaluates the NHiTS model.
- **`predict_nhits_nordbyen.py`**: Generates future predictions for NHiTS.
- **`visualize_nhits_predictions.py`**: Generates plots for NHiTS performance.
