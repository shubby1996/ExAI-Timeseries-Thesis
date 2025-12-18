# Nordbyen Heat Forecasting (Thesis Project)

This repository contains the codebase for a Master's Thesis focused on **Explainable AI (ExAI) for Timeseries Forecasting** in the context of district heating. The project compares state-of-the-art models like the **Temporal Fusion Transformer (TFT)** and **N-HiTS** for predicting hourly heat consumption.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Darts](https://unit8co.github.io/darts/) (Time Series library)
- [NeuralForecast](https://nixtla.github.io/neuralforecast/) (for TimesNet)
- PyTorch / PyTorch Lightning

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Access
The project uses historical heat consumption and weather data for Nordbyen, Brønderslev.
- **Data Link**: [Download Dataset](https://faubox.rrze.uni-erlangen.de/getlink/fi38h8qVsEbd37f3TpTBUZ/)
- Place the raw data in appropriate subdirectories (e.g., `dma_a_nordbyen_heat/`, `weather/`) or follow the alignment scripts.

---

## 📂 Repository Structure

- **`TFT/`**: Scripts for training, evaluation, and prediction using the Temporal Fusion Transformer.
- **`NHiTS/`**: Scripts specific to the N-HiTS (Neural Hierarchical Interpolation for Time Series) model.
- **`timesnet/`**: Scripts for the TimesNet model, leveraging multi-periodic temporal patterns.
- **`docs/`**: Detailed documentation on the project journey, technical architecture, and experiment results.
- **`models/`**: (Generated) Directory for saving trained model artifacts (`.pt`) and preprocessing state (`.pkl`).
- **`results/`**: (Generated) Output directory for performance metrics and visualization plots.
- **`model_preprocessing.py`**: Shared utility for data cleaning, scaling, and feature configuration.
- **`feature_engineering.py`**: Logic for generating thermodynamic and temporal features.

---

## 🛠️ The Pipeline

### 1. Data Preparation
Before modeling, the data must be aligned and features engineered:
- **Alignment**: Run `align_data.py` to synchronize heat consumption with weather data.
- **Feature Engineering**: Run `feature_engineering.py` to generate lags, rolling averages, and calendar features.

### 2. Temporal Fusion Transformer (TFT)
TFT is used for its ability to handle mixed inputs and provide interpretability via attention.
- **Train**: `python TFT/train_tft_nordbyen.py`
- **Evaluate**: `python TFT/evaluate_tft_nordbyen.py` (Performs walk-forward validation and calculates uncertainty).
- **Predict**: `python TFT/predict_tft_nordbyen.py` (Generates future forecasts).

### 3. N-HiTS
N-HiTS provides a faster, hierarchical approach to long-term forecasting.
- **Train**: `python NHiTS/train_nhits_nordbyen.py`
- **Evaluate**: `python NHiTS/evaluate_nhits_nordbyen.py`

### 4. TimesNet
TimesNet transforms 1D time series into 2D variations to capture multi-periodic patterns.
- **Train (MAE/MSE/Prob)**:
  - `python timesnet/train_timesnet_mae.py`
  - `python timesnet/train_timesnet_mse.py`
  - `python timesnet/train_timesnet_prob.py`
- **Evaluate**: `python timesnet/evaluate_timesnet_nordbyen.py`
- **Compare**: `python run_compare_nhits_timesnet.py` (Compares performance against NHiTS).

---

## 📖 Key Documentation

For a deep dive into the project, refer to these documents in the `docs/` folder:
- [**Project Journey**]({WORKDIR}/docs/project_journey.md): A chronological explanation of the technical decisions and data science process.
- [**Project Structure**]({WORKDIR}/docs/project_structure.md): Detailed map of files and folders.
- [**Experiment Results Summary**]({WORKDIR}/docs/experiment_results_summary.md): High-level comparison of model performance.
- [**Technical Deep Dive**]({WORKDIR}/docs/pipeline_technical_deep_dive.md): In-depth look at model specifications and pipeline logic.

---

## 📊 Results & Visualization
Evaluation metrics (MAE, RMSE, Quantile Loss, PICP) are saved to the `results/` folder. Corresponding visualization plots can be found in `results/` and `plots/`.
