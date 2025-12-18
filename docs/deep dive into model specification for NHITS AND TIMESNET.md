# Deep Dive into Model Specification for NHiTS and TimesNet

## NHiTS Model Specifications

### 1. NHiTS MAE Training Script
**File:** `NHiTS/train_nhits_mae.py`

- **Loss Function:** Mean Absolute Error (MAE) using `torch.nn.L1Loss`.
- **Model Save Name:** `nhits_deterministic_mae`
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and preprocesses it using `prepare_model_data`.
  - **Data Preprocessing:**
    - Splits data into training and validation sets.
    - Merges `future_covariates` into `past_covariates` (NHiTS requirement).
  - **Model Configuration:**
    - Input Chunk Length: 168 hours (7 days).
    - Output Chunk Length: 24 hours.
    - Stacks: 3.
    - Blocks: 1 per stack.
    - Layers: 2 per block.
    - Layer Widths: 512.
    - Dropout: 0.1.
    - Activation: ReLU.
    - Loss Function: `torch.nn.L1Loss` (MAE).
    - Early Stopping: Patience of 5 epochs, monitoring `val_loss`.
  - **Training:**
    - Trains the model using `train_target` and `train_past` with validation on `val_target` and `val_past`.
  - **Outputs:**
    - Saves the trained model to `models/nhits_deterministic_mae.pt`.
    - Saves preprocessing state to `models/nhits_deterministic_mae_preprocessing_state.pkl`.

### 2. NHiTS MSE Training Script
**File:** `NHiTS/train_nhits_mse.py`

- **Loss Function:** Mean Squared Error (MSE).
- **Model Save Name:** `nhits_nordbyen`
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and preprocesses it using `prepare_model_data`.
  - **Data Preprocessing:**
    - Splits data into training and validation sets.
    - Merges `future_covariates` into `past_covariates` (NHiTS requirement).
  - **Model Configuration:**
    - Input Chunk Length: 168 hours (7 days).
    - Output Chunk Length: 24 hours.
    - Stacks: 3.
    - Blocks: 1 per stack.
    - Layers: 2 per block.
    - Layer Widths: 512.
    - Dropout: 0.1.
    - Activation: ReLU.
    - Loss Function: MSE.
    - Early Stopping: Patience of 5 epochs, monitoring `val_loss`.
  - **Training:**
    - Trains the model using `train_target` and `train_past` with validation on `val_target` and `val_past`.
  - **Outputs:**
    - Saves the trained model to `models/nhits_nordbyen.pt`.
    - Saves preprocessing state to `models/nhits_nordbyen_preprocessing_state.pkl`.

### 3. NHiTS Probabilistic Training Script
**File:** `NHiTS/train_nhits_prob.py`

- **Loss Function:** Quantile Regression using `QuantileRegression` with quantiles [0.1, 0.5, 0.9].
- **Model Save Name:** `nhits_probabilistic_q`
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and preprocesses it using `prepare_model_data`.
  - **Data Preprocessing:**
    - Splits data into training and validation sets.
    - Merges `future_covariates` into `past_covariates` (NHiTS requirement).
  - **Model Configuration:**
    - Input Chunk Length: 168 hours (7 days).
    - Output Chunk Length: 24 hours.
    - Stacks: 3.
    - Blocks: 1 per stack.
    - Layers: 2 per block.
    - Layer Widths: 512.
    - Dropout: 0.1.
    - Activation: ReLU.
    - Likelihood: `QuantileRegression` with quantiles [0.1, 0.5, 0.9].
    - Early Stopping: Patience of 5 epochs, monitoring `val_loss`.
  - **Training:**
    - Trains the model using `train_target` and `train_past` with validation on `val_target` and `val_past`.
  - **Outputs:**
    - Saves the trained model to `models/nhits_probabilistic_q.pt`.
    - Saves preprocessing state to `models/nhits_probabilistic_q_preprocessing_state.pkl`.

---

## TimesNet Model Specifications

### 1. TimesNet MAE Training Script
**File:** `timesnet/train_timesnet_mae.py`

- **Loss Function:** Mean Absolute Error (MAE).
- **Model Save Name:** `timesnet_deterministic_mae`
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and converts it to NeuralForecast long format.
  - **Data Preprocessing:**
    - Filters relevant columns and removes NaNs.
    - Splits data into training and future datasets.
  - **Model Configuration:**
    - Horizon: 24 hours.
    - Input Size: 168 hours (7 days).
    - Loss: MAE.
    - Scaler: Standard scaling.
    - Learning Rate: 1e-3.
    - Max Steps: 20.
  - **Training:**
    - Uses `NeuralForecast` to train the model.
  - **Prediction:**
    - Generates forecasts for the next 24 hours.
    - Calculates MAE on the forecasted data.
  - **Outputs:**
    - Saves the trained model to `models/timesnet_deterministic_mae`.
    - Saves a plot of actual vs. predicted values to `results/stage2_mae_plot.png`.

### 2. TimesNet MSE Training Script
**File:** `timesnet/train_timesnet_mse.py`

- **Loss Function:** Mean Squared Error (MSE).
- **Model Save Name:** Not explicitly mentioned, but likely follows the MSE naming convention.
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and converts it to NeuralForecast long format.
  - **Data Preprocessing:**
    - Filters relevant columns and removes NaNs.
    - Splits data into training and future datasets.
  - **Model Configuration:**
    - Horizon: 24 hours.
    - Input Size: 168 hours (7 days).
    - Loss: MSE.
    - Scaler: Standard scaling.
    - Max Steps: 500.
    - Batch Size: 32.
  - **Training:**
    - Uses `NeuralForecast` to train the model.
  - **Prediction:**
    - Generates forecasts for the next 24 hours.
    - Calculates MSE on the forecasted data.
  - **Outputs:**
    - Saves a plot of actual vs. predicted values to `timesnet/stage1_mse_plot.png`.
    - Prints a warning if the MSE is unusually high.

### 3. TimesNet Probabilistic Training Script
**File:** `timesnet/train_timesnet_prob.py`

- **Loss Function:** Quantile Regression (Probabilistic Loss) using `MQLoss` with quantiles [0.1, 0.5, 0.9].
- **Model Save Name:** `timesnet_probabilistic_q`
- **Training Mechanism:**
  - **Data Loading:** Reads `nordbyen_features_engineered.csv` and converts it to NeuralForecast long format.
  - **Data Preprocessing:**
    - Filters relevant columns and removes NaNs.
    - Splits data into training and future datasets.
  - **Model Configuration:**
    - Horizon: 24 hours.
    - Input Size: 168 hours (7 days).
    - Loss: `MQLoss` with quantiles [0.1, 0.5, 0.9].
    - Scaler: Standard scaling.
    - Learning Rate: 1e-4.
    - Max Steps: 20.
  - **Training:**
    - Uses `NeuralForecast` to train the model.
  - **Prediction:**
    - Generates forecasts for the next 24 hours.
    - Outputs quantile predictions (e.g., `TimesNet-q-0.1`, `TimesNet-q-0.5`, `TimesNet-q-0.9`).
  - **Outputs:**
    - Saves the trained model to `models/timesnet_probabilistic_q`.
    - Saves a plot of actual vs. predicted quantiles to `results/stage3_prob_plot.png`.