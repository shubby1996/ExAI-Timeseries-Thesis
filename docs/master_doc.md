# ExAI Time Series Forecasting Project: Master Documentation

## 1. Project Overview & History

### **Objective**
The goal of this project is to develop high-performance deep learning models for forecasting **district heat consumption** ("Nordbyen"). Accurately predicting heat demand is critical for optimizing energy production and reducing waste.

### **Chronological Journey**
1.  **Project Initiation**: The project started with setting up a Python environment and organizing raw data (weather and heat consumption records).
2.  **Data Pipeline Construction**: We built a robust pipeline to align disparate data sources, engineer relevant features (lags, rolling windows, calendar events), and clean the dataset.
3.  **Model Selection**: We selected two state-of-the-art architectures:
    *   **NHITS** (Neural Hierarchical Interpolation for Time Series): A pure MLP-based model excellent for long-horizon forecasting.
    *   **TimesNet**: A temporal 2D-variation model that transforms 1D time series into 2D tensors to capture multi-scale variations.
4.  **Initial Benchmarking**: We created a benchmarking script (`benchmarker.py`) to rigorously evaluate these models using **walk-forward validation**. Initial runs were performed on CPU to verify logic.
5.  **GPU Acceleration**: To handle the computational load, we migrated to the Slurm cluster's GPU nodes (`rtx3080`). This required fixing environment issues and optimizing data loading.
6.  **Hyperparameter Optimization (HPO)**: Realizing that default parameters were suboptimal, we paused manual tuning and implemented an automated HPO system using **Optuna**. We tuned key architectural hyperparameters (e.g., layers, embedding sizes) and training parameters (learning rate, dropout).
7.  **Final Results**: The optimized models demonstrated significant improvements:
    *   **NHITS**: ~24% improvement in MAE.
    *   **TimesNet**: ~10% improvement in MAE.

---

## 2. Dataset & Preprocessing

### **Data Source**
*   **Target**: `heat_consumption` (Hourly district heating load).
*   **Exogenous Variables**: Weather data (`temp`, `wind_speed`, `humidity`, etc.) and calendar information.

### **Feature Engineering**
The `feature_engineering.py` script augments the raw data with:
*   **Time Features**: `hour`, `day_of_week`, `month`, `season`.
*   **Cyclical Encoding**: `hour_sin`, `hour_cos` (to preserve the continuity of time).
*   **Lag Features**: `heat_lag_1h`, `heat_lag_24h` (autoregressive signals).
*   **Rolling Statistics**: `heat_rolling_24h` (trend smoothing).
*   **Interactions**: `temp_wind_interaction` (wind chill effect), `temp_weekend` (occupancy patterns).
*   **Holidays**: Danish public holidays and school holidays.

### **Preprocessing (`model_preprocessing.py`)**
Before feeding data to models, we apply rigorous preprocessing:
1.  **Splitting**: Data is split chronologically into **Train** (up to 2018), **Validation** (2019), and **Test** (2020+).
2.  **Scaling**:
    *   `StandardScaler` (zero mean, unit variance) is fitted **strictly on the training set**.
    *   Separate scalers are used for the *Target*, *Past Covariates*, and *Future Covariates* to prevent data leakage.
3.  **TimeSeries Construction**: Data is converted into **Darts TimeSeries** objects (for NHITS) or **Pandas DataFrames** (for TimesNet/NeuralForecast).

---

## 3. Model Architectures

### **NHITS (Darts)**
*   **Type**: MLP-based Hierarchical Model.
*   **Mechanism**: Uses stacks of blocks with different pooling kernel sizes to capture frequencies at different scales (e.g., daily pattern vs. weekly trend).
*   **Input**:
    *   *Past Covariates*: Weather, lags, interactions.
    *   *Future Covariates*: Time features, holidays (known in advance).
*   **Why it works**: NHITS is computationally efficient and explicitly models the hierarchy of time series data, making it very effective for heat load which has strong daily and seasonal periodicities.

### **TimesNet (NeuralForecast)**
*   **Type**: CNN-based / FFT-based.
*   **Mechanism**: Uses Fast Fourier Transform (FFT) to identify significant periods. It reshapes the 1D series into 2D tensors based on these periods and applies Inception blocks (2D Convolutions) to capture intra-period and inter-period variations.
*   **Input**:
    *   *Future Exogenous*: Time features, holidays.
*   **Why it works**: By treating time series as 2D images of multiple periods, it captures complex temporal dependencies that standard RNNs or Transformers might miss.

---

## 4. Uncertainty Quantification

Both NHITS and TimesNet are configured to provide **probabilistic forecasts** rather than just point estimates. This is crucial for energy planning (e.g., knowing the 90% upper bound of demand).

### **Mechanism: Quantile Regression**
Instead of minimizing a simple error like MSE (Mean Squared Error), we minimize the **Calculated Quantile Loss** for a set of target quantiles: $\tau \in \{0.1, 0.5, 0.9\}$.
*   $q_{0.5}$ (Median): Serves as our primary point prediction (replacing the mean).
*   $q_{0.1}$ (10th percentile): The lower bound (10% chance demand is lower).
*   $q_{0.9}$ (90th percentile): The upper bound (90% chance demand is lower).

### **Implementation**
*   **NHITS**: Uses `darts.utils.likelihood_models.QuantileRegression(quantiles=[0.1, 0.5, 0.9])`.
*   **TimesNet**: Uses `neuralforecast.losses.pytorch.MQLoss(quantiles=[0.1, 0.5, 0.9])`.

### **Evaluation Metrics**
*   **PICP (Prediction Interval Coverage Probability)**: Percentage of actual values that fall between the predicted 10th and 90th percentiles. Ideally, this should be close to 80% ($0.9 - 0.1$).
*   **MIW (Mean Interval Width)**: The average distance between the upper ($q_{0.9}$) and lower ($q_{0.1}$) bounds. We want this to be as narrow as possible while maintaining high coverage.

---

## 5. Hyperparameter Optimization (HPO)

We built `hpo_tuner.py` to automate the search for the best model configurations using **Optuna**.

### **Strategy**
*   **Objective**: Minimize Mean Absolute Error (MAE) on the validation set.
*   **Pruning**: We used Optuna's pruners to stop unpromising trials early, saving GPU time.
*   **Search Spaces**:

**NHITS Search Space:**
*   `num_stacks`: [1-5] (Depth of hierarchy)
*   `num_blocks`: [1-3] (Complexity per stack)
*   `num_layers`: [1-4] (MLP depth)
*   `layer_widths`: [128, 256, 512, 1024] (Model capacity)
*   `lr`: Log-uniform [1e-5, 1e-2]
*   `dropout`: [0.0, 0.5]

**TimesNet Search Space:**
*   `hidden_size`: [32, 64, 128, 256]
*   `conv_hidden_size`: [32, 64, 128, 256]
*   `top_k`: [1-5] (Number of significant periods to model)
*   `lr`: Log-uniform [1e-5, 1e-2]
*   `dropout`: [0.0, 0.5]

---

## 6. Benchmarking & Testing

The `benchmarker.py` script performs the final rigorous evaluation.

### **Method: Walk-Forward Validation**
Instead of a single train/test split, we simulate real-world usage:
1.  Train on data up to time $T$.
2.  Predict the next 24 hours ($T+1$ to $T+24$).
3.  Record the predictions vs. actuals.
4.  Slide the window forward by 24 hours.
5.  Repeat for the entire test year (2020).

### **Why Walk-Forward?**
This method detects if a model degrades over time or fails to capture seasonal shifts, offering a much more honest assessment of performance than a simple hold-out set.

### **Metrics**
*   **MAE**: Mean Absolute Error (Primary metric).
*   **RMSE**: Root Mean Squared Error (Penalizes large errors).
*   **MAPE**: Mean Absolute Percentage Error.
*   **PICP/MIW**: Uncertainty metrics (Prediction Interval Coverage Probability / Mean Interval Width).

---

## 7. Execution Guide (Slurm)

To run this pipeline on the cluster:

**1. Run Hyperparameter Optimization:**
```bash
sbatch hpo_job.slurm NHITS 50    # Run 50 trials for NHITS
sbatch hpo_job.slurm TIMESNET 50 # Run 50 trials for TimesNet
```

**2. Run Final Benchmark:**
(Requires `results/best_params_*.json` from the HPO step)
```bash
sbatch train_script.slurm
```

**3. Outputs:**
*   `results/benchmark_results.csv`: Summary metrics.
*   `results/*_predictions.csv`: Detailed hourly predictions.

---

## 8. Detailed Implementation Walkthrough

Here is the exact step-by-step flow for each model type, from data loading to final prediction.

### **A. NHITS (Darts Implementation)**

1.  **Data Preparation (`model_preprocessing.py`)**
    *   **Loading**: `load_and_validate_features()` reads the CSV and sets the timestamp index.
    *   **Splitting**: `split_by_time()` cuts the data into Train (2015-2018), Val (2019), and Test (2020).
    *   **TimeSeries Conversion**: `build_timeseries_from_df()` converts Pandas DataFrames into Darts `TimeSeries` objects.
        *   **Target**: `heat_consumption`
        *   **Past Covariates**: Weather, lags, rolling stats (stacked into one multivariate series).
        *   **Future Covariates**: Calendar features, holidays (stacked into one multivariate series).
    *   **Scaling**: `fit_and_scale_splits()` fits a `StandardScaler` on the **Train** set only and transforms Val/Test.

2.  **Training (`benchmarker.py` -> `DartsAdapter.train`)**
    *   **Model Init**: `NHiTSModel` is initialized with hyperparameters from `results/best_params_NHITS.json` (e.g., `num_stacks=3`, `num_blocks=1`).
    *   **Covariate Stacking**: If both past and future covariates exist, they are often stacked for the model's `past_covariates` argument.
    *   **Fit**: `model.fit()` is called with the training target and covariates. It uses `val_series` for early stopping.

3.  **Inference (`benchmarker.py` -> `DartsAdapter.evaluate`)**
    *   **Walk-Forward Loop**: For each day in the test set:
        *   **History Lookup**: Grab the last 168 hours (7 days) of data ending at time $T$.
        *   **Future Lookup**: Grab the next 24 hours of future covariates (known ahead of time).
        *   **Predict**: `model.predict(n=24, series=history, past_covariates=history_covariates)`.
        *   **Uncertainty**: The model returns Monte Carlo samples (e.g., 100 samples). We calculate the 10th, 50th, and 90th percentiles from these samples to get our confidence intervals.

---

### **B. TimesNet (NeuralForecast Implementation)**

1.  **Data Preparation (`model_preprocessing.py` & `benchmarker.py`)**
    *   **DataFrame Format**: Unlike Darts, NeuralForecast requires a specific "long" DataFrame format with columns: `unique_id`, `ds` (date), and `y` (target).
    *   **Exogenous Handling**: We identify numerical columns (weather, holidays) and treat them as "Futr Exogenous" variables.
    *   **Splitting**: Slicing is done directly on the DataFrame based on the `ds` column.

2.  **Training (`benchmarker.py` -> `NeuralForecastAdapter.train`)**
    *   **Model Init**: `TimesNet` is initialized with parameters from `results/best_params_TIMESNET.json` (e.g., `top_k=3`, `hidden_size=64`).
    *   **Loss Function**: We explicitly set `loss=MQLoss(quantiles=[0.1, 0.5, 0.9])` to enable probabilistic forecasting.
    *   **Wrapper**: The model is wrapped in a `NeuralForecast` object.
    *   **Fit**: `nf.fit(df=train_df)` is called.

3.  **Inference (`benchmarker.py` -> `NeuralForecastAdapter.evaluate`)**
    *   **Walk-Forward Loop**: For each day in the test set:
        *   **History/Future Split**: We prepare a `hist_df` (past 168h) and `fut_df` (next 24h).
        *   **Predict**: `nf.predict(df=hist_df, futr_df=fut_df)`.
        *   **Output**: The model outputs a DataFrame with columns for each quantile (e.g., `TimesNet-q-0.1`, `TimesNet-median`, `TimesNet-q-0.9`). We extract these directly for our metrics.
