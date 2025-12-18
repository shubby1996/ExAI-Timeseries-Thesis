# The Data Science Journey: Nordbyen Heat Forecasting

This document chronicles the evolution of our forecasting pipeline, explaining the "why" and "how" behind every technical decision. It is designed to onboard new data scientists by walking through the project chronologically.

---

## Chapter 1: The Foundation (Data Engineering)

Before any modeling, we had to solve the **Data Alignment Problem**.

### 1.1 The Raw Materials
We started with two disconnected sources:
1.  **Heat Consumption**: Hourly timestamps, aggregated for the district.
2.  **Weather Data**: OpenWeatherMap API data (Brønderslev), containing temperature, wind, humidity, etc.

### 1.2 The Alignment Strategy (`align_data.py`)
**The Challenge**: Real-world data is messy. Timestamps don't match perfect hours, and sensors fail.
**The Solution**:
*   **Timezone Standardization**: Everything was forced to **UTC**. This is critical to avoid "Daylight Savings Time" bugs where an hour disappears or repeats in October/March.
    ```python
    # align_data.py
    if df_weather['timestamp'].dt.tz is None:
        df_weather['timestamp'] = df_weather['timestamp'].dt.tz_localize('UTC')
    else:
        df_weather['timestamp'] = df_weather['timestamp'].dt.tz_convert('UTC')
    ```
*   **Inner Join**: We only kept timestamps where we had *both* heat and weather data.
    ```python
    # align_data.py
    df_aligned = df_heat.join(df_weather, how='inner')
    ```
*   **Imputation**: We used `ffill` (forward fill) followed by `bfill` to handle small gaps.
    *   *Data Science Rationale*: Weather doesn't jump instantly. If we miss a reading at 14:00, the 13:00 temperature is a safe estimate.
    ```python
    # align_data.py
    df_aligned.fillna(method='ffill', inplace=True)
    df_aligned.fillna(method='bfill', inplace=True)
    ```

---

## Chapter 2: Encoding Domain Physics (Feature Engineering)

We didn't just throw raw data at the model. We engineered features to teach the model about **Thermodynamics** and **Human Behavior**.

### 2.1 Thermodynamics (Weather Features)
*   **Non-linearity**: Heat demand isn't linear. We added `temp_squared` because heating ramps up aggressively as it gets very cold.
*   **Wind Chill**: We created `temp * wind_speed`. A windy 0°C day drains more heat from buildings than a calm 0°C day.
    ```python
    # feature_engineering.py
    df['temp_squared'] = df['temp'] ** 2
    df['temp_wind_interaction'] = df['temp'] * df['wind_speed']
    ```
*   **Thermal Inertia**: Buildings are heavy thermal batteries. They don't cool down instantly.
    *   `heat_lag_1h`: What was the demand 1 hour ago? (Autocorrelation)
    *   `heat_rolling_24h`: What was the average demand over the last day?
    ```python
    # feature_engineering.py
    df['heat_lag_1h'] = df['heat_consumption'].shift(1)
    df['heat_rolling_24h'] = df['heat_consumption'].rolling(window=24).mean()
    ```

### 2.2 Human Behavior (Temporal Features)
*   **Cyclical Time**: We didn't use 0-23 for hours. We used **Sine/Cosine transformations**.
    *   *Why?* In a linear 0-23 system, 23:00 and 00:00 are far apart. In Sin/Cos space, they are neighbors. This preserves the "circle" of time.
    ```python
    # feature_engineering.py
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    ```
*   **The "Weekend Effect"**: We created `is_weekend` and `temp * is_weekend`. People stay home on weekends, changing the heating profile.

### 2.3 The "Holiday Problem" (`add_holidays.py` / `feature_engineering.py`)
Standard models fail on Christmas.
*   **Solution**: We integrated the `holidays` Python package for public holidays.
    ```python
    # feature_engineering.py
    dk_holidays = holidays.DK(years=years)
    df['public_holiday_name'] = df['date'].apply(lambda x: dk_holidays.get(x))
    ```
*   **The Extra Mile**: We manually researched and created `school_holidays.csv` for local school breaks, as these significantly impact district heating loads.

---

## Chapter 3: The Brain (TFT Architecture)

We chose the **Temporal Fusion Transformer (TFT)**.

### 3.1 Why TFT?
Most models (LSTM, ARIMA) are "black boxes" or too simple. TFT is designed specifically for multi-horizon forecasting with mixed inputs:
1.  **Static Covariates**: (Not used here, but capable of handling location embeddings).
2.  **Past Covariates**: Things we only know *historically* (Weather, Lagged Heat).
3.  **Future Covariates**: Things we know *in advance* (Time, Holidays).

### 3.2 The Configuration (`tft_preprocessing.py`)
*   **Input Chunk (168 hours)**: The model looks back **7 days**.
    *   *DS Rationale*: This captures exactly one full weekly cycle. It can "see" that last Tuesday was similar to this Tuesday.
*   **Output Chunk (24 hours)**: We predict the next full day at once.

```python
# tft_preprocessing.py
def default_tft_feature_config():
    return TFTFeatureConfig(
        target_col="heat_consumption",
        past_covariates_cols=[
            "temp", "dew_point", "humidity", "clouds_all",
            "wind_speed", "rain_1h", "snow_1h", "pressure",
            "heat_lag_1h", "heat_lag_24h", "heat_rolling_24h",
            "temp_squared", "temp_wind_interaction", "temp_weekend_interaction"
        ],
        future_covariates_cols=[
            "hour", "hour_sin", "hour_cos", "day_of_week", "month", 
            "is_weekend", "season",
            "is_public_holiday", "is_school_holiday"
        ]
    )
```

---

## Chapter 4: The Training Ground (`train_tft_nordbyen.py`)

This is the "nitty gritty" of how the model learns.

### 4.1 The Loss Function: Quantile Loss
We don't just predict the "mean" (average). We predict **Uncertainty**.
*   The model minimizes **Quantile Loss** (Pinball Loss).
*   It learns to output the **10th, 50th (Median), and 90th percentiles**.
*   *Result*: We get a "confidence interval". If the interval is wide, the model is saying "I'm not sure" (e.g., volatile weather).

### 4.2 The Mechanics of Training: Anatomy of a Batch

This is where the magic happens. The model doesn't see the whole timeline at once. It learns by practicing on thousands of small "windows" or "chunks" cut from the training data.

#### The Sliding Window Mechanism
Imagine a sliding window moving across our 3-year training history (2015-2018).
For **every single hour** in the dataset, we create a training sample consisting of:

1.  **The Encoder (Input) Chunk**: `input_chunk_length = 168` hours (7 days).
    *   *What the model sees*: The past.
    *   *Content*:
        *   **Target**: Heat consumption history ($t_{-168}$ to $t_{0}$).
        *   **Past Covariates**: Observed weather ($t_{-168}$ to $t_{0}$).
        *   **Future Covariates (Past part)**: Time/Calendar features for the past 7 days.

2.  **The Decoder (Output) Chunk**: `output_chunk_length = 24` hours (1 day).
    *   *What the model predicts*: The future.
    *   *Content*:
        *   **Future Covariates (Future part)**: We *know* it will be Monday tomorrow. We *know* it will be Christmas. The model gets this "known future" info for $t_{+1}$ to $t_{+24}$.
        *   **Target (Hidden)**: The actual heat consumption for the next 24 hours is hidden from the model input but used to calculate the loss.

#### Inside the "Black Box": Data Flow
When a batch of 64 of these windows enters the TFT:

1.  **Variable Selection Network**:
    *   The model looks at all inputs (Temperature, Wind, Hour of Day, Lagged Heat) and decides *which ones matter right now*.
    *   *Example*: "It's summer, so wind speed matters less. It's Monday morning, so 'Day of Week' matters a lot."

2.  **LSTM Encoders**:
    *   The 168 hours of history flow through an LSTM (Long Short-Term Memory) layer.
    *   This creates a "memory state" summarizing the last week (e.g., "The building is currently cold and heating up slowly").

3.  **Temporal Attention Layer**:
    *   The model looks back across the 168 hours to find specific patterns relevant to the *current* prediction.
    *   *Example*: To predict 08:00 tomorrow, it might pay strong attention to 08:00 yesterday and 08:00 last week.

4.  **The Prediction**:
    *   Combining the "Memory" (LSTM) + "Patterns" (Attention) + "Known Future" (Calendar), it outputs the forecast for the next 24 hours.

#### The Loss Calculation
*   The model outputs 3 values for each hour: 10th, 50th, and 90th percentiles.
*   We compare these to the **Actual Target** (which we hid).
*   **Backpropagation**: If the actual value was 50MW but the model predicted 40MW, we tweak the weights to push the prediction up.

### 4.3 Training Strategy
1.  **Splitting**:
    *   **Train**: 2015-2018 (The textbook).
    *   **Val**: 2019 (The practice exam).
    *   **Test**: 2020-2022 (The final exam).
    ```python
    # train_tft_nordbyen.py
    train_end="2018-12-31 23:00:00+00:00"
    val_end="2019-12-31 23:00:00+00:00"
    ```
2.  **Scaling**: CRITICAL STEP.
    *   We fit scalers (`MinMaxScaler`) **ONLY on Training data**.
    *   We apply those same scalers to Validation and Test.
    *   *Why?* If we scaled using the whole dataset, we would leak future information (Data Leakage), making our results fake.
    ```python
    # tft_preprocessing.py
    target_scaler.fit(train_ts["target"])
    # Transform ALL splits using the scaler fitted on TRAIN
    train_target_scaled = target_scaler.transform(train_ts["target"])
    val_target_scaled = target_scaler.transform(val_ts["target"])
    ```
3.  **Early Stopping**:
    *   We monitor `val_loss`. If it doesn't improve for **5 epochs**, we stop.
    *   *Benefit*: Prevents overfitting (memorizing the training data).
    ```python
    # train_tft_nordbyen.py
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001,
        mode="min",
    )
    ```

---

## Chapter 5: Reality Check (Evaluation & Uncertainty)

We don't trust a single plot. We use **Walk-Forward Validation** (`evaluate_tft_nordbyen.py`).

### 5.1 The Process
Instead of a simple train/test split, we simulate the real world:
1.  Stand at hour $T$.
2.  Predict $T+1$ to $T+24$.
3.  Step forward 24 hours.
4.  Repeat.

```python
# evaluate_tft_nordbyen.py
for i in range(n_predictions):
    start_idx = i * stride
    # ...
    pred_scaled = model.predict(
        n=output_chunk,
        series=historical_target,
        # ...
        num_samples=100,  # Generate probabilistic forecast
    )
```

### 5.2 Measuring Uncertainty
We added specific metrics to quantify our confidence:
*   **PICP (Prediction Interval Coverage Probability)**: We aim for 80% of actual values to fall between our 10th and 90th percentiles.
    ```python
    # evaluate_tft_nordbyen.py
    def calculate_picp(y_true, y_low, y_high):
        within_interval = (y_true >= y_low) & (y_true <= y_high)
        return np.mean(within_interval) * 100
    ```
*   **MIW (Mean Interval Width)**: How "sharp" are our predictions? Narrower is better, *if* accurate.

---

## Chapter 6: The Future (Production Inference)

The final piece of the puzzle was **True Forward Forecasting** (`predict_tft_nordbyen.py`).

### 6.1 The "Cold Start" Misconception
A common question is: *"Does the model just need tomorrow's date to predict tomorrow?"*
**NO.** The model is a sequence-to-sequence model. It **ALWAYS** needs context.

To predict $T_{+1} \dots T_{+24}$ (Tomorrow), you must provide:
1.  **The Past (Encoder Input)**: The *actual* observed data for $T_{-168} \dots T_{0}$ (The last 7 days).
    *   *Why?* The model needs to know "Is it currently cold?", "Was yesterday a holiday?", "Is the trend going up?".
2.  **The Future (Decoder Input)**: The *known* features for $T_{+1} \dots T_{+24}$.
    *   *Why?* The model needs to know "Is tomorrow a weekend?", "Is tomorrow Christmas?".

### 6.2 The Solution: `append_future_calendar_and_holidays`
We built a pipeline that automatically stitches these two parts together.

1.  **Load History**: We load the most recent data available (e.g., up to "Now").
2.  **Generate Future**: We calculate the calendar features for the next 24 hours.
3.  **Stitch & Feed**: We pass the *combined* timeline to the model. The model automatically grabs the last 168 hours of "Past" and the next 24 hours of "Future".

```python
# tft_preprocessing.py
def append_future_calendar_and_holidays(df_full, n_future, ...):
    # 1. Identify where "Now" is
    last_ts = df_full.index.max()
    
    # 2. Create timestamps for "Tomorrow"
    future_index = pd.date_range(start=last_ts + freq, periods=n_future, freq=freq)
    
    # 3. Calculate known features (Sin/Cos, Holidays)
    df_future["hour_sin"] = np.sin(2 * np.pi * df_future["hour"] / 24)
    df_future["is_public_holiday"] = ...
    
    # 4. Concatenate: History + Future
    # Past columns (Weather, Lags) are NaN for the future rows, which is fine 
    # because the TFT Decoder ONLY looks at Future Covariates.
    df_extended = pd.concat([df_full, df_future], axis=0)
    return df_extended
```

```python
# predict_tft_nordbyen.py
# The model.predict() method handles the slicing internally:
# It looks at the END of the provided series.
pred_scaled = model.predict(
    n=24,
    series=target_scaled,       # Contains history up to T
    past_covariates=past_scaled, # Contains history up to T
    future_covariates=future_scaled # Contains history T + Future T+24
)
```

---

### Summary Checklist for Juniors
When touching this codebase, remember:
1.  **Never fit scalers on test data.**
2.  **Always use UTC.**
3.  **TFT needs 168 hours of history** to make a prediction.
4.  **Uncertainty is a feature, not a bug.** Use the 10th/90th percentiles to communicate risk.
