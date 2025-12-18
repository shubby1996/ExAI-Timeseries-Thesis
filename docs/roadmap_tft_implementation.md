# TFT Model Training Roadmap

## Layer 1: Existing Scripts
- `align_data.py`
- `feature_engineering.py`
- `add_holidays.py` (Note: Logic integrated into `feature_engineering.py`)

## Layer 2: Darts/TFT Preprocessing & Splitting Pipeline

### Step 0 – Freeze and clarify Layer-1 output
**Goal:** Treat `nordbyen_features_engineered.csv` as the *canonical* feature dataset.
- **Tasks:**
    - Confirm holiday logic is consolidated (it is in `feature_engineering.py`).
    - Document Layer-1 contract:
        - **Input:** Raw heat + weather + `school_holidays.csv`.
        - **Output:** `nordbyen_features_engineered.csv`.
        - **Columns:**
            - `timestamp` (Index)
            - `heat_consumption` (Target)
            - **Weather:** `temp`, `dew_point`, `humidity`, `clouds_all`, `wind_speed`, `rain_1h`, `snow_1h`, `pressure`
            - **Time:** `hour`, `day_of_week`, `month`, `hour_sin`, `hour_cos`, `is_weekend`, `season`
            - **Lags:** `heat_lag_1h`, `heat_lag_24h`, `heat_rolling_24h`
            - **Interactions:** `temp_squared`, `temp_wind_interaction`, `temp_weekend_interaction`
            - **Holidays:** `public_holiday_name`, `is_public_holiday`, `school_holiday_name`, `is_school_holiday`
    - Ensure Layer-1 always overwrites this CSV with the same schema.

### Step 1 – Define feature roles for Darts/TFT
**Goal:** Define roles for Target, Past, Future, and Static covariates.
- **Target:** `heat_consumption`
- **Future-known covariates:**
    - Time: `hour_sin`, `hour_cos`, `day_of_week`, `month`, `is_weekend`, `season`
    - Holidays: `is_public_holiday`, `is_school_holiday`
- **Past-only covariates:**
    - Weather: `temp`, `dew_point`, `humidity`, `clouds_all`, `wind_speed`, `rain_1h`, `snow_1h`, `pressure`
    - Lags: `heat_lag_1h`, `heat_lag_24h`, `heat_rolling_24h`
    - Interactions: `temp_squared`, `temp_wind_interaction`, `temp_weekend_interaction`
- **Static covariates:** Optional `district_id = "nordbyen"`.

### Step 2 – Design data loading + schema validation
**Goal:** Create a robust loader.
- Reads `nordbyen_features_engineered.csv`.
- Validates:
    - `timestamp` index & frequency.
    - Existence of all required columns.
    - No duplicate timestamps.
    - No NaNs.

### Step 3 – Choose and formalize time-based splits
**Goal:** Define Train/Validation/Test splits.
- **Strategy:**
    - Train: Start → `T_train_end`
    - Validation: `T_train_end` → `T_val_end`
    - Test: `T_val_end` → End
- **Output:** 3 DataFrames/masks with `split` tags.

### Step 4 – Map DataFrames → Darts TimeSeries objects
**Goal:** Create Darts `TimeSeries` for each split.
- `series` (Target)
- `past_covariates` (Weather, Lags, Interactions)
- `future_covariates` (Time, Holidays)
- Define TFT hyperparameters: `input_chunk_length` (e.g., 168h), `output_chunk_length` (e.g., 24h).

### Step 5 – Decide on scaling/encoding strategy
**Goal:** Define preprocessing state.
- **Scalers:** Fit on **Train** only. Separate scalers for Target, Past, Future.
- **Categoricals:** Already encoded (0/1, integers).
- **State Storage:** Save scalers, feature roles, split dates, and hyperparameters (JSON + Pickle).

### Step 6 – Training & evaluation workflow
**Goal:** Standardize the training loop.
1. Run Layer-1 (if needed).
2. Run Layer-2 (Load, Validate, Split, Scale, Create TimeSeries).
3. Instantiate `TFTModel`.
4. Train on Train, Evaluate on Validation.
5. Save Model & Preprocessing State.

### Step 7 – Inference / forecasting workflow
**Goal:** Production-ready forecasting.
1. Load latest data (Layer-1).
2. Apply **saved** scalers/mappings (Step 5).
3. Construct TimeSeries (Last `input_chunk_length` for history, Next `output_chunk_length` for future).
4. Forecast using loaded model.
