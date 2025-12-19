"""
Layer 2: Model preprocessing module.

This module provides:
- Feature configuration (defining Target, Past, and Future covariates)
- Data loading and schema validation
- Time-based train/val/test splitting
- TimeSeries conversion and scaling
- Inference helpers for loading and applying preprocessing state

It treats `nordbyen_features_engineered.csv` as input.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import pandas as pd
import pickle
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import sys

# Backward compatibility for unpickling old states
# This allows loading pickle files that reference 'tft_preprocessing'
sys.modules['tft_preprocessing'] = sys.modules[__name__]


@dataclass
class ModelFeatureConfig:
    """
    Configuration describing how columns in the engineered CSV
    map to Model / Darts concepts.
    
    This dataclass defines the schema mapping: which CSV columns play which role.

    time_col: column name for timestamps

    target_col: what you’re predicting

    past_covariates_cols: features you only know up to “now” (weather observed, lagged target, rolling stats, etc.)

    future_covariates_cols: features you know for future timestamps (calendar, holidays)

    static_covariates_cols: constant per-series features (unused here)

    The important part: this is the semantic contract between CSV columns and the model.

    """
    time_col: str = "timestamp"
    target_col: str = "heat_consumption"

    # Darts/TFT-style roles
    past_covariates_cols: List[str] = field(default_factory=list)
    future_covariates_cols: List[str] = field(default_factory=list)
    static_covariates_cols: List[str] = field(default_factory=list)


@dataclass
class PreprocessingState:
    """
    Holds everything needed to consistently preprocess data
    for models at training and inference time.
    """
    feature_config: ModelFeatureConfig
    target_scaler: Optional[Scaler] = None
    past_covariates_scaler: Optional[Scaler] = None
    future_covariates_scaler: Optional[Scaler] = None


def default_feature_config() -> ModelFeatureConfig:
    """
    Create the default feature-role mapping for nordbyen_features_engineered.csv.
    
    Returns
    -------
    ModelFeatureConfig
        Configuration with predefined feature roles:
        - Target: heat_consumption
        - Past covariates: weather features, lags, interactions
        - Future covariates: time features and holidays
    """
    return ModelFeatureConfig(
        time_col="timestamp",
        target_col="heat_consumption",
        past_covariates_cols=[
            "temp", "dew_point", "humidity", "clouds_all",
            "wind_speed", "rain_1h", "snow_1h", "pressure",
            "heat_lag_1h", "heat_lag_24h", "heat_rolling_24h",
            "temp_squared", "temp_wind_interaction", "temp_weekend_interaction",
        ],
        future_covariates_cols=[
            "hour", "hour_sin", "hour_cos",
            "day_of_week", "month", "is_weekend", "season",
            "is_public_holiday", "is_school_holiday",
        ],
        static_covariates_cols=[],
    )


def load_and_validate_features(
    csv_path: str,
    cfg: Optional[ModelFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Load nordbyen_features_engineered.csv and validate its schema.

    - Parses the time column as datetime and sets it as index.
    - Ensures that all required columns from cfg are present.
    - Sorts by time.
    - Checks for duplicate timestamps.

    Parameters
    ----------
    csv_path : str
        Path to the engineered features CSV file.
    cfg : ModelFeatureConfig, optional
        Feature configuration. If None, uses default_feature_config().

    Returns
    -------
    df : pd.DataFrame
        Validated DataFrame indexed by time.
        
    Raises
    ------
    ValueError
        If time column is missing, timestamps are duplicated, or required columns are missing.
    """
    if cfg is None:
        cfg = default_feature_config()

    # Load
    df = pd.read_csv(csv_path, parse_dates=[cfg.time_col])
    if cfg.time_col not in df.columns:
        raise ValueError(f"Expected time column '{cfg.time_col}' not found in {csv_path}")

    # Sort & index
    df = df.sort_values(cfg.time_col)
    df = df.set_index(cfg.time_col)

    # Check for duplicates
    if df.index.has_duplicates:
        raise ValueError("Time index has duplicate timestamps; please fix in Layer 1.")

    # Validate columns presence
    required_cols = [cfg.target_col] + cfg.past_covariates_cols + cfg.future_covariates_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in engineered dataset: {missing}")

    print(f"  [OK] Loaded and validated {len(df)} rows from {csv_path}")
    print(f"  [OK] Time range: {df.index.min()} to {df.index.max()}")
    print(f"  [OK] All {len(required_cols)} required columns present")

    return df


def split_by_time(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full DataFrame into train / val / test based on timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by time.
    train_end : pd.Timestamp
        Timestamp for the end of training period (inclusive).
    val_end : pd.Timestamp
        Timestamp for the end of validation period (inclusive).

    Returns
    -------
    train_df : pd.DataFrame
        Training subset.
    val_df : pd.DataFrame
        Validation subset (excludes overlapping row with train).
    test_df : pd.DataFrame
        Test subset (excludes overlapping row with val).
        
    Raises
    ------
    ValueError
        If DataFrame index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex before splitting.")

    # Ensure train_end and val_end have the same timezone as the DataFrame index
    if df.index.tz is not None:
        if train_end.tz is None:
            train_end = train_end.tz_localize(df.index.tz)
        else:
            train_end = train_end.tz_convert(df.index.tz)
        
        if val_end.tz is None:
            val_end = val_end.tz_localize(df.index.tz)
        else:
            val_end = val_end.tz_convert(df.index.tz)

    train_df = df.loc[:train_end]
    val_df = df.loc[train_end:val_end].iloc[1:]  # avoid overlapping row
    test_df = df.loc[val_end:].iloc[1:]         # avoid overlapping row

    print(f"\n[OK] Split summary:")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df


def build_timeseries_from_df(
    df: pd.DataFrame,
    cfg: Optional[ModelFeatureConfig] = None,
) -> Dict[str, Optional[TimeSeries]]:
    """
    Build Darts TimeSeries objects (target, past_covariates, future_covariates)
    from a single engineered DataFrame.

    Assumes df is indexed by time (DatetimeIndex) and already validated.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and feature columns.
    cfg : ModelFeatureConfig, optional
        Feature configuration. If None, uses default_feature_config().
    
    Returns
    -------
    dict
        Dictionary with keys: "target", "past_covariates", "future_covariates"
        Values are TimeSeries objects or None if no columns for that role.
    """
    if cfg is None:
        cfg = default_feature_config()

    # Use hourly frequency (we know our data is hourly)
    freq = 'H'

    # Target series (univariate)
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=[cfg.target_col],
        freq=freq,
    )

    # Past covariates (multivariate) – may be empty list
    past_covariates_series = None
    if cfg.past_covariates_cols:
        past_covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=cfg.past_covariates_cols,
            freq=freq,
        )

    # Future covariates (multivariate) – may be empty list
    future_covariates_series = None
    if cfg.future_covariates_cols:
        future_covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=cfg.future_covariates_cols,
            freq=freq,
        )

    return {
        "target": target_series,
        "past_covariates": past_covariates_series,
        "future_covariates": future_covariates_series,
    }


def fit_and_scale_splits(
    train_ts: Dict[str, Optional[TimeSeries]],
    val_ts: Dict[str, Optional[TimeSeries]],
    test_ts: Dict[str, Optional[TimeSeries]],
    cfg: Optional[ModelFeatureConfig] = None,
) -> Tuple[
    PreprocessingState,
    Dict[str, Optional[TimeSeries]],
    Dict[str, Optional[TimeSeries]],
    Dict[str, Optional[TimeSeries]],
]:
    """
    Fit scalers on training TimeSeries and apply them to train/val/test.

    Parameters
    ----------
    train_ts, val_ts, test_ts : dict
        Each dict should have keys: "target", "past_covariates", "future_covariates"
        with TimeSeries or None as values.
    cfg : ModelFeatureConfig, optional
        Feature configuration.

    Returns
    -------
    state : PreprocessingState
        Contains fitted scalers and the feature config.
    train_scaled, val_scaled, test_scaled : dict
        Same structure as input dicts, but with scaled TimeSeries.
    """
    if cfg is None:
        cfg = default_feature_config()

    # Initialize scalers
    target_scaler = Scaler()
    past_cov_scaler = Scaler() if train_ts["past_covariates"] is not None else None
    fut_cov_scaler = Scaler() if train_ts["future_covariates"] is not None else None

    # --- Target ---
    target_scaler.fit(train_ts["target"])
    train_target_scaled = target_scaler.transform(train_ts["target"])
    val_target_scaled = target_scaler.transform(val_ts["target"])
    test_target_scaled = target_scaler.transform(test_ts["target"])

    # --- Past covariates ---
    if past_cov_scaler is not None:
        past_cov_scaler.fit(train_ts["past_covariates"])
        train_past_scaled = past_cov_scaler.transform(train_ts["past_covariates"])
        val_past_scaled = past_cov_scaler.transform(val_ts["past_covariates"])
        test_past_scaled = past_cov_scaler.transform(test_ts["past_covariates"])
    else:
        train_past_scaled = val_past_scaled = test_past_scaled = None

    # --- Future covariates ---
    if fut_cov_scaler is not None:
        fut_cov_scaler.fit(train_ts["future_covariates"])
        train_future_scaled = fut_cov_scaler.transform(train_ts["future_covariates"])
        val_future_scaled = fut_cov_scaler.transform(val_ts["future_covariates"])
        test_future_scaled = fut_cov_scaler.transform(test_ts["future_covariates"])
    else:
        train_future_scaled = val_future_scaled = test_future_scaled = None

    state = PreprocessingState(
        feature_config=cfg,
        target_scaler=target_scaler,
        past_covariates_scaler=past_cov_scaler,
        future_covariates_scaler=fut_cov_scaler,
    )

    train_scaled = {
        "target": train_target_scaled,
        "past_covariates": train_past_scaled,
        "future_covariates": train_future_scaled,
    }
    val_scaled = {
        "target": val_target_scaled,
        "past_covariates": val_past_scaled,
        "future_covariates": val_future_scaled,
    }
    test_scaled = {
        "target": test_target_scaled,
        "past_covariates": test_past_scaled,
        "future_covariates": test_future_scaled,
    }

    print(f"\n[OK] Scalers fitted and applied:")
    print(f"  Target scaler: fitted on {len(train_ts['target'])} train samples")
    if past_cov_scaler:
        print(f"  Past covariates scaler: fitted on {len(train_ts['past_covariates'])} train samples")
    if fut_cov_scaler:
        print(f"  Future covariates scaler: fitted on {len(train_ts['future_covariates'])} train samples")

    return state, train_scaled, val_scaled, test_scaled


def prepare_model_data(
    csv_path: str,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    cfg: Optional[ModelFeatureConfig] = None,
) -> Tuple[
    PreprocessingState,
    Dict[str, Optional[TimeSeries]],
    Dict[str, Optional[TimeSeries]],
    Dict[str, Optional[TimeSeries]],
]:
    """
    End-to-end helper for preparing model-ready data from engineered CSV.
    
    This function:
    1. Loads and validates the CSV
    2. Splits by time into train/val/test
    3. Builds TimeSeries objects
    4. Fits scalers on train and transforms all splits
    
    Parameters
    ----------
    csv_path : str
        Path to nordbyen_features_engineered.csv
    train_end : pd.Timestamp
        End of training period (inclusive)
    val_end : pd.Timestamp
        End of validation period (inclusive)
    cfg : ModelFeatureConfig, optional
        Feature configuration
    
    Returns
    -------
    state : PreprocessingState
        Fitted scalers and configuration
    train_scaled, val_scaled, test_scaled : dict
        Scaled TimeSeries ready for model training
    """
    if cfg is None:
        cfg = default_feature_config()
    
    print("=" * 70)
    print("PREPARING MODEL DATA")
    print("=" * 70)
    
    # Step 1: Load and validate
    print("\n[1/4] Loading and validating CSV...")
    df_full = load_and_validate_features(csv_path, cfg)
    
    # Step 2: Split by time
    print("\n[2/4] Splitting by time...")
    train_df, val_df, test_df = split_by_time(df_full, train_end, val_end)
    
    # Step 3: Build TimeSeries
    print("\n[3/4] Building TimeSeries objects...")
    train_ts = build_timeseries_from_df(train_df, cfg)
    val_ts = build_timeseries_from_df(val_df, cfg)
    test_ts = build_timeseries_from_df(test_df, cfg)
    print(f"  - Target shape: {train_ts['target'].values().shape}")
    print(f"  - Past covariates shape: {train_ts['past_covariates'].values().shape if train_ts['past_covariates'] else 'None'}")
    print(f"  - Future covariates shape: {train_ts['future_covariates'].values().shape if train_ts['future_covariates'] else 'None'}")
    
    # Step 4: Fit and scale
    print("\n[4/4] Fitting scalers and transforming...")
    state, train_scaled, val_scaled, test_scaled = fit_and_scale_splits(
        train_ts, val_ts, test_ts, cfg
    )
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\n[OK] Ready for model training!")
    
    return state, train_scaled, val_scaled, test_scaled


def load_preprocessing_state(path: str) -> PreprocessingState:
    """
    Load a previously saved PreprocessingState from disk.
    
    This function deserializes the preprocessing state that was saved during
    training, including the feature configuration and all fitted scalers.
    
    Parameters
    ----------
    path : str
        Path to the saved preprocessing state pickle file.
    
    Returns
    -------
    PreprocessingState
        Loaded preprocessing state with fitted scalers and feature config.
    
    Examples
    --------
    >>> state = load_preprocessing_state("models/tft_preprocessing_state.pkl")
    >>> print(f"Target scaler fitted: {state.target_scaler is not None}")
    """
    with open(path, "rb") as f:
        state: PreprocessingState = pickle.load(f)
    return state


def apply_state_to_full_df(
    df_full: pd.DataFrame,
    state: PreprocessingState,
) -> Dict[str, Optional[TimeSeries]]:
    """
    Apply saved preprocessing state to a full DataFrame for inference.
    
    Given the full engineered DataFrame and a fitted PreprocessingState,
    this function rebuilds the TimeSeries and applies the stored scalers.
    This ensures that inference data is transformed consistently with
    the training data.
    
    Parameters
    ----------
    df_full : pd.DataFrame
        Full engineered dataset, indexed by time and already validated.
        Should have the same schema as the training data.
    state : PreprocessingState
        Contains feature config and fitted scalers from training.
    
    Returns
    -------
    scaled_series : dict
        Dictionary with keys: "target", "past_covariates", "future_covariates"
        and values as scaled TimeSeries (or None if not applicable).
    
    Examples
    --------
    >>> df = load_and_validate_features("data.csv")
    >>> state = load_preprocessing_state("state.pkl")
    >>> scaled = apply_state_to_full_df(df, state)
    >>> print(scaled["target"].values().shape)
    """
    cfg = state.feature_config

    # Build unscaled TimeSeries using the same config as training
    ts_dict = build_timeseries_from_df(df_full, cfg)

    # Apply the fitted scalers from training
    target_scaled = state.target_scaler.transform(ts_dict["target"])

    if state.past_covariates_scaler is not None and ts_dict["past_covariates"] is not None:
        past_scaled = state.past_covariates_scaler.transform(ts_dict["past_covariates"])
    else:
        past_scaled = None

    if state.future_covariates_scaler is not None and ts_dict["future_covariates"] is not None:
        future_scaled = state.future_covariates_scaler.transform(ts_dict["future_covariates"])
    else:
        future_scaled = None

    return {
        "target": target_scaled,
        "past_covariates": past_scaled,
        "future_covariates": future_scaled,
    }


# Aliases for backward compatibility (pickling)
TFTFeatureConfig = ModelFeatureConfig
default_tft_feature_config = default_feature_config
prepare_tft_data = prepare_model_data


def append_future_calendar_and_holidays(
    df_full: pd.DataFrame,
    n_future: int,
    freq: str = "H",
    school_holidays_path: Optional[str] = None,
    country: str = "DK",
) -> pd.DataFrame:
    """
    Append `n_future` new rows with future timestamps and calendar/holiday features
    to the existing engineered DataFrame.

    This mirrors the logic used in feature_engineering.py for:
    - hour, day_of_week, month
    - hour_sin, hour_cos
    - is_weekend
    - season (Winter=0, Spring=1, Summer=2, Fall=3)
    - is_public_holiday (DK)
    - is_school_holiday (from school_holidays.csv)

    Past-only covariates (weather, lags, etc.) are left as NaN for future rows
    and will not be used by TFT for the prediction horizon.

    Parameters
    ----------
    df_full : pd.DataFrame
        Engineered dataset indexed by time (DatetimeIndex).
    n_future : int
        Number of future time steps to append.
    freq : str, optional
        Frequency string (default 'H' for hourly).
    school_holidays_path : str, optional
        Path to school_holidays.csv. If None, defaults to 'school_holidays.csv'
        in current directory.
    country : str, optional
        Country code for public holidays (default 'DK' for Denmark).

    Returns
    -------
    df_extended : pd.DataFrame
        Original DataFrame plus future rows with calendar/holiday features.
        Past covariates (weather, lags) are NaN for future rows.

    Examples
    --------
    >>> df = load_and_validate_features("data.csv")
    >>> df_extended = append_future_calendar_and_holidays(df, n_future=24)
    >>> # Now df_extended has 24 future hours with calendar features
    
    Raises
    ------
    ValueError
        If df_full is not indexed by DatetimeIndex or n_future <= 0.
    """
    import numpy as np
    import holidays
    import os
    
    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise ValueError("df_full must be indexed by a DatetimeIndex.")

    if n_future <= 0:
        return df_full

    last_ts = df_full.index.max()
    future_index = pd.date_range(
        start=last_ts + pd.tseries.frequencies.to_offset(freq),
        periods=n_future,
        freq=freq
    )

    # Create future dataframe
    df_future = pd.DataFrame(index=future_index)

    # --- Time features ---
    df_future["hour"] = df_future.index.hour
    df_future["day_of_week"] = df_future.index.dayofweek  # 0=Mon, 6=Sun
    df_future["month"] = df_future.index.month

    # Cyclical encoding for hour
    df_future["hour_sin"] = np.sin(2 * np.pi * df_future["hour"] / 24)
    df_future["hour_cos"] = np.cos(2 * np.pi * df_future["hour"] / 24)

    # Weekend flag
    df_future["is_weekend"] = df_future["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Season (same mapping as in feature_engineering.py):
    # Winter=0, Spring=1, Summer=2, Fall=3 using (month%12 + 3)//3 - 1
    df_future["season"] = df_future["month"].apply(lambda x: (x % 12 + 3) // 3 - 1)

    # --- Holiday features ---
    # Public holidays
    years = sorted(df_future.index.year.unique().tolist())
    dk_holidays = holidays.country_holidays(country, years=years)

    df_future["date"] = df_future.index.date
    df_future["public_holiday_name"] = df_future["date"].apply(lambda x: dk_holidays.get(x))
    df_future["is_public_holiday"] = df_future["public_holiday_name"].notna().astype(int)
    df_future["public_holiday_name"] = df_future["public_holiday_name"].fillna("None")

    # School holidays
    if school_holidays_path is None:
        school_holidays_path = "school_holidays.csv"
    
    if os.path.exists(school_holidays_path):
        school_hol = pd.read_csv(school_holidays_path, parse_dates=["start_date", "end_date"])
        school_holiday_map = {}
        for _, row in school_hol.iterrows():
            date_range = pd.date_range(start=row["start_date"], end=row["end_date"])
            for d in date_range:
                school_holiday_map[d.date()] = row["description"]

        df_future["school_holiday_name"] = df_future["date"].map(school_holiday_map)
        df_future["is_school_holiday"] = df_future["school_holiday_name"].notna().astype(int)
        df_future["school_holiday_name"] = df_future["school_holiday_name"].fillna("None")
    else:
        # If no school_holidays.csv, mirror training fallback
        df_future["is_school_holiday"] = 0
        df_future["school_holiday_name"] = "None"

    # Drop temporary date column
    df_future.drop(columns=["date"], inplace=True)

    # Ensure all columns present in df_full exist in df_future, even if NaN
    for col in df_full.columns:
        if col not in df_future.columns:
            df_future[col] = np.nan

    # Reorder columns to match df_full
    df_future = df_future[df_full.columns]

    # Concatenate
    df_extended = pd.concat([df_full, df_future], axis=0)

    return df_extended
