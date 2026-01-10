# Tommerby Water Consumption - Feature Engineering Pipeline

## Overview

This folder contains the complete data preprocessing and feature engineering pipeline for the **Tommerby (Water B)** water consumption forecasting project. The pipeline transforms raw water consumption data and weather observations into a feature-rich dataset ready for time series modeling.

**Input Data:**
- `../../raw_data/water_b_tommerby/tommerby_CombinedDataframe_SummedAveraged_withoutOutliers.csv` - Raw water consumption data
- `../../raw_data/weather/Weather_Bronderslev_20152022.csv` - Weather observations from Brønderslev

**Output Data:**
- `tommerby_features_engineered.csv` - Final feature-engineered dataset with hourly observations

---

## Pipeline Architecture

The pipeline consists of two main stages orchestrated by `build_features_tommerby.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 1: Data Alignment                      │
│                    (align_data_tommerby.py)                      │
│                                                                   │
│  Water Data + Weather Data → tommerby_water_weather_aligned.csv │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 2: Feature Engineering                    │
│              (feature_engineering_tommerby.py)                   │
│                                                                   │
│  Aligned Data → Feature-Rich → tommerby_features_engineered.csv │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Alignment (`align_data_tommerby.py`)

### Purpose
Merges water consumption time series with weather observations on a common hourly timestamp index, ensuring temporal consistency between target variable and meteorological covariates.

### Input Processing

#### Water Consumption Data
- **Source:** `tommerby_CombinedDataframe_SummedAveraged_withoutOutliers.csv`
- **Format:** Two-column CSV with timestamp and consumption value
- **Timestamp Format:** UTC with timezone info (e.g., `2015-05-31 23:00:00+00:00`)
- **Target Variable:** `water_consumption` (cubic meters per hour)

#### Weather Data
- **Source:** `Weather_Bronderslev_20152022.csv`
- **Format:** Multi-column CSV with weather observations
- **Timestamp Column:** `dt_iso` in format `YYYY-MM-DD HH:MM:SS +0000 UTC`
- **Features:** Temperature, humidity, wind speed, clouds, precipitation, pressure, etc.

### Processing Steps

1. **Load Water Data**
   - Read CSV with custom column names: `['timestamp', 'water_consumption']`
   - Parse timestamps to pandas datetime with UTC timezone awareness
   - Set timestamp as index

2. **Load Weather Data**
   - Parse `dt_iso` column with explicit format specification
   - Ensure timezone consistency (UTC)
   - Set timestamp as index
   - Drop redundant columns: `dt_iso`, `city_name`, `lat`, `lon`, `sea_level`, `grnd_level`

3. **Align Datasets**
   - Perform inner join on timestamp index
   - Keep only timestamps present in both datasets (intersection)
   - Ensures complete feature coverage for all water consumption observations

4. **Handle Missing Values**
   - Apply forward fill (`ffill`) first
   - Apply backward fill (`bfill`) for any remaining NaN values
   - Guarantees no missing data in aligned dataset

### Output
- **File:** `tommerby_water_weather_aligned.csv`
- **Contents:** Hourly water consumption with all weather features aligned by timestamp

---

## Stage 2: Feature Engineering (`feature_engineering_tommerby.py`)

### Purpose
Transforms aligned raw data into a feature-rich dataset with temporal patterns, lag features, interaction terms, and holiday indicators suitable for deep learning time series models.

### Feature Categories

#### 1. Feature Selection (Input Features)
Selected weather features:
- `temp` - Air temperature (°C)
- `dew_point` - Dew point temperature (°C)
- `humidity` - Relative humidity (%)
- `clouds_all` - Cloud coverage (%)
- `wind_speed` - Wind speed (m/s)
- `rain_1h` - Rainfall in last hour (mm)
- `snow_1h` - Snowfall in last hour (mm)
- `pressure` - Atmospheric pressure (hPa)

Target variable:
- `water_consumption` - Hourly water consumption (m³/h)

#### 2. Time Features
**Raw Temporal Features:**
- `hour` - Hour of day (0-23)
- `day_of_week` - Day of week (0=Monday, 6=Sunday)
- `month` - Month of year (1-12)
- `season` - Season index (0=Winter, 1=Spring, 2=Summer, 3=Fall)

**Cyclical Encoding:**
- `hour_sin` - Sine component of hour (captures cyclical nature)
- `hour_cos` - Cosine component of hour (captures cyclical nature)

**Binary Indicators:**
- `is_weekend` - Weekend flag (1 for Sat/Sun, 0 otherwise)

#### 3. Lag Features (Demand History)
Captures temporal dependencies and autocorrelation:
- `water_lag_1h` - Water consumption 1 hour ago
- `water_lag_24h` - Water consumption 24 hours ago (daily pattern)
- `water_rolling_24h` - 24-hour rolling mean of water consumption

#### 4. Interaction & Polynomial Features
Captures non-linear relationships:
- `temp_squared` - Quadratic temperature effect
- `temp_wind_interaction` - Wind chill proxy (temperature × wind speed)
- `temp_weekend_interaction` - Temperature effect during weekends

#### 5. Holiday Features
**Public Holidays:**
- Uses `holidays` Python package for Danish (DK) public holidays
- `is_public_holiday` - Binary indicator (1 for public holidays)
- `public_holiday_name` - Name of the holiday (or "None")

**School Holidays:**
- Loaded from `../school_holidays.csv` if available
- `is_school_holiday` - Binary indicator (1 for school holidays)
- `school_holiday_name` - Description of school holiday period (or "None")

### Processing Steps

1. **Load Aligned Data**
   - Read `tommerby_water_weather_aligned.csv`
   - Parse timestamp column and set as index

2. **Feature Selection**
   - Filter columns to keep only target and selected weather features
   - Remove unused weather columns

3. **Engineer Time Features**
   - Extract temporal components (hour, day, month, season)
   - Create cyclical encodings for hour
   - Generate weekend and seasonal indicators

4. **Create Lag Features**
   - Generate 1-hour and 24-hour lags of water consumption
   - Compute 24-hour rolling average

5. **Generate Interaction Features**
   - Create polynomial and interaction terms
   - Capture non-linear weather effects

6. **Add Holiday Features**
   - Identify Danish public holidays for all years in dataset
   - Map school holidays if CSV file exists
   - Create binary indicators and name columns

7. **Clean Data**
   - Drop NaN values introduced by lag operations
   - Report number of rows dropped

### Output
- **File:** `tommerby_features_engineered.csv`
- **Features:** ~30+ columns including target, weather, time, lag, interaction, and holiday features
- **Observations:** Slightly fewer than aligned data due to lag-induced NaN removal

---

## Running the Pipeline

### Option 1: Run Complete Pipeline
```bash
cd processing/tommerby_processing
python build_features_tommerby.py
```

This executes both stages sequentially:
1. Data alignment
2. Feature engineering

### Option 2: Run Individual Stages
```bash
# Stage 1 only
python align_data_tommerby.py

# Stage 2 only (requires aligned data to exist)
python feature_engineering_tommerby.py
```

### Option 3: Selective Execution
```python
from build_features_tommerby import build_tommerby_features

# Skip alignment if aligned file already exists and is up-to-date
build_tommerby_features(run_align=False, run_feature_engineering=True)

# Run only alignment
build_tommerby_features(run_align=True, run_feature_engineering=False)
```

---

## Data Quality & Validation

### Missing Value Handling
- **Alignment Stage:** Forward/backward fill ensures no missing weather data
- **Feature Engineering Stage:** Drops rows with NaN from lag features (typically first 24 hours)

### Temporal Consistency
- All timestamps are UTC-aligned
- Inner join ensures complete feature coverage
- No gaps in hourly observations after processing

### Feature Integrity
- Cyclical features properly normalized (sin/cos in [-1, 1])
- Holiday features validated against official calendars
- Lag features maintain temporal ordering

---

## Dependencies

```python
pandas>=1.3.0
numpy>=1.20.0
holidays>=0.14  # For Danish public holidays
```

**Optional:**
- `school_holidays.csv` in parent directory (`../school_holidays.csv`)
  - Columns: `start_date`, `end_date`, `description`
  - If missing, school holiday features set to 0

---

## Next Steps

After running this pipeline:
1. The output `tommerby_features_engineered.csv` is ready for model training
2. Use with `water_tommerby_benchmark/scripts/run_benchmarker_water.py`
3. Models will automatically handle feature scaling and train/val/test splits

---

## Notes

- The pipeline mirrors the Centrum water processing pipeline for consistency
- Weather data source is identical (Brønderslev station) for fair comparison
- Feature engineering choices match Centrum pipeline to enable comparative analysis
- Modify `feature_engineering_tommerby.py` to add custom features if needed
