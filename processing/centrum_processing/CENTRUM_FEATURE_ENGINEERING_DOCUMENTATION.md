# Centrum Water Consumption - Feature Engineering Pipeline

## Overview

This folder contains the complete data preprocessing and feature engineering pipeline for the **Centrum (Water A)** water consumption forecasting project. The pipeline transforms raw water consumption data and weather observations into a feature-rich dataset ready for time series modeling.

**Input Data:**
- `../water_a_centrum/centrum_CombinedDataframe_SummedAveraged_withoutOutliers.csv` - Raw water consumption data
- `../weather/Weather_Bronderslev_20152022.csv` - Weather observations from Brønderslev

**Output Data:**
- `centrum_features_engineered.csv` - Final feature-engineered dataset with 47,830 hourly observations

---

## Pipeline Architecture

The pipeline consists of two main stages orchestrated by `build_features_centrum.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 1: Data Alignment                      │
│                    (align_data_centrum.py)                       │
│                                                                   │
│  Water Data + Weather Data → centrum_water_weather_aligned.csv  │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 2: Feature Engineering                    │
│              (feature_engineering_centrum.py)                    │
│                                                                   │
│  Aligned Data → Feature-Rich → centrum_features_engineered.csv  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Alignment (`align_data_centrum.py`)

### Purpose
Merges water consumption time series with weather observations on a common hourly timestamp index, ensuring temporal consistency between target variable and meteorological covariates.

### Input Processing

#### Water Consumption Data
- **Source:** `centrum_CombinedDataframe_SummedAveraged_withoutOutliers.csv`
- **Format:** Two-column CSV with timestamp and consumption value
- **Original Shape:** 47,854 hourly observations
- **Timestamp Format:** UTC with timezone info (e.g., `2015-05-31 23:00:00+00:00`)
- **Target Variable:** `water_consumption` (cubic meters per hour)

**Processing Steps:**
1. Parse timestamps as timezone-aware datetime objects (UTC)
2. Set timestamp as DataFrame index for time-based operations
3. Rename consumption column to `water_consumption` for clarity

#### Weather Data
- **Source:** `Weather_Bronderslev_20152022.csv`
- **Format:** Multi-column CSV with weather observations from OpenWeatherMap API
- **Original Shape:** 65,784 hourly observations across 22 features
- **Timestamp Format:** ISO 8601 string (`2015-01-01 00:00:00 +0000 UTC`)
- **Coverage:** January 1, 2015 - December 31, 2022

**Processing Steps:**
1. Parse `dt_iso` column to timezone-aware datetime (UTC)
2. Ensure timezone consistency with water consumption data
3. Set timestamp as DataFrame index
4. Remove redundant columns:
   - `dt_iso` (converted to index)
   - `city_name`, `lat`, `lon` (constant metadata)
   - `sea_level`, `grnd_level` (often missing or constant)

### Weather Features Retained

The following 22 meteorological features are preserved from the raw weather data:

| Feature | Description | Units | Relevance to Water Consumption |
|---------|-------------|-------|-------------------------------|
| `temp` | Air temperature | °C | Primary driver - warmer weather increases outdoor water use (irrigation, pools) |
| `feels_like` | Apparent temperature | °C | Accounts for humidity and wind chill effects on perceived temperature |
| `temp_min` | Minimum temperature in period | °C | Daily temperature range indicator |
| `temp_max` | Maximum temperature in period | °C | Daily temperature range indicator |
| `pressure` | Atmospheric pressure | hPa | Indirect indicator of weather systems |
| `humidity` | Relative humidity | % | Affects evaporation and irrigation needs |
| `dew_point` | Dew point temperature | °C | Moisture content indicator - affects comfort and irrigation decisions |
| `clouds_all` | Cloud coverage | % | Solar radiation proxy - affects heating/cooling and outdoor activities |
| `wind_speed` | Wind speed | m/s | Affects evaporation rates and perceived temperature |
| `wind_deg` | Wind direction | degrees | Directional weather pattern indicator |
| `wind_gust` | Wind gust speed | m/s | Extreme wind events |
| `rain_1h` | Rainfall in last hour | mm | **Critical** - reduces irrigation needs, may increase runoff |
| `rain_3h` | Rainfall in last 3 hours | mm | Lagged precipitation effect |
| `snow_1h` | Snowfall in last hour | mm | Winter precipitation - frozen water reduces consumption |
| `snow_3h` | Snowfall in last 3 hours | mm | Accumulated snow effect |
| `visibility` | Visibility distance | meters | Weather condition severity |
| `weather_id` | Weather condition code | categorical | OpenWeatherMap weather classification |
| `weather_main` | Weather category | categorical | High-level weather type (Clear, Rain, Snow, etc.) |
| `weather_description` | Detailed weather | categorical | Detailed weather description |
| `weather_icon` | Weather icon code | categorical | Visual weather representation |
| `dt` | Unix timestamp | seconds | Original timestamp in epoch format |

### Data Alignment Strategy

**Join Method:** Inner join on timestamp index
- Retains only timestamps present in both datasets
- Ensures complete feature vectors (no missing weather data)
- Result: 47,854 observations (all water timestamps have matching weather)

**Rationale:**
- Water consumption is the target - we keep all water timestamps
- Weather data has broader coverage (2015-2022 vs water's 2015-2020)
- Inner join preserves temporal integrity required for forecasting

### Missing Value Handling

**Strategy:** Forward Fill + Backward Fill
```python
df_aligned.fillna(method='ffill', inplace=True)  # Carry last valid observation forward
df_aligned.fillna(method='bfill', inplace=True)  # Fill remaining gaps backward
```

**Missing Value Statistics (Before Filling):**
- `water_consumption`: 5,136 missing (10.7%) - sensor gaps or transmission errors
- `visibility`: 47,854 missing (100%) - not reported for this location
- `wind_gust`: 34,326 missing (71.7%) - only reported during significant gusts
- `rain_1h`: 41,564 missing (86.9%) - zero rainfall represented as missing
- `rain_3h`: 45,070 missing (94.2%)
- `snow_1h`: 47,550 missing (99.4%) - minimal snow events in Denmark
- `snow_3h`: 47,648 missing (99.6%)

**After Filling:**
- `visibility`: Still 100% missing (will be dropped in feature engineering)
- All other features: Complete (0% missing)

**Output:** `centrum_water_weather_aligned.csv` (7.3 MB, 47,854 rows × 23 columns)

---

## Stage 2: Feature Engineering (`feature_engineering_centrum.py`)

### Purpose
Transform aligned raw data into a rich feature space incorporating domain knowledge about water consumption patterns, temporal dynamics, and meteorological interactions.

### Feature Selection & Cleaning

**Selected Weather Features (8):**
```python
['temp', 'dew_point', 'humidity', 'clouds_all', 'wind_speed', 'rain_1h', 'snow_1h', 'pressure']
```

**Selection Rationale:**
- **Temperature (`temp`)**: Primary consumption driver
- **Dew Point (`dew_point`)**: Better moisture indicator than humidity alone
- **Humidity (`humidity`)**: Evaporation and comfort effects
- **Cloud Coverage (`clouds_all`)**: Solar radiation proxy
- **Wind Speed (`wind_speed`)**: Evaporation rate modifier
- **Rainfall (`rain_1h`)**: Direct irrigation impact
- **Snowfall (`snow_1h`)**: Winter consumption patterns
- **Pressure (`pressure`)**: Weather system stability

**Excluded Features:**
- `feels_like`, `temp_min`, `temp_max`: Redundant with `temp`
- `wind_deg`, `wind_gust`: Low predictive value for water consumption
- `rain_3h`, `snow_3h`: Prefer hourly resolution (1h)
- `visibility`: 100% missing data
- Categorical weather codes: High cardinality, better captured by numerical features

### Feature Engineering Categories

#### 1. Temporal Features (10 features)

Understanding time-based patterns in water consumption:

**Direct Time Components:**
- `hour` (0-23): Hourly consumption patterns (morning peak, night lull)
- `day_of_week` (0-6): Monday=0, Sunday=6 - weekday vs weekend behavior
- `month` (1-12): Seasonal consumption trends
- `season` (0-3): Winter=0, Spring=1, Summer=2, Fall=3

**Cyclical Encoding:**
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
- Captures circular nature of time (23:00 is close to 00:00)
- Prevents model from treating hour 23 as "far" from hour 0
- Provides smooth gradients for learning

**Binary Indicators:**
- `is_weekend` (0/1): Saturday and Sunday flag
  - Rationale: Residential patterns shift, commercial usage drops

**Why These Features?**
- **Hourly patterns**: Morning showers, meal times, irrigation schedules
- **Weekly patterns**: Weekday work schedules vs weekend activities
- **Seasonal patterns**: Summer irrigation, winter heating-related usage
- **Cyclical encoding**: Mathematically sound temporal representation

#### 2. Autoregressive Lag Features (3 features)

Historical consumption patterns as predictors:

**Short-term Lag:**
- `water_lag_1h`: Previous hour's consumption
  - Captures immediate temporal dependency
  - Smooth transitions in consumption (inertia)

**Daily Lag:**
- `water_lag_24h`: Consumption exactly 24 hours ago
  - Captures daily periodicity
  - Same hour yesterday as a baseline

**Rolling Statistics:**
- `water_rolling_24h`: Mean consumption over last 24 hours
  - Smoothed daily baseline
  - Filters out hourly noise
  - Captures recent trend

**Implementation:**
```python
water_lag_1h = water_consumption.shift(1)       # t-1
water_lag_24h = water_consumption.shift(24)     # t-24
water_rolling_24h = water_consumption.rolling(window=24).mean()  # avg(t-24:t-1)
```

**Missing Values:**
- First 24 rows dropped (no valid lags)
- Final dataset: 47,830 rows (47,854 - 24)

#### 3. Weather Interaction Features (3 features)

Non-linear relationships and interaction effects:

**Polynomial Features:**
- `temp_squared = temp²`
  - Captures non-linear temperature-consumption relationship
  - Quadratic response: extreme cold/heat may have disproportionate effects

**Temperature Interactions:**
- `temp_wind_interaction = temp × wind_speed`
  - Wind chill effect: cold + wind feels colder
  - Evaporation effect: heat + wind increases water loss

**Behavioral Interactions:**
- `temp_weekend_interaction = temp × is_weekend`
  - Weekend irrigation behavior differs from weekday
  - Temperature effects stronger on weekends (more home activity)

**Why Interactions?**
- Real-world phenomena are rarely additive
- Temperature impact varies with wind conditions
- Human behavior (weekend) modulates weather effects

#### 4. Holiday Features (4 features)

Special day effects on consumption patterns:

**Public Holidays (Denmark):**
- `is_public_holiday` (0/1): Binary indicator
- `public_holiday_name` (categorical): Holiday name or "None"
  - Uses `holidays` Python library for Danish public holidays
  - Covers: New Year, Easter, Christmas, Constitution Day, etc.

**School Holidays:**
- `is_school_holiday` (0/1): Binary indicator  
- `school_holiday_name` (categorical): Holiday period or "None"
  - Source: `../school_holidays.csv` (if available)
  - Covers summer, winter, autumn breaks

**Impact on Consumption:**
- **Public holidays**: Reduced commercial activity, increased residential
- **School holidays**: Families at home, travel patterns, irrigation
- **Combined effect**: Summer school holidays + warm weather → peak irrigation

**Note:** Current pipeline shows school holidays file not found - set to 0 (can be added later)

### Feature Summary

**Final Feature Set (26 features):**

| Category | Features | Count |
|----------|----------|-------|
| **Target** | water_consumption | 1 |
| **Raw Weather** | temp, dew_point, humidity, clouds_all, wind_speed, rain_1h, snow_1h, pressure | 8 |
| **Temporal** | hour, day_of_week, month, hour_sin, hour_cos, is_weekend, season | 7 |
| **Autoregressive** | water_lag_1h, water_lag_24h, water_rolling_24h | 3 |
| **Interactions** | temp_squared, temp_wind_interaction, temp_weekend_interaction | 3 |
| **Holidays** | is_public_holiday, is_school_holiday, public_holiday_name, school_holiday_name | 4 |
| **Total** | | **26** |

### Output Characteristics

**File:** `centrum_features_engineered.csv`
- **Size:** 11 MB
- **Rows:** 47,830 hourly observations
- **Columns:** 27 (26 features + timestamp index)
- **Time Range:** June 1, 2015 - September 30, 2020 (approximately 5.3 years)
- **Frequency:** Hourly (1H)
- **Completeness:** 100% (all missing values handled)

---

## Running the Pipeline

### Full Pipeline (Both Stages)

```bash
cd /home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis/centrum_processing
module load python/3.12-conda
python build_features_centrum.py
```

### Individual Stages

**Stage 1 Only (Re-align data):**
```python
from align_data_centrum import align_data_centrum
align_data_centrum()
```

**Stage 2 Only (Re-engineer features):**
```python
from feature_engineering_centrum import engineer_features_centrum
engineer_features_centrum()
```

**Selective Execution:**
```python
from build_features_centrum import build_centrum_features

# Skip alignment if aligned file is current
build_centrum_features(run_align=False, run_feature_engineering=True)
```

---

## Dependencies

**Python Packages:**
- `pandas>=2.0.0`: Data manipulation and time series operations
- `numpy>=1.24.0`: Numerical computations and array operations
- `holidays>=0.85`: Danish public holiday calendar

**Install:**
```bash
module load python/3.12-conda
pip install pandas numpy holidays
```

---

## Data Quality Notes

### Known Issues

1. **Visibility Feature**: 100% missing - excluded from modeling
2. **Precipitation Data**: ~90% missing values (interpreted as zero rainfall)
3. **Water Consumption Gaps**: 5,136 missing observations (10.7%)
   - Handled via forward/backward fill
   - May introduce slight temporal smoothing

### Assumptions

1. **Missing rainfall = No rainfall**: Weather API reports precipitation only when it occurs
2. **Temporal continuity**: Forward fill assumes consumption changes gradually
3. **Timezone consistency**: All data aligned to UTC
4. **Hourly resolution**: Sub-hourly dynamics not captured

### Data Split Recommendations

For time series modeling:
```python
# Chronological split (no data leakage)
train: 2015-06-01 to 2019-12-31  (~80%)
val:   2020-01-01 to 2020-06-30  (~10%)
test:  2020-07-01 to 2020-09-30  (~10%)
```

---

## Comparison with Nordbyen (Heat) Pipeline

| Aspect | Centrum (Water) | Nordbyen (Heat) |
|--------|----------------|-----------------|
| **Target Variable** | water_consumption | heat_consumption |
| **Domain** | Water distribution network | District heating network |
| **Data Source** | water_a_centrum/ | dma_a_nordbyen_heat/ |
| **Weather Relevance** | Rain, temp, humidity | Temperature, wind chill |
| **Seasonal Pattern** | Summer peak (irrigation) | Winter peak (heating) |
| **Feature Count** | 26 features | 26 features |
| **Pipeline Structure** | Identical | Identical |

**Key Difference**: Temperature effect is inverted
- **Water**: Higher temp → Higher consumption (irrigation)
- **Heat**: Higher temp → Lower consumption (heating needs)

---

## Next Steps

1. **Exploratory Data Analysis**:
   - Visualize consumption patterns by hour/day/season
   - Analyze temperature-consumption relationship
   - Identify anomalies and outliers

2. **Feature Importance Analysis**:
   - Train baseline models (Random Forest, XGBoost)
   - Evaluate feature importance scores
   - Consider feature selection/pruning

3. **Model Development**:
   - Adapt existing time series models (N-HiTS, TFT, TimesNet)
   - Configure for water consumption forecasting
   - Benchmark against Nordbyen performance

4. **Additional Features** (Future Work):
   - School holiday integration (add school_holidays.csv)
   - Population/demographic factors
   - Historical maintenance events
   - Economic indicators (water pricing changes)

---

## Contact & Support

For questions about this pipeline or water consumption forecasting:
- Dataset: Water A - Centrum (Brønderslev, Denmark)
- Temporal Coverage: 2015-2020
- Pipeline Version: 1.0 (December 2025)
- Related Project: ExAI-Timeseries-Thesis

---

**Last Updated:** December 22, 2025
