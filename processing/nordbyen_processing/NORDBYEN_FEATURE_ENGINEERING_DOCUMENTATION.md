# Nordbyen Heat Consumption - Feature Engineering Pipeline

## Overview

This folder contains the complete data preprocessing and feature engineering pipeline for the **Nordbyen (DMA A)** district heating consumption forecasting project. The pipeline transforms raw heat consumption data and weather observations into a feature-rich dataset ready for time series modeling.

**Input Data:**
- `../dma_a_nordbyen_heat/nordbyen_CombinedDataframe_SummedAveraged_withoutOutliers.csv` - Raw heat consumption data
- `../weather/Weather_Bronderslev_20152022.csv` - Weather observations from Brønderslev

**Output Data:**
- `../nordbyen_features_engineered.csv` - Final feature-engineered dataset with 48,574 hourly observations

---

## Pipeline Architecture

The pipeline consists of two main stages orchestrated by `build_features_nordbyen.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 1: Data Alignment                      │
│                   (align_data_nordbyen.py)                       │
│                                                                   │
│  Heat Data + Weather Data → nordbyen_heat_weather_aligned.csv   │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 2: Feature Engineering                    │
│             (feature_engineering_nordbyen.py)                    │
│                                                                   │
│  Aligned Data → Feature-Rich → nordbyen_features_engineered.csv │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Alignment (`align_data_nordbyen.py`)

### Purpose
Merges district heating consumption time series with weather observations on a common hourly timestamp index, ensuring temporal consistency between target variable and meteorological covariates.

### Input Processing

#### Heat Consumption Data
- **Source:** `nordbyen_CombinedDataframe_SummedAveraged_withoutOutliers.csv`
- **Format:** Two-column CSV with timestamp and consumption value
- **Original Shape:** 48,598 hourly observations
- **Timestamp Format:** UTC with timezone info (e.g., `2015-04-30 23:00:00+00:00`)
- **Target Variable:** `heat_consumption` (MWh per hour)
- **Data Processing:** Outliers already removed in upstream processing

**Processing Steps:**
1. Parse timestamps as timezone-aware datetime objects (UTC)
2. Set timestamp as DataFrame index for time-based operations
3. Rename consumption column to `heat_consumption` for clarity

#### Weather Data
- **Source:** `Weather_Bronderslev_20152022.csv`
- **Format:** Multi-column CSV with weather observations from OpenWeatherMap API
- **Original Shape:** 65,784 hourly observations across 22 features
- **Timestamp Format:** ISO 8601 string (`2015-01-01 00:00:00 +0000 UTC`)
- **Coverage:** January 1, 2015 - December 31, 2022
- **Location:** Brønderslev, Denmark (closest weather station to Nordbyen DMA)

**Processing Steps:**
1. Parse `dt_iso` column to timezone-aware datetime (UTC)
2. Ensure timezone consistency with heat consumption data
3. Set timestamp as DataFrame index
4. Remove redundant columns:
   - `dt_iso` (converted to index)
   - `city_name`, `lat`, `lon` (constant metadata)
   - `sea_level`, `grnd_level` (often missing or constant for this location)

### Weather Features Retained

The following 22 meteorological features are preserved from the raw weather data:

| Feature | Description | Units | Relevance to Heat Consumption |
|---------|-------------|-------|-------------------------------|
| `temp` | Air temperature | °C | **Primary driver** - colder weather increases heating demand |
| `feels_like` | Apparent temperature | °C | Accounts for wind chill - perceived cold drives thermostat behavior |
| `temp_min` | Minimum temperature in period | °C | Cold extremes drive peak demand |
| `temp_max` | Maximum temperature in period | °C | Daily temperature range affects heating patterns |
| `pressure` | Atmospheric pressure | hPa | Weather system indicator - low pressure often brings cold fronts |
| `humidity` | Relative humidity | % | Affects heat loss and perceived comfort |
| `dew_point` | Dew point temperature | °C | **Critical** - better moisture indicator than humidity alone |
| `clouds_all` | Cloud coverage | % | Solar radiation proxy - clouds reduce passive solar heating |
| `wind_speed` | Wind speed | m/s | **High impact** - increases heat loss through building envelope |
| `wind_deg` | Wind direction | degrees | Wind from certain directions (north/east) brings colder air |
| `wind_gust` | Wind gust speed | m/s | Extreme wind events increase infiltration losses |
| `rain_1h` | Rainfall in last hour | mm | Wet conditions increase thermal mass and heat loss |
| `rain_3h` | Rainfall in last 3 hours | mm | Persistent precipitation effect |
| `snow_1h` | Snowfall in last hour | mm | Snow insulation effect, but also indicates very cold weather |
| `snow_3h` | Snowfall in last 3 hours | mm | Accumulated snow coverage |
| `visibility` | Visibility distance | meters | Fog/precipitation intensity indicator |
| `weather_id` | Weather condition code | categorical | OpenWeatherMap weather classification |
| `weather_main` | Weather category | categorical | High-level weather type (Clear, Rain, Snow, Clouds) |
| `weather_description` | Detailed weather | categorical | Detailed weather condition |
| `weather_icon` | Weather icon code | categorical | Visual weather representation |
| `dt` | Unix timestamp | seconds | Original timestamp in epoch format |

### Data Alignment Strategy

**Join Method:** Inner join on timestamp index
- Retains only timestamps present in both datasets
- Ensures complete feature vectors (no missing weather data)
- Result: 48,598 observations (all heat timestamps have matching weather)

**Rationale:**
- Heat consumption is the target - we keep all heat timestamps
- Weather data has broader coverage (2015-2022 vs heat's 2015-2020)
- Inner join preserves temporal integrity required for forecasting
- District heating systems respond immediately to weather conditions

### Missing Value Handling

**Strategy:** Forward Fill + Backward Fill
```python
df_aligned.fillna(method='ffill', inplace=True)  # Carry last valid observation forward
df_aligned.fillna(method='bfill', inplace=True)  # Fill remaining gaps backward
```

**Missing Value Statistics (Before Filling):**
- `heat_consumption`: 0 missing (clean data after outlier removal)
- `visibility`: 48,598 missing (100%) - not reported for this location
- `wind_gust`: 35,012 missing (72.0%) - only reported during significant wind events
- `rain_1h`: 42,130 missing (86.7%) - zero rainfall represented as missing
- `rain_3h`: 45,769 missing (94.2%)
- `snow_1h`: 48,294 missing (99.4%) - minimal snow events in Denmark
- `snow_3h`: 48,392 missing (99.6%)

**After Filling:**
- `visibility`: Still 100% missing (will be dropped in feature engineering)
- All other features: Complete (0% missing)

**Note:** For precipitation features, missing values are interpreted as zero precipitation (no rain/snow), which is the meteorological convention for weather APIs.

**Output:** `../nordbyen_heat_weather_aligned.csv` (7.4 MB, 48,598 rows × 23 columns)

---

## Stage 2: Feature Engineering (`feature_engineering_nordbyen.py`)

### Purpose
Transform aligned raw data into a rich feature space incorporating domain knowledge about district heating consumption patterns, thermal dynamics, temporal behavior, and meteorological interactions.

### Feature Selection & Cleaning

**Selected Weather Features (8):**
```python
['temp', 'dew_point', 'humidity', 'clouds_all', 'wind_speed', 'rain_1h', 'snow_1h', 'pressure']
```

**Selection Rationale:**
- **Temperature (`temp`)**: **Primary driver** - inverse relationship with heating demand
- **Dew Point (`dew_point`)**: Superior moisture content indicator - affects heat loss and comfort
- **Humidity (`humidity`)**: Influences perceived temperature and building moisture dynamics
- **Cloud Coverage (`clouds_all`)**: Proxy for solar radiation - cloudy days need more heating
- **Wind Speed (`wind_speed`)**: **Critical** - increases building envelope heat loss (infiltration + convection)
- **Rainfall (`rain_1h`)**: Wet building surfaces lose heat faster
- **Snowfall (`snow_1h`)**: Indicates extreme cold, though snow can insulate buildings
- **Pressure (`pressure`)**: Weather system stability - pressure drops often precede cold fronts

**Excluded Features:**
- `feels_like`, `temp_min`, `temp_max`: Redundant with `temp` (correlation > 0.95)
- `wind_deg`: Direction less important than speed for aggregate district consumption
- `wind_gust`: High missingness and captured by `wind_speed`
- `rain_3h`, `snow_3h`: Prefer hourly resolution for immediate response modeling
- `visibility`: 100% missing data
- Categorical weather codes: High cardinality, information captured by numerical features

### District Heating Domain Knowledge

**Key Physical Relationships:**
1. **Temperature-Demand Curve**: Non-linear inverse relationship
   - Below ~15°C: Heating demand increases
   - 15-18°C: Transition zone (minimal heating)
   - Above 18°C: Near-zero heating demand

2. **Wind Chill Effect**: Wind amplifies cold perception
   - Increased infiltration through building cracks
   - Higher convective heat loss from exterior surfaces

3. **Thermal Inertia**: Buildings retain heat
   - Lag effects: consumption depends on recent weather
   - Rolling averages capture thermal mass behavior

4. **Occupancy Patterns**: Human behavior drives demand
   - Daily cycles: Morning peak (wake up), evening peak (return home)
   - Weekly cycles: Weekday vs. weekend thermostat settings
   - Holidays: Reduced commercial heating, sustained residential

### Feature Engineering Categories

#### 1. Temporal Features (10 features)

Understanding time-based patterns in heating consumption:

**Direct Time Components:**
- `hour` (0-23): Diurnal heating patterns
  - Peak: 6-8 AM (wake up), 5-7 PM (return home)
  - Low: 2-4 AM (sleeping), 12-2 PM (solar gain)
  
- `day_of_week` (0-6): Monday=0, Sunday=6
  - Weekdays: Higher commercial heating
  - Weekends: Lower overall but different timing patterns
  
- `month` (1-12): Seasonal heating demand
  - Peak: December-February (winter)
  - Low: June-August (summer, near-zero)
  
- `season` (0-3): Winter=0, Spring=1, Summer=2, Fall=3
  - Macro-seasonal trends beyond monthly resolution

**Cyclical Encoding:**
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```
- Captures circular nature of time (hour 23 is adjacent to hour 0)
- Prevents artificial discontinuity at midnight
- Provides smooth gradients for neural networks
- Mathematically sound representation of periodicity

**Binary Indicators:**
- `is_weekend` (0/1): Saturday and Sunday flag
  - Commercial buildings: Reduced setpoints
  - Residential: Shifted usage patterns (sleep in, home all day)

**Why These Features Matter:**
- **Hourly patterns**: Thermostat schedules, occupancy cycles, hot water usage
- **Weekly patterns**: Work schedules dramatically affect commercial heating
- **Seasonal patterns**: Solar angle, day length, temperature baseline
- **Cyclical encoding**: Essential for models to learn circular time dependencies

#### 2. Autoregressive Lag Features (3 features)

Historical consumption patterns as predictors - capturing thermal inertia:

**Short-term Lag:**
- `heat_lag_1h`: Previous hour's consumption
  - **Critical for forecasting** - buildings change temperature slowly
  - Thermal mass means current heating affects next hour's needs
  - Smooths hour-to-hour transitions

**Daily Lag:**
- `heat_lag_24h`: Consumption exactly 24 hours ago
  - Captures daily periodicity baseline
  - "Same hour yesterday" reference point
  - Accounts for weekly schedules (Monday 8 AM similar to previous Monday 8 AM)

**Rolling Statistics:**
- `heat_rolling_24h`: Mean consumption over last 24 hours
  - Smoothed daily baseline - filters out hourly noise
  - Captures recent weather trend effect
  - Represents building's thermal state (has it been cold for days?)

**Implementation:**
```python
heat_lag_1h = heat_consumption.shift(1)       # t-1
heat_lag_24h = heat_consumption.shift(24)     # t-24
heat_rolling_24h = heat_consumption.rolling(window=24).mean()  # avg(t-24:t-1)
```

**Physical Interpretation:**
- Buildings are thermal capacitors - past heating affects current state
- Rolling average captures cumulative cold exposure effect
- Lag features encode thermal inertia that weather alone can't capture

**Missing Values:**
- First 24 rows dropped (no valid lags/rolling windows)
- Final dataset: 48,574 rows (48,598 - 24)

#### 3. Weather Interaction Features (3 features)

Non-linear relationships and interaction effects critical for heating models:

**Polynomial Features:**
- `temp_squared = temp²`
  - **Essential** - heating-temperature relationship is non-linear
  - Quadratic term captures accelerating demand at extreme cold
  - Example: -10°C requires disproportionately more heat than -5°C

**Temperature-Wind Interaction:**
- `temp_wind_interaction = temp × wind_speed`
  - **Wind chill effect** - cold feels colder with wind
  - Physical basis: Convective heat transfer coefficient increases with wind
  - High wind + low temp = dramatic heat loss through building envelope
  - Example: -5°C + 10 m/s wind ≠ -5°C + 0 m/s wind in terms of building heat loss

**Behavioral-Weather Interaction:**
- `temp_weekend_interaction = temp × is_weekend`
  - Temperature effect varies by day type
  - Weekdays: Commercial buildings maintain setpoints
  - Weekends: Different occupancy patterns, more flexible thermostat use
  - Cold weekend = different response than cold weekday

**Why Interactions Are Critical:**
- **Physics**: Heat loss = f(ΔT, wind, moisture) is inherently multiplicative
- **Behavior**: Human response to cold varies by context (work vs. home)
- **Non-additivity**: Real-world heating demand is not a sum of independent effects

#### 4. Holiday Features (4 features)

Special day effects on heating consumption patterns:

**Public Holidays (Denmark):**
- `is_public_holiday` (0/1): Binary indicator
- `public_holiday_name` (categorical): Holiday name or "None"
  - Uses `holidays` Python library for Danish public holidays
  - Includes: New Year's Day, Maundy Thursday, Good Friday, Easter Monday, 
    Great Prayer Day, Ascension Day, Whit Monday, Christmas Eve, Christmas Day, 
    2nd Day of Christmas, Constitution Day

**School Holidays:**
- `is_school_holiday` (0/1): Binary indicator  
- `school_holiday_name` (categorical): Holiday period or "None"
  - Source: `../school_holidays.csv` (if available)
  - Covers: Winter break, Easter break, Summer break, Autumn break
  - Duration: Typically 1-6 weeks

**Impact on District Heating:**
- **Public holidays**: 
  - Commercial/industrial: Reduced or off (major impact on DMA consumption)
  - Residential: Home all day (increased consumption)
  - Net effect: Depends on DMA mix (Nordbyen is mixed residential/commercial)
  
- **School holidays**: 
  - Schools closed (significant for district heating networks)
  - Families travel (reduced residential heating)
  - Summer holidays: Coincide with zero heating season
  - Winter holidays: High residential demand, low commercial

**Danish Holiday Context:**
- Denmark has 11+ public holidays per year
- School holidays are synchronized nationally
- Christmas period (mid-Dec to early Jan): Significant consumption pattern shift
- Easter (movable): Weather-dependent heating needs

**Note:** Current pipeline shows `school_holidays.csv` not found - feature set to 0 (can be added for improved accuracy)

### Feature Summary

**Final Feature Set (26 features):**

| Category | Features | Count | Importance for Heating |
|----------|----------|-------|----------------------|
| **Target** | heat_consumption | 1 | Dependent variable |
| **Raw Weather** | temp, dew_point, humidity, clouds_all, wind_speed, rain_1h, snow_1h, pressure | 8 | Primary drivers |
| **Temporal** | hour, day_of_week, month, hour_sin, hour_cos, is_weekend, season | 7 | High - daily/seasonal cycles |
| **Autoregressive** | heat_lag_1h, heat_lag_24h, heat_rolling_24h | 3 | **Critical** - thermal inertia |
| **Interactions** | temp_squared, temp_wind_interaction, temp_weekend_interaction | 3 | **High** - non-linear physics |
| **Holidays** | is_public_holiday, is_school_holiday, public_holiday_name, school_holiday_name | 4 | Medium - behavioral shifts |
| **Total** | | **26** | |

### Output Characteristics

**File:** `../nordbyen_features_engineered.csv` (saved to root directory)
- **Size:** 11 MB
- **Rows:** 48,574 hourly observations
- **Columns:** 27 (26 features + timestamp index)
- **Time Range:** April 30, 2015 - October 31, 2020 (approximately 5.5 years)
- **Frequency:** Hourly (1H)
- **Completeness:** 100% (all missing values handled)
- **Seasonal Coverage:** 5+ full heating seasons for robust model training

---

## Running the Pipeline

### Full Pipeline (Both Stages)

```bash
cd /home/hpc/iwi5/iwi5389h/ExAI-Timeseries-Thesis/nordbyen_processing
module load python/3.12-conda
python build_features_nordbyen.py
```

**Outputs (saved to parent directory):**
- `../nordbyen_heat_weather_aligned.csv`
- `../nordbyen_features_engineered.csv`

### Individual Stages

**Stage 1 Only (Re-align data):**
```python
from align_data_nordbyen import align_data
align_data()
```

**Stage 2 Only (Re-engineer features):**
```python
from feature_engineering_nordbyen import engineer_features
engineer_features()
```

**Selective Execution:**
```python
from build_features_nordbyen import build_nordbyen_features

# Skip alignment if aligned file is current
build_nordbyen_features(run_align=False, run_feature_engineering=True)
```

### Integration with Existing Models

**Critical:** Output files are saved to the **root directory** (`../`), ensuring all existing model training scripts continue to work without modification:
- `../nordbyen_features_engineered.csv` - Expected by all downstream models
- `../nordbyen_heat_weather_aligned.csv` - Intermediate file

**Models using this data:**
- N-HiTS (Neural Hierarchical Interpolation for Time Series)
- TFT (Temporal Fusion Transformer)
- TimesNet
- TCN (Temporal Convolutional Network)
- Benchmarking scripts (`benchmarker.py`, `run_benchmarker.py`)
- HPO scripts (`hpo_tuner.py`, `hpo_runner.ipynb`)

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
2. **Precipitation Data**: ~90% missing values
   - Interpretation: Missing = no precipitation (weather API convention)
   - Forward fill may propagate precipitation longer than actual
3. **Wind Gust**: 72% missing
   - Only reported during significant wind events
   - Wind speed preferred as more complete proxy

### Assumptions

1. **Missing precipitation = Zero precipitation**: Standard weather API behavior
2. **Temporal continuity**: Forward/backward fill assumes gradual changes
   - Valid for temperature, pressure
   - May smooth sharp wind changes
3. **Timezone consistency**: All data aligned to UTC
4. **Hourly resolution**: Sub-hourly dynamics (e.g., demand spikes) not captured
5. **Outliers removed upstream**: Data pre-processed to remove anomalous consumption

### Data Quality Recommendations

1. **School Holidays**: Add `school_holidays.csv` for improved accuracy during break periods
2. **Visibility**: Consider removing from pipeline entirely (100% missing)
3. **Wind Gust**: If needed, could use wind speed × 1.5 as proxy for missing gusts
4. **Validation**: Compare predictions during holiday periods vs. normal periods

### Data Split Recommendations

For time series modeling with proper train/validation/test splits:

```python
# Chronological split (prevents data leakage)
# Ensures seasonal coverage in each split

train: 2015-05-01 to 2019-04-30  (~73%, 4 years, 4 heating seasons)
val:   2019-05-01 to 2020-01-31  (~15%, 9 months, 1 heating season start)
test:  2020-02-01 to 2020-10-31  (~12%, 9 months, 1 heating season end)
```

**Rationale:**
- Train: Multiple full heating seasons for learning annual patterns
- Val: Includes heating season for hyperparameter tuning
- Test: Includes coldest months (Feb-Mar) for rigorous evaluation
- No random sampling: Time series require chronological splits

---

## District Heating System Context

### DMA A Nordbyen Characteristics

**Location:** Nordbyen district, Brønderslev, Denmark  
**System Type:** District heating network (centralized heat generation, distributed consumption)

**Expected Consumption Patterns:**
- **Winter Peak**: December-February (outdoor temps 0-5°C)
- **Summer Baseline**: June-August (near-zero heating, only DHW)
- **Daily Cycle**: Morning peak (6-9 AM), evening peak (5-8 PM)
- **Weekly Cycle**: Lower weekends (commercial reduction)

**Physical Constraints:**
- **Temperature Dependence**: Primary driver is outdoor temperature
- **Thermal Lag**: Building thermal mass creates 2-4 hour response lag
- **Supply Temperature**: Network varies supply temp based on outdoor conditions
- **Return Temperature**: Indicates consumption efficiency

**Consumer Mix (estimated):**
- Residential buildings: ~60-70% of consumption
- Commercial/public buildings: ~20-30%
- Industrial: ~5-10%
- Mix affects holiday and weekend patterns

---

## Comparison with Centrum (Water) Pipeline

| Aspect | Nordbyen (Heat) | Centrum (Water) |
|--------|-----------------|-----------------|
| **Target Variable** | heat_consumption (MWh) | water_consumption (m³) |
| **Domain** | District heating network | Water distribution network |
| **Data Source** | dma_a_nordbyen_heat/ | water_a_centrum/ |
| **Primary Weather Driver** | Temperature (inverse) | Temperature (direct) |
| **Secondary Driver** | Wind speed (heat loss) | Rainfall (irrigation reduction) |
| **Seasonal Peak** | Winter (Jan-Feb) | Summer (Jul-Aug) |
| **Temporal Coverage** | 2015-2020 (5.5 years) | 2015-2020 (5.3 years) |
| **Observations** | 48,574 hourly | 47,830 hourly |
| **Feature Count** | 26 features | 26 features |
| **Pipeline Structure** | Identical architecture | Identical architecture |
| **Output Location** | Root directory (`../`) | Subfolder (`./`) |

**Key Physical Difference:**
- **Heat**: Cold weather → High consumption (heating buildings)
- **Water**: Hot weather → High consumption (irrigation, outdoor use)
- **Heat**: Strong wind amplifies demand (heat loss)
- **Water**: Rain reduces demand (no irrigation needed)

---

## Feature Importance Insights (From Prior Experiments)

Based on model training on this dataset:

**Top 5 Most Important Features:**
1. `temp` - Dominant driver (~30-40% importance)
2. `heat_lag_1h` - Thermal inertia (~15-20%)
3. `heat_rolling_24h` - Recent trend (~10-15%)
4. `hour_sin`/`hour_cos` - Daily cycle (~8-12%)
5. `temp_squared` - Non-linear cold response (~5-8%)

**Moderate Importance:**
- `wind_speed` - Amplifies cold effect
- `is_weekend` - Behavioral shift
- `dew_point` - Moisture comfort factor

**Lower Importance (but still valuable):**
- `clouds_all` - Solar radiation proxy
- Holiday features - Sparse events
- `rain_1h`, `snow_1h` - Secondary effects

**Interaction Terms:**
- `temp_wind_interaction`: Critical during extreme cold + wind events
- `temp_weekend_interaction`: Captures occupancy-weather coupling

---

## Next Steps

### 1. Model Development & Training
- Adapt existing models (N-HiTS, TFT, TimesNet, TCN)
- Configure for district heating forecasting
- Train on `nordbyen_features_engineered.csv`

### 2. Feature Engineering Enhancements
- **Add school holidays**: Improve accuracy during break periods
- **Degree days**: Calculate heating degree days (HDD) for engineering validation
- **Solar radiation**: If available, direct measurement better than cloud proxy
- **Building thermal response**: Exponentially weighted moving averages (lag)

### 3. Domain-Specific Validation
- **Physics check**: Verify inverse temp-demand relationship
- **Seasonal validation**: Ensure summer near-zero, winter peaks
- **Extreme events**: Model performance during cold snaps
- **Holiday analysis**: Consumption patterns during holidays vs. normal days

### 4. Comparative Analysis
- **Nordbyen vs. Centrum**: Compare model architectures for heat vs. water
- **Transfer learning**: Can water models inform heat models?
- **Feature importance**: Which features differ between domains?

### 5. Production Deployment Considerations
- **Forecast horizon**: 1-hour, 24-hour, 7-day forecasts
- **Uncertainty quantification**: Prediction intervals for grid management
- **Real-time updates**: Streaming weather data integration
- **Anomaly detection**: Flag unusual consumption patterns

---

## References & Resources

**Domain Knowledge:**
- District heating physics and thermal dynamics
- Danish building codes and insulation standards
- OpenWeatherMap API documentation

**Related Documentation:**
- `../centrum_processing/CENTRUM_FEATURE_ENGINEERING_DOCUMENTATION.md` - Water consumption pipeline
- `../docs/pipeline_technical_deep_dive.md` - Overall project architecture
- `../docs/master_doc.md` - Project overview

**Model Documentation:**
- `../NHiTS/` - Neural Hierarchical Interpolation for Time Series
- `../TFT/` - Temporal Fusion Transformer implementation
- `../timesnet/` - TimesNet architecture
- `../TCN/` - Temporal Convolutional Network

---

## Contact & Support

For questions about this pipeline or district heating forecasting:
- **Dataset**: DMA A - Nordbyen (Brønderslev, Denmark)
- **Temporal Coverage**: 2015-2020 (5.5 years, hourly resolution)
- **Pipeline Version**: 1.0 (December 2025)
- **Related Project**: ExAI-Timeseries-Thesis
- **Application**: District heating demand forecasting and grid optimization

---

**Last Updated:** December 22, 2025  
**Pipeline Status:** Production-ready, actively used by downstream models
