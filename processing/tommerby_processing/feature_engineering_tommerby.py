import pandas as pd
import numpy as np
import os

# Define file paths
DATA_DIR = "."
INPUT_FILE = "tommerby_water_weather_aligned.csv"
OUTPUT_FILE = "tommerby_features_engineered.csv"
OUTPUT_FILE_FROM = "tommerby_features_engineered_from_2018-04-01.csv"
START_DATE = "2018-04-01"

def engineer_features_tommerby():
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Original Shape: {df.shape}")

    # 1. Feature Selection
    print("Selecting Features...")
    # Keep relevant weather columns
    weather_cols = ['temp', 'dew_point', 'humidity', 'clouds_all', 'wind_speed', 'rain_1h', 'snow_1h', 'pressure']
    # Ensure columns exist before selecting
    weather_cols = [c for c in weather_cols if c in df.columns]
    
    # Target variable
    target_col = ['water_consumption']
    
    # Select only these columns
    df = df[target_col + weather_cols].copy()

    # 2. Time Feature Engineering
    print("Engineering Time Features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Season (Approximation: 12,1,2=Winter, 3,4,5=Spring, 6,7,8=Summer, 9,10,11=Fall)
    # Mapping: Winter=0, Spring=1, Summer=2, Fall=3
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3 - 1) 

    # 3. Demand History (Lags & Rolling)
    print("Engineering Lag & Rolling Features...")
    df['water_lag_1h'] = df['water_consumption'].shift(1)
    df['water_lag_24h'] = df['water_consumption'].shift(24)
    df['water_rolling_24h'] = df['water_consumption'].rolling(window=24).mean()
    
    # 4. Interaction & Polynomial Features
    print("Engineering Interactions...")
    # Non-linear temperature effect
    df['temp_squared'] = df['temp'] ** 2
    
    # Wind chill proxy (interaction)
    df['temp_wind_interaction'] = df['temp'] * df['wind_speed']
    
    # Weekend interaction (occupancy effect on water usage curve)
    df['temp_weekend_interaction'] = df['temp'] * df['is_weekend']

    # 5. Holiday Features
    print("Engineering Holiday Features...")
    import holidays
    
    # Public Holidays (DK)
    years = df.index.year.unique().tolist()
    dk_holidays = holidays.DK(years=years)
    
    df['date'] = df.index.date
    df['public_holiday_name'] = df['date'].apply(lambda x: dk_holidays.get(x))
    df['is_public_holiday'] = df['public_holiday_name'].notna().astype(int)
    df['public_holiday_name'] = df['public_holiday_name'].fillna("None")
    
    # School Holidays
    school_holidays_file = os.path.join("..", "school_holidays.csv")
    if os.path.exists(school_holidays_file):
        school_hol = pd.read_csv(school_holidays_file, parse_dates=['start_date', 'end_date'])
        school_holiday_map = {}
        for _, row in school_hol.iterrows():
            date_range = pd.date_range(start=row['start_date'], end=row['end_date'])
            for d in date_range:
                school_holiday_map[d.date()] = row.get('description', None)
        
        df['school_holiday_name'] = df['date'].map(school_holiday_map)
        df['is_school_holiday'] = df['school_holiday_name'].notna().astype(int)
        df['school_holiday_name'] = df['school_holiday_name'].fillna("None")
    else:
        # Fallback: try to extract school-holiday info from Centrum engineered features (if available)
        centrum_file = os.path.join("..", "centrum_processing", "centrum_features_engineered_from_2018-04-01.csv")
        if os.path.exists(centrum_file):
            print(f"school_holidays.csv not found — extracting school-holiday dates from {centrum_file}")
            cf = pd.read_csv(centrum_file, parse_dates=['timestamp'])
            if 'is_school_holiday' in cf.columns and 'school_holiday_name' in cf.columns:
                cf['date'] = cf['timestamp'].dt.date
                # For each date where any hour is marked as school holiday, take the most common name
                grp = cf[cf['is_school_holiday'] == 1].groupby('date')['school_holiday_name']
                school_holiday_map = {}
                for d, s in grp:
                    try:
                        name = s.mode().iloc[0]
                    except Exception:
                        name = s.iloc[0]
                    school_holiday_map[d] = name

                df['school_holiday_name'] = df['date'].map(school_holiday_map)
                df['is_school_holiday'] = df['school_holiday_name'].notna().astype(int)
                df['school_holiday_name'] = df['school_holiday_name'].fillna("None")
            else:
                print("Centrum engineered file missing school holiday columns — setting school holidays to 0.")
                df['is_school_holiday'] = 0
                df['school_holiday_name'] = "None"
        else:
            print("Warning: school_holidays.csv and fallback Centrum file not found. Setting school holidays to 0.")
            df['is_school_holiday'] = 0
            df['school_holiday_name'] = "None"
        
    # Drop temporary date column
    df.drop(columns=['date'], inplace=True)

    # Drop intermediate columns if not needed (keeping hour/month/day_of_week for now as they are useful for trees)
    # Dropping NaNs created by lags
    print("Handling Missing Values...")
    original_len = len(df)
    df.dropna(inplace=True)
    dropped = original_len - len(df)
    print(f"Dropped {dropped} rows due to NaN values (from lag features).")
    
    # Save full engineered file
    print(f"Final Shape: {df.shape}")
    print(df.head())

    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE)

    # Create and save a filtered dataset starting from START_DATE for benchmarking consistency
    try:
        start_ts = pd.to_datetime(START_DATE, utc=True)
    except Exception:
        start_ts = pd.to_datetime(START_DATE)

    df_from = df[df.index >= start_ts]
    print(f"Saving filtered dataset from {START_DATE} to {OUTPUT_FILE_FROM} (rows: {len(df_from)})")
    df_from.to_csv(OUTPUT_FILE_FROM)

    print("Done.")

if __name__ == "__main__":
    engineer_features_tommerby()
