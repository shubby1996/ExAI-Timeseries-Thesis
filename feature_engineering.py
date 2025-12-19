import pandas as pd
import numpy as np
import os

# Define file paths
DATA_DIR = "."
INPUT_FILE = os.path.join(DATA_DIR, "nordbyen_heat_weather_aligned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")

def engineer_features():
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
    target_col = ['heat_consumption']
    
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
    df['heat_lag_1h'] = df['heat_consumption'].shift(1)
    df['heat_lag_24h'] = df['heat_consumption'].shift(24)
    df['heat_rolling_24h'] = df['heat_consumption'].rolling(window=24).mean()

    # 4. Interaction & Polynomial Features
    print("Engineering Interactions...")
    # Non-linear temperature effect
    df['temp_squared'] = df['temp'] ** 2
    
    # Wind chill proxy (interaction)
    df['temp_wind_interaction'] = df['temp'] * df['wind_speed']
    
    # Weekend interaction (occupancy effect on heating curve)
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
    school_holidays_file = os.path.join(DATA_DIR, "school_holidays.csv")
    if os.path.exists(school_holidays_file):
        school_hol = pd.read_csv(school_holidays_file, parse_dates=['start_date', 'end_date'])
        school_holiday_map = {}
        for _, row in school_hol.iterrows():
            date_range = pd.date_range(start=row['start_date'], end=row['end_date'])
            for d in date_range:
                school_holiday_map[d.date()] = row['description']
        
        df['school_holiday_name'] = df['date'].map(school_holiday_map)
        df['is_school_holiday'] = df['school_holiday_name'].notna().astype(int)
        df['school_holiday_name'] = df['school_holiday_name'].fillna("None")
    else:
        print("Warning: school_holidays.csv not found. Setting school holidays to 0.")
        df['is_school_holiday'] = 0
        df['school_holiday_name'] = "None"
        
    # Drop temporary date column
    df.drop(columns=['date'], inplace=True)

    # Drop intermediate columns if not needed (keeping hour/month/day_of_week for now as they are useful for trees)
    # Dropping NaNs created by lags
    print("Handling Missing Values...")
    original_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {original_len - len(df)} rows due to lags.")

    print(f"Final Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    engineer_features()
