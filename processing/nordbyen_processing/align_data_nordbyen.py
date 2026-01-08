import pandas as pd
import os

# Define file paths
DATA_DIR = ".."
HEAT_FILE = os.path.join(DATA_DIR, "dma_a_nordbyen_heat", "nordbyen_CombinedDataframe_SummedAveraged_withoutOutliers.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "weather", "Weather_Bronderslev_20152022.csv")
OUTPUT_FILE = "nordbyen_heat_weather_aligned.csv"

def align_data():
    print("Loading Heat Data...")
    # Load Heat Data
    # The file has a header row that looks like " ,0". The first column is the index/timestamp, the second is the value.
    # We'll skip the header and name columns manually.
    df_heat = pd.read_csv(HEAT_FILE, header=0, names=['timestamp', 'heat_consumption'])
    
    # Parse timestamps (they are already in UTC format like '2015-04-30 23:00:00+00:00')
    df_heat['timestamp'] = pd.to_datetime(df_heat['timestamp'])
    df_heat.set_index('timestamp', inplace=True)
    
    print(f"Heat Data Loaded. Shape: {df_heat.shape}")
    print(df_heat.head())

    print("\nLoading Weather Data...")
    # Load Weather Data
    df_weather = pd.read_csv(WEATHER_FILE)
    
    # Parse timestamps. Format is '2015-01-01 00:00:00 +0000 UTC'
    # We need to handle the '+0000 UTC' part.
    # Simplest way is to parse with 'mixed' or specify format if consistent.
    # The format seems consistent: '%Y-%m-%d %H:%M:%S +0000 UTC'
    df_weather['timestamp'] = pd.to_datetime(df_weather['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC')
    
    # Localize to UTC (although the string had +0000, pandas might make it naive if we stripped it, 
    # but here we parsed it. Let's ensure it's timezone aware and matches heat data).
    if df_weather['timestamp'].dt.tz is None:
        df_weather['timestamp'] = df_weather['timestamp'].dt.tz_localize('UTC')
    else:
        df_weather['timestamp'] = df_weather['timestamp'].dt.tz_convert('UTC')

    df_weather.set_index('timestamp', inplace=True)
    
    # Select relevant columns
    # User requested to keep all columns
    # weather_cols = ['temp', 'humidity', 'wind_speed', 'clouds_all', 'rain_1h', 'snow_1h']
    # available_cols = [c for c in weather_cols if c in df_weather.columns]
    # df_weather = df_weather[available_cols]
    
    # Drop the original string timestamp column as we have the index now
    if 'dt_iso' in df_weather.columns:
        df_weather.drop(columns=['dt_iso'], inplace=True)

    # Drop constant or empty columns as requested by user
    cols_to_drop = ['city_name', 'lat', 'lon', 'sea_level', 'grnd_level']
    # Only drop if they exist
    cols_to_drop = [c for c in cols_to_drop if c in df_weather.columns]
    if cols_to_drop:
        df_weather.drop(columns=cols_to_drop, inplace=True)

    
    print(f"Weather Data Loaded. Shape: {df_weather.shape}")
    print(df_weather.head())

    print("\nAligning Data...")
    # Merge datasets
    # We use 'inner' join to keep only timestamps present in both (intersection)
    # or 'left' if we want to keep all heat data rows. 
    # Given the goal is forecasting heat with weather as covariate, we need weather for every heat point.
    # Inner join is safest to ensure complete data, but let's check if we lose much.
    
    df_aligned = df_heat.join(df_weather, how='inner')
    
    print(f"Aligned Data Shape: {df_aligned.shape}")
    
    # Handle missing values
    print("\nHandling Missing Values...")
    missing_before = df_aligned.isnull().sum()
    print("Missing values before filling:")
    print(missing_before[missing_before > 0])
    
    # User requested ffill or bfill
    df_aligned.fillna(method='ffill', inplace=True)
    df_aligned.fillna(method='bfill', inplace=True)
    
    missing_after = df_aligned.isnull().sum()
    print("Missing values after filling:")
    print(missing_after[missing_after > 0])

    print(f"\nSaving to {OUTPUT_FILE}...")
    df_aligned.to_csv(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    align_data()
