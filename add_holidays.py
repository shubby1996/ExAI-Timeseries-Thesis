import pandas as pd
import holidays
import os

# Define file paths
DATA_DIR = "."
INPUT_FILE = os.path.join(DATA_DIR, "processing", "nordbyen_processing", "nordbyen_features_engineered.csv")
SCHOOL_HOLIDAYS_FILE = os.path.join(DATA_DIR, "school_holidays.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nordbyen_features_with_holidays.csv")

def add_holidays():
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'], index_col='timestamp')
    
    # Ensure we have a date column for mapping
    df['date'] = df.index.date
    
    print("Adding Public Holidays (DK)...")
    # Get years from data
    years = df.index.year.unique().tolist()
    
    # specific country holidays
    dk_holidays = holidays.DK(years=years)
    
    # Map to dataframe
    df['public_holiday_name'] = df['date'].apply(lambda x: dk_holidays.get(x))
    df['is_public_holiday'] = df['public_holiday_name'].notna().astype(int)
    df['public_holiday_name'] = df['public_holiday_name'].fillna("None")
    
    print("Adding School Holidays...")
    if os.path.exists(SCHOOL_HOLIDAYS_FILE):
        school_hol = pd.read_csv(SCHOOL_HOLIDAYS_FILE, parse_dates=['start_date', 'end_date'])
        
        # Create a dictionary of date -> holiday name
        school_holiday_map = {}
        for _, row in school_hol.iterrows():
            date_range = pd.date_range(start=row['start_date'], end=row['end_date'])
            for d in date_range:
                school_holiday_map[d.date()] = row['description']
        
        # Map to dataframe
        df['school_holiday_name'] = df['date'].map(school_holiday_map)
        df['is_school_holiday'] = df['school_holiday_name'].notna().astype(int)
        df['school_holiday_name'] = df['school_holiday_name'].fillna("None")
    else:
        print("Warning: school_holidays.csv not found. Skipping school holidays.")
        df['is_school_holiday'] = 0
        df['school_holiday_name'] = "None"

    # Drop temporary date column
    df.drop(columns=['date'], inplace=True)
    
    print("Sample Holiday Data:")
    print(df[df['is_public_holiday'] == 1][['is_public_holiday', 'public_holiday_name']].head())
    print(df[df['is_school_holiday'] == 1][['is_school_holiday', 'school_holiday_name']].head())

    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    add_holidays()
