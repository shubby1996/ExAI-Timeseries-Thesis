
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import prepare_model_data, default_feature_config

if __name__ == "__main__":
    print("Testing prepare_model_data...")
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    
    TRAIN_END = pd.Timestamp("2018-12-31 23:00:00+00:00")
    VAL_END = pd.Timestamp("2019-12-31 23:00:00+00:00")
    
    try:
        state, train_scaled, val_scaled, test_scaled = prepare_model_data(
            csv_path=CSV_PATH,
            train_end=TRAIN_END,
            val_end=VAL_END
        )
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
