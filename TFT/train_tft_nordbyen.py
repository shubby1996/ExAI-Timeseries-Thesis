"""
TFT Model Training Script for Nordbyen Heat Consumption Forecasting.

This script:
1. Loads preprocessed data using prepare_tft_data()
2. Instantiates a Darts TFTModel
3. Trains with past and future covariates
4. Saves the trained model and preprocessing state
"""

import os
import sys
import pickle
import pandas as pd
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping

# Add parent directory to path to allow importing model_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
    PreprocessingState,
)


def train_tft_nordbyen(
    csv_path: str,
    train_end: str,
    val_end: str,
    input_chunk_length: int = 168,   # encoder length (7 days)
    output_chunk_length: int = 24,   # forecast horizon (24 hours)
    hidden_size: int = 64,
    lstm_layers: int = 1,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 64,
    n_epochs: int = 50,
    model_name: str = "tft_nordbyen",
    model_save_dir: str = "models",
) -> None:
    """
    Train a Darts TFTModel on nordbyen_features_engineered.csv.

    Parameters
    ----------
    csv_path : str
        Path to the engineered features CSV.
    train_end : str
        End timestamp for training period (e.g. "2020-12-31 23:00:00").
    val_end : str
        End timestamp for validation period (e.g. "2021-12-31 23:00:00").
    input_chunk_length : int
        Number of past time steps the model looks at (encoder length).
    output_chunk_length : int
        Number of future time steps to predict (forecast horizon).
    hidden_size : int
        Size of hidden layers.
    lstm_layers : int
        Number of LSTM layers.
    num_attention_heads : int
        Number of attention heads in TFT.
    dropout : float
        Dropout rate.
    batch_size : int
        Training batch size.
    n_epochs : int
        Number of training epochs.
    model_name : str
        Name for the saved model.
    model_save_dir : str
        Directory to save model and preprocessing state.
    """
    cfg = default_feature_config()

    # Convert string dates to pandas timestamps
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    print("=" * 70)
    print("TFT MODEL TRAINING - NORDBYEN HEAT CONSUMPTION")
    print("=" * 70)
    
    print("\nPreparing data for TFT...")
    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path=csv_path,
        train_end=train_end_ts,
        val_end=val_end_ts,
        cfg=cfg,
    )

    # Unpack scaled series
    train_target = train_scaled["target"]
    train_past = train_scaled["past_covariates"]
    train_future = train_scaled["future_covariates"]

    val_target = val_scaled["target"]
    val_past = val_scaled["past_covariates"]
    val_future = val_scaled["future_covariates"]

    print(f"\n✓ Data prepared:")
    print(f"  Train samples: {len(train_target)}")
    print(f"  Val samples: {len(val_target)}")
    print(f"  Past covariates: {train_past.width if train_past else 0} features")
    print(f"  Future covariates: {train_future.width if train_future else 0} features")

    # Setup early stopping
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001,
        mode="min",
    )

    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pt")
    prep_state_save_path = os.path.join(model_save_dir, f"{model_name}_preprocessing_state.pkl")

    # Define TFT model
    print(f"\n✓ Initializing TFT model:")
    print(f"  Input chunk length: {input_chunk_length} hours")
    print(f"  Output chunk length: {output_chunk_length} hours")
    print(f"  Hidden size: {hidden_size}")
    print(f"  LSTM layers: {lstm_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model_name=model_name,
        random_state=42,
        add_relative_index=False,  # we already have cyclical time features
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",  # use GPU if available
        },
        save_checkpoints=True,
        force_reset=True,
    )

    print(f"\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    model.fit(
        series=train_target,
        past_covariates=train_past,
        future_covariates=train_future,
        val_series=val_target,
        val_past_covariates=val_past,
        val_future_covariates=val_future,
        verbose=True,
    )

    print(f"\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Save model
    print(f"\n✓ Saving model to {model_save_path}...")
    model.save(model_save_path)

    # Save preprocessing state (scalers + feature config)
    print(f"✓ Saving preprocessing state to {prep_state_save_path}...")
    with open(prep_state_save_path, "wb") as f:
        pickle.dump(state, f)

    print(f"\n✓ Training artifacts saved successfully!")
    print(f"  Model: {model_save_path}")
    print(f"  Preprocessing state: {prep_state_save_path}")
    
    print(f"\n" + "=" * 70)
    print("READY FOR INFERENCE")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    # Use relative path to data directory (parent of this script)
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_processing", "nordbyen_features_engineered.csv")
    MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

    # Data spans: 2015-05 to 2022-05 (roughly 7 years)
    # Splits:
    # - Train: 2015-05 to 2018-12 (~3.5 years)
    # - Val:   2019-01 to 2019-12 (~1 year)
    # - Test:  2020-01 to 2022-05 (~2.5 years)
    
    train_tft_nordbyen(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",  # UTC timezone
        val_end="2019-12-31 23:00:00+00:00",    # UTC timezone
        input_chunk_length=168,  # 7 days of historical data
        output_chunk_length=24,  # predict next 24 hours
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=64,
        n_epochs=50,
        model_name="tft_nordbyen",
        model_save_dir=MODEL_SAVE_DIR,
    )
