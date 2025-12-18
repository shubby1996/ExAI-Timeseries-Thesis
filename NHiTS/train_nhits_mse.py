"""
NHiTS Model Training Script for Nordbyen Heat Consumption Forecasting.

This script:
1. Loads preprocessed data using prepare_model_data()
2. Instantiates a Darts NHiTSModel
3. Trains with past and future covariates
4. Saves the trained model and preprocessing state
"""

import os
import sys
import pickle
import pandas as pd
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping

# Add parent directory to path to allow importing model_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
    PreprocessingState,
)


def train_nhits_nordbyen(
    csv_path: str,
    train_end: str,
    val_end: str,
    input_chunk_length: int = 168,   # encoder length (7 days)
    output_chunk_length: int = 24,   # forecast horizon (24 hours)
    num_stacks: int = 3,
    num_blocks: int = 1,
    num_layers: int = 2,
    layer_widths: int = 512,
    dropout: float = 0.1,
    activation: str = "ReLU",
    max_pool_1d: bool = True,
    batch_size: int = 32,
    n_epochs: int = 50,
    model_name: str = "nhits_nordbyen_mse",
    model_save_dir: str = "models",
) -> None:
    """
    Train a Darts NHiTSModel on nordbyen_features_engineered.csv.

    Parameters
    ----------
    csv_path : str
        Path to the engineered features CSV.
    train_end : str
        End timestamp for training period.
    val_end : str
        End timestamp for validation period.
    input_chunk_length : int
        Number of past time steps the model looks at.
    output_chunk_length : int
        Number of future time steps to predict.
    num_stacks : int
        Number of stacks in NHiTS.
    num_blocks : int
        Number of blocks per stack.
    num_layers : int
        Number of layers per block.
    layer_widths : int
        Width of layers.
    pooling_kernel_sizes : tuple
        Pooling kernel sizes for each stack.
    n_freq_downsample : tuple
        Downsampling factors for each stack.
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    max_pool_1d : bool
        Whether to use MaxPool1d.
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
    print("NHiTS MODEL TRAINING - NORDBYEN HEAT CONSUMPTION")
    print("=" * 70)
    
    print("\nPreparing data for NHiTS...")
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
    
    # Debug: Print time ranges
    print(f"  Train target: {train_target.start_time()} to {train_target.end_time()}")
    if train_past:
        print(f"  Train past: {train_past.start_time()} to {train_past.end_time()}")
    if train_future:
        print(f"  Train future: {train_future.start_time()} to {train_future.end_time()}")

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

    # Define NHiTS model
    print(f"\n✓ Initializing NHiTS model:")
    print(f"  Input chunk length: {input_chunk_length} hours")
    print(f"  Output chunk length: {output_chunk_length} hours")
    print(f"  Stacks: {num_stacks}")
    print(f"  Blocks: {num_blocks}")
    print(f"  Layers: {num_layers}")
    
    model = NHiTSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        dropout=dropout,
        activation=activation,
        MaxPool1d=max_pool_1d,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model_name=model_name,
        random_state=42,
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",
        },
        save_checkpoints=True,
        force_reset=True,
    )

    # NHiTS only supports past_covariates, so we merge future_covariates into past_covariates
    print("  -> Merging future_covariates into past_covariates (NHiTS requirement)")
    if train_future:
        if train_past:
            train_past = train_past.stack(train_future)
        else:
            train_past = train_future
        train_future = None

    if val_future:
        if val_past:
            val_past = val_past.stack(val_future)
        else:
            val_past = val_future
        val_future = None

    print(f"\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    model.fit(
        series=train_target,
        past_covariates=train_past,
        # future_covariates=None,  # Not supported by NHiTS
        val_series=val_target,
        val_past_covariates=val_past,
        # val_future_covariates=None,
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
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

    # Data spans: 2015-05 to 2022-05 (roughly 7 years)
    # Splits:
    # - Train: 2015-05 to 2018-12 (~3.5 years)
    # - Val:   2019-01 to 2019-12 (~1 year)
    # - Test:  2020-01 to 2022-05 (~2.5 years)
    
    train_nhits_nordbyen(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",  # UTC timezone
        val_end="2019-12-31 23:00:00+00:00",    # UTC timezone
        input_chunk_length=168,  # 7 days of historical data
        output_chunk_length=24,  # predict next 24 hours
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        batch_size=32,
        n_epochs=50,
        model_name="nhits_nordbyen",
        model_save_dir=MODEL_SAVE_DIR,
    )
