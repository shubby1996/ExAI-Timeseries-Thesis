"""
TCN Model Training Script for Nordbyen Heat Consumption Forecasting.

This script:
1. Loads preprocessed data using prepare_model_data()
2. Instantiates a Darts TCNModel
3. Trains with past and future covariates
4. Saves the trained model and preprocessing state
"""

import os
import sys
import pickle
import pandas as pd
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

# Add parent directory to path to allow importing model_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
    PreprocessingState,
)


def train_tcn_nordbyen(
    csv_path: str,
    train_end: str,
    val_end: str,
    input_chunk_length: int = 168,   # encoder length (7 days)
    output_chunk_length: int = 24,   # forecast horizon (24 hours)
    kernel_size: int = 3,
    num_filters: int = 32,
    num_layers: int = None,  # If None, will be calculated automatically
    dilation_base: int = 2,
    weight_norm: bool = False,
    dropout: float = 0.1,
    batch_size: int = 32,
    n_epochs: int = 50,
    model_name: str = "tcn_nordbyen",
    model_save_dir: str = "models",
) -> None:
    """
    Train a Darts TCNModel on nordbyen_features_engineered.csv.
    """
    cfg = default_feature_config()

    # Convert string dates to pandas timestamps
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    print("=" * 70)
    print("TCN MODEL TRAINING - NORDBYEN HEAT CONSUMPTION")
    print("=" * 70)
    
    print("\nPreparing data for TCN...")
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

    # Define TCN model
    print(f"\n✓ Initializing TCN model:")
    print(f"  Input chunk length: {input_chunk_length} hours")
    print(f"  Output chunk length: {output_chunk_length} hours")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Num filters: {num_filters}")
    
    model = TCNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        kernel_size=kernel_size,
        num_filters=num_filters,
        num_layers=num_layers,
        dilation_base=dilation_base,
        weight_norm=weight_norm,
        dropout=dropout,
        # Use a conservative learning rate to reduce chance of numerical instability
        lr=1e-4,
        # Use Adam optimizer with a small weight decay
        optimizer_cls=None,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
        batch_size=batch_size,
        n_epochs=n_epochs,
        model_name=model_name,
        random_state=42,
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",
            # Clip gradients to avoid exploding gradients producing NaNs
            "gradient_clip_val": 1.0,
        },
        save_checkpoints=True,
        force_reset=True,
    )

    print(f"\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    # Try fitting with future covariates first
    # If TCN doesn't support them, we might need to merge them like NHiTS
    # But let's try standard first as TCN architecture can theoretically handle them
    try:
        model.fit(
            series=train_target,
            past_covariates=train_past,
            future_covariates=train_future,
            val_series=val_target,
            val_past_covariates=val_past,
            val_future_covariates=val_future,
            verbose=True,
        )
    except ValueError as e:
        if "future_covariates" in str(e):
            print("\n⚠ TCNModel does not support future_covariates directly.")
            print("  -> Retrying by merging future covariates into past covariates...")
            
            # Merge future into past
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
            
            # Re-initialize model to be safe (though fit should reset)
            # Actually we can just call fit again
            model.fit(
                series=train_target,
                past_covariates=train_past,
                # future_covariates=None,
                val_series=val_target,
                val_past_covariates=val_past,
                # val_future_covariates=None,
                verbose=True,
            )
        else:
            raise e

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
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

    train_tcn_nordbyen(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",
        val_end="2019-12-31 23:00:00+00:00",
        input_chunk_length=168,
        output_chunk_length=24,
        kernel_size=3,
        num_filters=32,
        dropout=0.1,
        batch_size=32,
        n_epochs=50,
        model_name="tcn_nordbyen",
        model_save_dir=MODEL_SAVE_DIR,
    )
