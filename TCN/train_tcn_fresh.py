"""
TCN Model Training - Fresh Implementation
Simple, clean training script for Nordbyen heat forecasting
"""

import os
import sys
import pickle
import pandas as pd
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
)


def train_tcn_fresh(
    csv_path: str,
    train_end: str,
    val_end: str,
    model_name: str = "tcn_nordbyen_fresh",
    model_save_dir: str = "models",
):
    """Train TCN model - simple and clean"""
    
    cfg = default_feature_config()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    print("=" * 70)
    print("TCN MODEL TRAINING - FRESH START")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path=csv_path,
        train_end=train_end_ts,
        val_end=val_end_ts,
        cfg=cfg,
    )

    train_target = train_scaled["target"]
    train_past = train_scaled["past_covariates"]
    train_future = train_scaled["future_covariates"]
    val_target = val_scaled["target"]
    val_past = val_scaled["past_covariates"]
    val_future = val_scaled["future_covariates"]

    print(f"  Train: {len(train_target)} samples")
    print(f"  Val: {len(val_target)} samples")
    
    # Merge future into past (TCN doesn't support future covariates)
    print("\n[2/4] Merging covariates...")
    if train_future:
        train_past = train_past.stack(train_future) if train_past else train_future
        val_past = val_past.stack(val_future) if val_past else val_future
    print(f"  Total features: {len(train_past.components)}")

    # Setup
    print("\n[3/4] Creating model...")
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{model_name}.pt")
    state_path = os.path.join(model_save_dir, f"{model_name}_preprocessing_state.pkl")
    
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001,
        mode="min",
    )
    
    # Create model - SIMPLE configuration
    model = TCNModel(
        input_chunk_length=168,  # 7 days
        output_chunk_length=24,   # 24 hours
        kernel_size=3,
        num_filters=32,
        num_layers=4,
        dilation_base=2,
        weight_norm=True,
        dropout=0.0,  # NO DROPOUT
        batch_size=32,
        n_epochs=50,
        model_name=model_name,
        random_state=42,
        optimizer_kwargs={"lr": 1e-4},
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",
            "gradient_clip_val": 1.0,
        },
        save_checkpoints=False,  # Avoid checkpoint issues
        force_reset=True,
    )

    print("  Model created")
    print("  - No dropout (dropout=0.0)")
    print("  - No checkpoints")
    print("  - Simple configuration")

    # Train
    print("\n[4/4] Training...")
    print("=" * 70)
    model.fit(
        series=train_target,
        past_covariates=train_past,
        val_series=val_target,
        val_past_covariates=val_past,
        verbose=True,
    )
    print("=" * 70)

    # Save
    print("\nSaving...")
    try:
        model.save(model_path)
        print(f"  Model: {model_path}")
    except Exception as e:
        print(f"  Standard save failed: {e}")
        print("  Saving state_dict instead...")
        import torch
        torch.save(model.model.state_dict(), model_path.replace('.pt', '_statedict.pt'))
        print(f"  State dict: {model_path.replace('.pt', '_statedict.pt')}")
    
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    print(f"  Preprocessing: {state_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_processing", "nordbyen_features_engineered.csv")
    MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

    train_tcn_fresh(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",
        val_end="2019-12-31 23:00:00+00:00",
        model_name="tcn_nordbyen_fresh",
        model_save_dir=MODEL_SAVE_DIR,
    )
