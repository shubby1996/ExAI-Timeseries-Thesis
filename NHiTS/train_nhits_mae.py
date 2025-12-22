"""
NHiTS Deterministic Training (MAE Optimized)
Strict Control Baseline for Probabilistic Experiment
"""

import os
import sys
import pickle
import pandas as pd
import torch
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
)


def train_nhits_deterministic(
    csv_path: str,
    train_end: str,
    val_end: str,
    model_name: str = "nhits_deterministic_mae",
    model_save_dir: str = "models",
):
    """Train NHiTS model optimized for MAE (L1Loss)"""
    
    cfg = default_feature_config()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    print("=" * 70)
    print("NHiTS DETERMINISTIC TRAINING - L1LOSS (MAE)")
    print("Control Group for Probabilistic Experiment")
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
    
    # Create model - L1LOSS configuration
    model = NHiTSModel(
        input_chunk_length=168,
        output_chunk_length=24,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        dropout=0.1,
        activation="ReLU",
        MaxPool1d=True,
        batch_size=32,
        n_epochs=50,
        model_name=model_name,
        loss_fn=torch.nn.L1Loss(),  # CRITICAL: optimize MAE directly
        random_state=42,
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",
        },
        save_checkpoints=True,
        force_reset=True,
    )

    print("  Model created")
    print("  - Loss Function: L1Loss (MAE)")
    
    # Merge future into past (NHiTS requirement)
    print("\n[2/4] Merging covariates...")
    if train_future:
        train_past = train_past.stack(train_future) if train_past else train_future
        val_past = val_past.stack(val_future) if val_past else val_future

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
    model.save(model_path)
    print(f"  Model: {model_path}")
    
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

    train_nhits_deterministic(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",
        val_end="2019-12-31 23:00:00+00:00",
        model_name="nhits_deterministic_mae",
        model_save_dir=MODEL_SAVE_DIR,
    )
