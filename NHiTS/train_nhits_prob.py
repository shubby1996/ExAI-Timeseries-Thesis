"""
NHiTS Probabilistic Training (Quantile Regression)
Experiment Phase 2: Uncertainty Quantification
"""

import os
import sys
import pickle
import pandas as pd
import torch
from darts.models import NHiTSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model_preprocessing import (
    default_feature_config,
    prepare_model_data,
)


def train_nhits_probabilistic(
    csv_path: str,
    train_end: str,
    val_end: str,
    model_name: str = "nhits_probabilistic_q",
    model_save_dir: str = "models",
):
    """Train NHiTS model with Quantile Regression (0.1, 0.5, 0.9)"""
    
    cfg = default_feature_config()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    print("=" * 70)
    print("NHiTS PROBABILISTIC TRAINING - QUANTILE REGRESSION")
    print("Target Quantiles: 0.1, 0.5, 0.9")
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
    
    # Create model - PROBABILISTIC configuration
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
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]), # Standard Probability Configuration
        random_state=42,
        pl_trainer_kwargs={
            "callbacks": [early_stopper],
            "accelerator": "auto",
        },
        save_checkpoints=True,
        force_reset=True,
    )

    print("  Model created")
    print("  - Likelihood: QuantileRegression([0.1, 0.5, 0.9])")
    
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
    CSV_PATH = os.path.join(DATA_DIR, "nordbyen_features_engineered.csv")
    MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")

    train_nhits_probabilistic(
        csv_path=CSV_PATH,
        train_end="2018-12-31 23:00:00+00:00",
        val_end="2019-12-31 23:00:00+00:00",
        model_name="nhits_probabilistic_q",
        model_save_dir=MODEL_SAVE_DIR,
    )
