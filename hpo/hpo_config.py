"""
Hyperparameter search spaces for all models.
Defines reasonable ranges based on literature and experience.
"""

def get_nhits_search_space(trial):
    """NHITS hyperparameter search space."""
    return {
        "num_stacks": trial.suggest_int("num_stacks", 2, 5),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "layer_widths": trial.suggest_categorical("layer_widths", [256, 512, 1024]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
    }


def get_tft_search_space(trial):
    """TFT hyperparameter search space (simplified for speed)."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),  # Reduced from [32, 64, 128, 256]
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),  # Reduced from 1-3
        "num_attention_heads": trial.suggest_categorical("num_attention_heads", [2, 4]),  # Reduced from [2, 4, 8]
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),  # Narrower range for stability
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),  # Narrower range for stability
    }


def get_timesnet_search_space(trial):
    """TimesNet hyperparameter search space."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "conv_hidden_size": trial.suggest_categorical("conv_hidden_size", [32, 64, 128]),
        "top_k": trial.suggest_int("top_k", 2, 5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
    }


# Dataset to CSV path mapping
DATASET_PATHS = {
    "heat": "processing/nordbyen_processing/nordbyen_features_engineered.csv",
    "water_centrum": "processing/centrum_processing/centrum_features_engineered_from_2018-04-01.csv",
    "water_tommerby": "processing/tommerby_processing/tommerby_features_engineered_from_2018-04-01.csv",
}

# Training/validation splits per dataset
# Heat: 2015-05 to 2020-11 (use 70% train, 20% val, 10% test)
# Water Centrum: 2018-04 to 2020-11 (use 70% train, 20% val, 10% test)
# Water Tommerby: Similar to centrum
SPLIT_CONFIG = {
    "heat": {
        "train_end": "2019-05-31 23:00:00",  # ~70% of heat data (2015-05 to 2019-05)
        "val_end": "2020-05-31 23:00:00",    # Next 20% (2019-06 to 2020-05)
        # Test: remaining 10% (2020-06 to 2020-11)
    },
    "water_centrum": {
        "train_end": "2019-11-30 23:00:00",  # ~70% of data (2018-04 to 2019-11)
        "val_end": "2020-06-30 23:00:00",    # Next 20% (2019-12 to 2020-06)
        # Test: remaining 10% (2020-07 to 2020-11)
    },
    "water_tommerby": {
        "train_end": "2019-11-30 23:00:00",  # ~70% of data (2018-04 to 2019-11)
        "val_end": "2020-06-30 23:00:00",    # Next 20% (2019-12 to 2020-06)
        # Test: remaining 10% (2020-07 to 2020-11)
    },
}

# HPO training config (faster than full training)
HPO_TRAINING_CONFIG = {
    "n_epochs": 15,  # Reduced for HPO speed (vs 100 for final training)
    "batch_size": 32,
    "input_chunk_length": 168,
    "output_chunk_length": 24,
}

# TFT-specific faster config (TFT is much slower than NHiTS)
HPO_TRAINING_CONFIG_TFT = {
    "n_epochs": 8,  # Much fewer epochs for TFT (vs 15 for NHiTS)
    "batch_size": 64,  # Larger batches = faster training
    "input_chunk_length": 168,
    "output_chunk_length": 24,
}
