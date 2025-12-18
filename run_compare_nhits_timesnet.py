"""
Orchestrator: run_compare_nhits_timesnet.py

Quick smoke end-to-end pipeline that reuses existing project code.
By default runs in "smoke" mode (fast) to validate the full flow.
Use `--full` to attempt full training (may be long).
"""

import os
import sys
import argparse
import subprocess
import pickle
import pandas as pd

# Ensure project root is on path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

from model_preprocessing import default_feature_config, prepare_model_data

# Import NHiTS training helper for full run (but we'll implement a smoke trainer inline)
from NHiTS.train_nhits_prob import train_nhits_probabilistic

# TimesNet training scripts are under timesnet/ and expose main()
import importlib
import shutil
import glob


def run_preprocessing(csv_path, train_end, val_end, save_dir):
    print("[Pipeline] Running preprocessing (prepare_model_data)...")
    cfg = default_feature_config()
    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path=csv_path,
        train_end=pd.Timestamp(train_end),
        val_end=pd.Timestamp(val_end),
        cfg=cfg,
    )
    # Save state for reuse
    os.makedirs(save_dir, exist_ok=True)
    state_path = os.path.join(save_dir, "pipeline_preprocessing_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    print(f"[Pipeline] Preprocessing state saved to {state_path}")
    return state, train_scaled, val_scaled, test_scaled


def standardize_models_and_results(root: str, model_dir: str, results_dir: str):
    """Ensure consistent model locations/names and clean results folder for a fresh full run.

    - Move any TimesNet model folders from `timesnet/models` into `models/timesnet_probabilistic_q`.
    - Ensure `results/` exists and remove stale timesnet_/nhits_ files so outputs are fresh.
    """
    print("[Pipeline] Standardizing model names and cleaning results folder...")
    timesnet_models_src = os.path.join(root, 'timesnet', 'models')
    target_timesnet = os.path.join(model_dir, 'timesnet_probabilistic_q')
    os.makedirs(model_dir, exist_ok=True)

    # Move any timesnet model dir into standardized models folder
    if os.path.exists(timesnet_models_src):
        for d in os.listdir(timesnet_models_src):
            src = os.path.join(timesnet_models_src, d)
            if os.path.isdir(src) and 'timesnet' in d.lower():
                if not os.path.exists(target_timesnet):
                    print(f"[Pipeline] Moving {src} -> {target_timesnet}")
                    shutil.move(src, target_timesnet)
                else:
                    print(f"[Pipeline] Target already exists: {target_timesnet}, removing source {src}")
                    shutil.rmtree(src, ignore_errors=True)
        # remove timesnet/models if empty
        try:
            os.rmdir(timesnet_models_src)
        except OSError:
            pass

    # Also normalize any timesnet folders directly under model_dir (rename latest to standard)
    if os.path.exists(model_dir):
        cand = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'timesnet' in d.lower()]
        if cand and not os.path.exists(target_timesnet):
            # pick most recently modified candidate
            cand.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            chosen = os.path.join(model_dir, cand[0])
            print(f"[Pipeline] Renaming {chosen} -> {target_timesnet}")
            shutil.move(chosen, target_timesnet)

    # Ensure results dir exists and clean old matching files
    os.makedirs(results_dir, exist_ok=True)
    patterns = [os.path.join(results_dir, 'timesnet_*'), os.path.join(results_dir, 'nhits_*'), os.path.join(root, 'timesnet_predictions*'), os.path.join(root, 'nhits_predictions*')]
    removed = 0
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
                removed += 1
            except Exception:
                pass
    print(f"[Pipeline] Cleaned {removed} stale result files (if any).")


def train_nhits_smoke(csv_path, train_end, val_end, model_save_dir):
    """Train a small NHiTS probabilistic model for smoke testing."""
    print("[Pipeline] Training small NHiTS probabilistic model (smoke)")
    import pandas as pd
    from darts.models import NHiTSModel
    from darts.utils.likelihood_models import QuantileRegression
    from pytorch_lightning.callbacks import EarlyStopping
    from model_preprocessing import default_feature_config, prepare_model_data
    import pickle
    import os

    cfg = default_feature_config()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    state, train_scaled, val_scaled, test_scaled = prepare_model_data(
        csv_path=csv_path,
        train_end=train_end_ts,
        val_end=val_end_ts,
        cfg=cfg,
    )

    train_target = train_scaled['target']
    train_past = train_scaled['past_covariates']
    train_future = train_scaled['future_covariates']
    val_target = val_scaled['target']
    val_past = val_scaled['past_covariates']
    val_future = val_scaled['future_covariates']

    # merge future into past (NHiTS requirement)
    if train_future:
        train_past = train_past.stack(train_future) if train_past else train_future
        val_past = val_past.stack(val_future) if val_past else val_future

    early_stopper = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, mode='min')

    os.makedirs(model_save_dir, exist_ok=True)
    model_name = 'nhits_probabilistic_q_smoke'
    model_path = os.path.join(model_save_dir, f"{model_name}.pt")
    state_path = os.path.join(model_save_dir, f"{model_name}_preprocessing_state.pkl")

    model = NHiTSModel(
        input_chunk_length=168,
        output_chunk_length=24,
        num_stacks=2,
        num_blocks=1,
        num_layers=1,
        layer_widths=128,
        dropout=0.1,
        activation='ReLU',
        MaxPool1d=False,
        batch_size=16,
        n_epochs=3,
        model_name=model_name,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        pl_trainer_kwargs={"callbacks": [early_stopper], "accelerator": "auto"},
        save_checkpoints=False,
        force_reset=True,
    )

    model.fit(
        series=train_target,
        past_covariates=train_past,
        val_series=val_target,
        val_past_covariates=val_past,
        verbose=False,
    )

    model.save(model_path)
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)

    print(f"[Pipeline] NHiTS smoke model saved to {model_path}")
    return model_path, state_path


def run_timesnet_train_smoke():
    print("[Pipeline] Running TimesNet probabilistic training (smoke) via timesnet/train_timesnet_prob.py")
    script = os.path.join(ROOT, 'timesnet', 'train_timesnet_prob.py')
    subprocess.check_call([sys.executable, script])
    print("[Pipeline] TimesNet smoke training complete")


def run_predict_evaluate_visualize():
    print('[Pipeline] Running predictions, evaluations and visualizations')
    # NHiTS predict
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'NHiTS', 'predict_nhits_nordbyen.py')])
    # TimesNet predict
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'timesnet', 'predict_timesnet_nordbyen.py')])
    # NHiTS evaluate
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'NHiTS', 'evaluate_nhits_nordbyen.py')])
    # TimesNet evaluate
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'timesnet', 'evaluate_timesnet_nordbyen.py')])
    # Visualize both
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'NHiTS', 'visualize_nhits_predictions.py')])
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'timesnet', 'visualize_timesnet_predictions.py')])
    # Compare
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'timesnet', 'compare_timesnet_vs_nhits.py')])
    print('[Pipeline] Predict/evaluate/visualize/compare completed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full training (may be long)')
    args = parser.parse_args()

    CSV_PATH = os.path.join(ROOT, 'nordbyen_features_engineered.csv')
    MODEL_DIR = os.path.join(ROOT, 'models')
    TRAIN_END = '2018-12-31 23:00:00+00:00'
    VAL_END = '2019-12-31 23:00:00+00:00'

    # 1. Preprocess
    state, train_scaled, val_scaled, test_scaled = run_preprocessing(CSV_PATH, TRAIN_END, VAL_END, MODEL_DIR)

    # Prepare results dir and standardize model names before training
    RESULTS_DIR = os.path.join(ROOT, 'results')
    standardize_models_and_results(ROOT, MODEL_DIR, RESULTS_DIR)

    # 2. Train
    if args.full:
        print('[Pipeline] Full mode: delegating to existing training scripts (this can take a long time)')
        # NHiTS full (call existing script)
        subprocess.check_call([sys.executable, os.path.join(ROOT, 'NHiTS', 'train_nhits_probabilistic.py')])
        # TimesNet: existing script
        subprocess.check_call([sys.executable, os.path.join(ROOT, 'timesnet', 'train_timesnet_prob.py')])
    else:
        # Smoke: small NHiTS trained inline and TimesNet smoke training
        train_nhits_smoke(CSV_PATH, TRAIN_END, VAL_END, MODEL_DIR)
        run_timesnet_train_smoke()

    # 3. Predict / Evaluate / Visualize / Compare
    run_predict_evaluate_visualize()


if __name__ == '__main__':
    main()
