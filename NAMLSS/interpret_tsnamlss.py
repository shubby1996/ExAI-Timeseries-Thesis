import argparse
import pickle
from functools import partial
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from torch.utils.data import DataLoader
from train_tsnamlss import collate_tensor_only
import os, json

from step1_3_data_pipeline import (
    TSConfig,
    load_and_prepare,
    chronological_split_indices,
    fit_scalers_on_train,
    apply_scalers,
    WindowDataset,
)

from train_tsnamlss import TSNAMLSSNormal, collate_tensor_only  # reuse exact model


'''
we’ll implement two tools:
    - Decomposition per horizon: for a chosen test sample, print/plot
         - endo/exo/time contributions to μ and σ across h=1..24
    - Occlusion heatmaps: lag × horizon for Δμ and Δσ, separately for endo and exo

python interpret_tsnamlss.py \
  --csv_path nordbyen_features_engineered.csv \
  --ckpt best_tsnamlss.pt \
  --sample_idx 100 \
  --out_dir interp_out \
  --device cpu

It will generate these files:
    decomp_mu_sample100.png
    decomp_rawsig_sample100.png
    sigma_sample100.png
    occ_abs_dmu_endo_sample100.png
    occ_abs_dmu_exo_sample100.png
    occ_abs_drawsig_endo_sample100.png
    occ_abs_drawsig_exo_sample100.png

'''


def save_horizon_profile(out_dir: Path, h, series_dict, title, ylabel, fname):
    fig = go.Figure()
    for name, y in series_dict.items():
        fig.add_trace(go.Scatter(x=h, y=y, mode='lines', name=name))
    fig.update_layout(
        title=title,
        xaxis_title="horizon step (1..H)",
        yaxis_title=ylabel,
        hovermode='x unified',
        height=500,
        width=900,
    )
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))


def save_heatmap(out_dir: Path, mat, title, xlabel, ylabel, fname):
    fig = go.Figure(data=go.Heatmap(z=mat, colorscale='Viridis'))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=600,
        width=900,
    )
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))


def save_history_forecast_with_pi(
    out_dir: Path,
    ts_hist,
    y_hist,
    ts_future,
    y_true_future,
    mu_future,
    lo_future,
    hi_future,
    title,
    fname,
):
    fig = go.Figure()
    
    # history
    x_hist = np.arange(len(ts_hist))
    fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='lines', name='history (truth)',
                              line=dict(color='blue')))
    
    # future aligned right after history
    x_fut = np.arange(len(ts_hist), len(ts_hist) + len(ts_future))
    fig.add_trace(go.Scatter(x=x_fut, y=y_true_future, mode='lines', name='future truth',
                              line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_fut, y=mu_future, mode='lines', name='forecast μ',
                              line=dict(color='red')))
    
    # PI as shaded region
    fig.add_trace(go.Scatter(x=x_fut, y=hi_future, fill=None, mode='lines',
                              line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=x_fut, y=lo_future, fill='tonexty', mode='lines',
                              line_color='rgba(0,0,0,0)', name='forecast PI',
                              fillcolor='rgba(255,0,0,0.2)'))
    
    # vertical split marker
    fig.add_vline(x=len(ts_hist) - 1, line_dash="dash", line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title="time (history then forecast)",
        yaxis_title="heat_consumption (original units)",
        hovermode='x unified',
        height=500,
        width=1000,
    )
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))


def save_stackplot(out_dir: Path, h, contribs, labels, total, title, fname):
    fig = go.Figure()
    
    # Add stacked traces
    for i, (contrib, label) in enumerate(zip(contribs, labels)):
        fig.add_trace(go.Scatter(x=h, y=contrib, mode='lines', name=label,
                                  stackgroup='one', fillcolor=px.colors.qualitative.Plotly[i % 10]))
    
    # Add total line on top
    fig.add_trace(go.Scatter(x=h, y=total, mode='lines', name='total',
                              line=dict(color='black', width=3)))
    
    fig.update_layout(
        title=title,
        xaxis_title="horizon step",
        yaxis_title="contribution (scaled)",
        hovermode='x unified',
        height=500,
        width=900,
    )
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))

@torch.no_grad()
def forward_single(model, device, sample, target):
    target_hist = sample["target_hist"].unsqueeze(0).to(device)         # (1,L)
    future_cov = sample["future_cov"].unsqueeze(0).to(device)  # (1,H,num_future_cov)
    
    # Build endogenous dict (AR features)
    endo_hists = {}
    for endo_col in model.endo_cols:
        endo_hists[endo_col] = sample[f"{endo_col}_hist"].unsqueeze(0).to(device)  # (1,L)
    
    # Build exogenous dict
    exo_hists = {}
    for exo_col in model.exo_cols:
        exo_hists[exo_col] = sample[exo_col].unsqueeze(0).to(device)  # (1,L)
    
    out = model(target_hist, endo_hists, exo_hists, future_cov)

    # out["raw"] is additive (endo+exo+future_cov+beta): (1,H,K)
    raw = out["raw"][0].cpu().numpy()  # (H,K)

    # Stream-level contributions: (1,H,K) -> (H,K)
    c_target = out["contrib_target"][0].cpu().numpy()
    c_endo = out["contrib_endo_sum"][0].cpu().numpy()
    c_exo  = out["contrib_exo_sum"][0].cpu().numpy()
    c_future_cov = out["contrib_future_sum"][0].cpu().numpy()
    c_beta = out["beta"][0].cpu().numpy()

    # Per-covariate contributions
    c_per_cov = {}
    for endo_col in model.endo_cols:
        c_per_cov[f"endo_{endo_col}"] = out[f"contrib_endo_{endo_col}"][0].cpu().numpy()
    for exo_col in model.exo_cols:
        c_per_cov[f"exo_{exo_col}"] = out[f"contrib_{exo_col}"][0].cpu().numpy()
    for cov_col in model.future_cov_cols:
        c_per_cov[f"future_{cov_col}"] = out[f"contrib_future_{cov_col}"][0].cpu().numpy()

    mu = out["mu"][0].cpu().numpy()         # (H,)
    sigma = out["sigma"][0].cpu().numpy()   # (H,)

    return {
        "mu": mu,
        "sigma": sigma,
        "raw": raw,
        "target": c_target,
        "endo": c_endo,
        "exo": c_exo,
        "future_cov": c_future_cov,
        "beta": c_beta,
        "per_cov": c_per_cov,
    }


@torch.no_grad()
def occlusion_maps(model, device, sample, target, baseline=0.0):
    """
    Occlusion on scaled inputs.
    baseline=0 corresponds to "mean" after standardization (nice default).
    Returns:
      delta_mu_endo: (L,H)
      delta_rawsig_endo: (L,H)
      delta_mu_exo: (L,H)
      delta_rawsig_exo: (L,H)
    """
    # Original forward
    base = forward_single(model, device, sample, target)
    mu0 = base["mu"]                       # (H,)
    rawsig0 = base["raw"][:, 1]            # raw sigma component (H,)

    L = sample["target_hist"].shape[0]
    H = sample["target_future"].shape[0]

    # allocate
    dmu_endo = np.zeros((L, H), dtype=np.float32)
    drs_endo = np.zeros((L, H), dtype=np.float32)
    dmu_exo  = np.zeros((L, H), dtype=np.float32)
    drs_exo  = np.zeros((L, H), dtype=np.float32)

    # clone tensors for perturbations
    target_hist_orig = sample["target_hist"].clone()
    # Use the first exogenous column for sensitivity analysis
    exo_col_name = model.exo_cols[0] if model.exo_cols else None
    
    if exo_col_name:
        x_hist_orig = sample[exo_col_name].clone()

    for lag in range(L):
        # --- occlude target lag ---
        s1 = dict(sample)
        y_mod = target_hist_orig.clone()
        y_mod[lag] = baseline
        s1["target_hist"] = y_mod
        out1 = forward_single(model, device, s1, target)
        dmu_endo[lag] = (out1["mu"] - mu0)
        drs_endo[lag] = (out1["raw"][:, 1] - rawsig0)

        # --- occlude exo lag (first exogenous column) ---
        if exo_col_name:
            s2 = dict(sample)
            x_mod = x_hist_orig.clone()
            x_mod[lag] = baseline
            s2[exo_col_name] = x_mod
            out2 = forward_single(model, device, s2, target)
            dmu_exo[lag] = (out2["mu"] - mu0)
            drs_exo[lag] = (out2["raw"][:, 1] - rawsig0)

    return dmu_endo, drs_endo, dmu_exo, drs_exo

@torch.no_grad()
def normal_quantile_np(mu_np, sigma_np, q):
    mu = torch.from_numpy(mu_np)
    sig = torch.from_numpy(sigma_np)
    dist = torch.distributions.Normal(mu, sig)
    q_t = torch.tensor(q, dtype=mu.dtype)
    return dist.icdf(q_t).numpy()

@torch.no_grad()
def compute_effects_over_testset(model, test_loader, device, target, out_dir):
    """
    Computes CGA²M+-style effect/importance for stream-level and per-covariate terms:
      - μ contributions: contrib_endo/exo/time[...,0]
      - rawσ contributions: contrib_endo/exo/time[...,1]
    Uses y_future (scaled) as paper-style normalizer and also rawσ as internal normalizer.
    Saves JSON + per-horizon importance plots for both streams and per-covariate.
    """
    model.eval()

    y_all = []
    mu_parts = {"target": [], "endo": [], "exo": [], "future_cov": []}
    rs_parts = {"target": [], "endo": [], "exo": [], "future_cov": []}
    mu_parts_per_cov = {}  # per-covariate μ contributions
    rs_parts_per_cov = {}  # per-covariate rawσ contributions
    rawsig_all = []

    for batch in test_loader:
        target_hist = batch["target_hist"].to(device)
        endo_hists = {k: v.to(device) for k, v in batch["endo_hists"].items()}
        exo_hists = {k: v.to(device) for k, v in batch["exo_hists"].items()}
        future_cov = batch["future_cov"].to(device)
        target_future = batch["target_future"].to(device)  # scaled (B,H)

        out = model(target_hist, endo_hists, exo_hists, future_cov)

        # collect normalizer y
        y_all.append(target_future.detach().cpu())

        # Stream-level mean parts (K=0)
        mu_parts["target"].append(out["contrib_target"][..., 0].detach().cpu())
        mu_parts["endo"].append(out["contrib_endo_sum"][..., 0].detach().cpu())
        mu_parts["exo"].append(out["contrib_exo_sum"][..., 0].detach().cpu())
        mu_parts["future_cov"].append(out["contrib_future_sum"][..., 0].detach().cpu())

        # Stream-level rawsigma parts (K=1)
        rs_parts["target"].append(out["contrib_target"][..., 1].detach().cpu())
        rs_parts["endo"].append(out["contrib_endo_sum"][..., 1].detach().cpu())
        rs_parts["exo"].append(out["contrib_exo_sum"][..., 1].detach().cpu())
        rs_parts["future_cov"].append(out["contrib_future_sum"][..., 1].detach().cpu())

        # Per-covariate μ contributions
        for endo_col in model.endo_cols:
            key = f"endo_{endo_col}"
            if key not in mu_parts_per_cov:
                mu_parts_per_cov[key] = []
            mu_parts_per_cov[key].append(out[f"contrib_endo_{endo_col}"][..., 0].detach().cpu())
        
        for exo_col in model.exo_cols:
            key = f"exo_{exo_col}"
            if key not in mu_parts_per_cov:
                mu_parts_per_cov[key] = []
            mu_parts_per_cov[key].append(out[f"contrib_{exo_col}"][..., 0].detach().cpu())
        
        for cov_col in model.future_cov_cols:
            key = f"future_{cov_col}"
            if key not in mu_parts_per_cov:
                mu_parts_per_cov[key] = []
            mu_parts_per_cov[key].append(out[f"contrib_future_{cov_col}"][..., 0].detach().cpu())

        # Per-covariate rawσ contributions
        for endo_col in model.endo_cols:
            key = f"endo_{endo_col}"
            if key not in rs_parts_per_cov:
                rs_parts_per_cov[key] = []
            rs_parts_per_cov[key].append(out[f"contrib_endo_{endo_col}"][..., 1].detach().cpu())
        
        for exo_col in model.exo_cols:
            key = f"exo_{exo_col}"
            if key not in rs_parts_per_cov:
                rs_parts_per_cov[key] = []
            rs_parts_per_cov[key].append(out[f"contrib_{exo_col}"][..., 1].detach().cpu())
        
        for cov_col in model.future_cov_cols:
            key = f"future_{cov_col}"
            if key not in rs_parts_per_cov:
                rs_parts_per_cov[key] = []
            rs_parts_per_cov[key].append(out[f"contrib_future_{cov_col}"][..., 1].detach().cpu())

        rawsig_all.append(out["raw"][..., 1].detach().cpu())

    y_all_t = torch.cat(y_all, dim=0)  # (N,H)

    mu_target = torch.cat(mu_parts["target"], dim=0)
    mu_endo = torch.cat(mu_parts["endo"], dim=0)
    mu_exo  = torch.cat(mu_parts["exo"],  dim=0)
    mu_future_cov = torch.cat(mu_parts["future_cov"], dim=0)

    rs_target = torch.cat(rs_parts["target"], dim=0)
    rs_endo = torch.cat(rs_parts["endo"], dim=0)
    rs_exo  = torch.cat(rs_parts["exo"],  dim=0)
    rs_future_cov = torch.cat(rs_parts["future_cov"], dim=0)

    rawsig_t = torch.cat(rawsig_all, dim=0)

    # Aggregate per-covariate tensors
    mu_per_cov = {}
    rs_per_cov = {}
    for key in mu_parts_per_cov.keys():
        mu_per_cov[key] = torch.cat(mu_parts_per_cov[key], dim=0)
        rs_per_cov[key] = torch.cat(rs_parts_per_cov[key], dim=0)

    # --- Stream-level effects + importances ---
    mu_effects = {
        "target": effect_global(mu_target, y_all_t),
        "endo": effect_global(mu_endo, y_all_t),
        "exo":  effect_global(mu_exo,  y_all_t),
        "future_cov": effect_global(mu_future_cov, y_all_t),
    }
    mu_imps = normalize_importances(mu_effects)

    # Per-covariate μ effects/importances
    mu_effects_per_cov = {key: effect_global(mu_per_cov[key], y_all_t) for key in mu_per_cov.keys()}
    mu_imps_per_cov = normalize_importances(mu_effects_per_cov)

    rs_effects_y = {
        "target": effect_global(rs_target, y_all_t),
        "endo": effect_global(rs_endo, y_all_t),
        "exo":  effect_global(rs_exo,  y_all_t),
        "future_cov": effect_global(rs_future_cov, y_all_t),
    }
    rs_imps_y = normalize_importances(rs_effects_y)

    # Per-covariate rawσ effects/importances (norm by y)
    rs_effects_y_per_cov = {key: effect_global(rs_per_cov[key], y_all_t) for key in rs_per_cov.keys()}
    rs_imps_y_per_cov = normalize_importances(rs_effects_y_per_cov)

    rs_effects_rs = {
        "target": effect_global(rs_target, rawsig_t),
        "endo": effect_global(rs_endo, rawsig_t),
        "exo":  effect_global(rs_exo,  rawsig_t),
        "future_cov": effect_global(rs_future_cov, rawsig_t),
    }
    rs_imps_rs = normalize_importances(rs_effects_rs)

    # Per-covariate rawσ effects/importances (norm by rawσ)
    rs_effects_rs_per_cov = {key: effect_global(rs_per_cov[key], rawsig_t) for key in rs_per_cov.keys()}
    rs_imps_rs_per_cov = normalize_importances(rs_effects_rs_per_cov)

    # --- Per-horizon effects + importances (stream-level) ---
    mu_eff_h = {
        "target": effect_by_horizon(mu_target, y_all_t),
        "endo": effect_by_horizon(mu_endo, y_all_t),
        "exo":  effect_by_horizon(mu_exo,  y_all_t),
        "future_cov": effect_by_horizon(mu_future_cov, y_all_t),
    }
    mu_imp_h = normalize_importances_by_horizon(mu_eff_h)

    # Per-horizon per-covariate μ effects/importances
    mu_eff_h_per_cov = {key: effect_by_horizon(mu_per_cov[key], y_all_t) for key in mu_per_cov.keys()}
    mu_imp_h_per_cov = normalize_importances_by_horizon(mu_eff_h_per_cov)

    rs_eff_h_y = {
        "target": effect_by_horizon(rs_target, y_all_t),
        "endo": effect_by_horizon(rs_endo, y_all_t),
        "exo":  effect_by_horizon(rs_exo,  y_all_t),
        "future_cov": effect_by_horizon(rs_future_cov, y_all_t),
    }
    rs_imp_h_y = normalize_importances_by_horizon(rs_eff_h_y)

    # Per-horizon per-covariate rawσ effects/importances (norm by y)
    rs_eff_h_y_per_cov = {key: effect_by_horizon(rs_per_cov[key], y_all_t) for key in rs_per_cov.keys()}
    rs_imp_h_y_per_cov = normalize_importances_by_horizon(rs_eff_h_y_per_cov)

    rs_eff_h_rs = {
        "target": effect_by_horizon(rs_target, rawsig_t),
        "endo": effect_by_horizon(rs_endo, rawsig_t),
        "exo":  effect_by_horizon(rs_exo,  rawsig_t),
        "future_cov": effect_by_horizon(rs_future_cov, rawsig_t),
    }
    rs_imp_h_rs = normalize_importances_by_horizon(rs_eff_h_rs)

    # Per-horizon per-covariate rawσ effects/importances (norm by rawσ)
    rs_eff_h_rs_per_cov = {key: effect_by_horizon(rs_per_cov[key], rawsig_t) for key in rs_per_cov.keys()}
    rs_imp_h_rs_per_cov = normalize_importances_by_horizon(rs_eff_h_rs_per_cov)

    payload = {
        "mu_effects_norm_y": mu_effects,
        "mu_importances_norm_y": mu_imps,
        "mu_effects_per_cov_norm_y": mu_effects_per_cov,
        "mu_importances_per_cov_norm_y": mu_imps_per_cov,
        "rawsig_effects_norm_y": rs_effects_y,
        "rawsig_importances_norm_y": rs_imps_y,
        "rawsig_effects_per_cov_norm_y": rs_effects_y_per_cov,
        "rawsig_importances_per_cov_norm_y": rs_imps_y_per_cov,
        "rawsig_effects_norm_rawsig": rs_effects_rs,
        "rawsig_importances_norm_rawsig": rs_imps_rs,
        "rawsig_effects_per_cov_norm_rawsig": rs_effects_rs_per_cov,
        "rawsig_importances_per_cov_norm_rawsig": rs_imps_rs_per_cov,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "effect_importance_stream_level.json", "w") as f:
        json.dump(payload, f, indent=2)

    np.save(out_dir / "mu_importance_by_horizon_norm_y.npy", mu_imp_h)
    np.save(out_dir / "mu_importance_by_horizon_per_cov_norm_y.npy", mu_imp_h_per_cov)
    np.save(out_dir / "rawsig_importance_by_horizon_norm_y.npy", rs_imp_h_y)
    np.save(out_dir / "rawsig_importance_by_horizon_per_cov_norm_y.npy", rs_imp_h_y_per_cov)
    np.save(out_dir / "rawsig_importance_by_horizon_norm_rawsig.npy", rs_imp_h_rs)
    np.save(out_dir / "rawsig_importance_by_horizon_per_cov_norm_rawsig.npy", rs_imp_h_rs_per_cov)

    # Stream-level plots
    plot_importance_by_horizon(
        out_dir, mu_imp_h,
        title="MU stream importance by horizon (normalized per horizon, norm by y)",
        fname="mu_importance_by_horizon.png",
    )
    plot_importance_by_horizon(
        out_dir, rs_imp_h_y,
        title="RAWSIG stream importance by horizon (normalized per horizon, norm by y)",
        fname="rawsig_importance_by_horizon_norm_y.png",
    )
    plot_importance_by_horizon(
        out_dir, rs_imp_h_rs,
        title="RAWSIG stream importance by horizon (normalized per horizon, norm by rawsig)",
        fname="rawsig_importance_by_horizon_norm_rawsig.png",
    )

    # Per-covariate plots
    plot_importance_by_horizon(
        out_dir, mu_imp_h_per_cov,
        title="MU per-covariate importance by horizon (normalized per horizon, norm by y)",
        fname="mu_importance_by_horizon_per_cov.png",
    )
    plot_importance_by_horizon(
        out_dir, rs_imp_h_y_per_cov,
        title="RAWSIG per-covariate importance by horizon (normalized per horizon, norm by y)",
        fname="rawsig_importance_by_horizon_per_cov_norm_y.png",
    )
    plot_importance_by_horizon(
        out_dir, rs_imp_h_rs_per_cov,
        title="RAWSIG per-covariate importance by horizon (normalized per horizon, norm by rawsig)",
        fname="rawsig_importance_by_horizon_per_cov_norm_rawsig.png",
    )

    print("\n=== EFFECT / IMPORTANCE (TEST SET, stream-level) ===")
    print("MU importances:", mu_imps)
    print("RAWSIG importances (norm by y):", rs_imps_y)
    print("RAWSIG importances (norm by rawsig):", rs_imps_rs)

    print("\n=== EFFECT / IMPORTANCE (TEST SET, per-covariate, μ) ===")
    print("MU importances per covariate:", mu_imps_per_cov)

    print("\n=== EFFECT / IMPORTANCE (TEST SET, per-covariate, rawσ norm by y) ===")
    print("RAWSIG importances per covariate (norm by y):", rs_imps_y_per_cov)

    print("\n=== EFFECT / IMPORTANCE (TEST SET, per-covariate, rawσ norm by rawσ) ===")
    print("RAWSIG importances per covariate (norm by rawsig):", rs_imps_rs_per_cov)

    print("Saved effect/importance artifacts (stream-level + per-covariate) to:", out_dir.resolve())


@torch.no_grad()
def plot_full_dataset_forecasts(model, df_raw, scalers, cfg, train_rng, val_rng, test_rng, device, out_dir):
    """
    Generate forecasts for entire train, val, test sets and create comprehensive plot 
    showing full timeline with ground truth, predictions, and prediction intervals.
    Uses efficient batching for speed.
    """
    from step1_3_data_pipeline import apply_scalers, WindowDataset
    from torch.utils.data import DataLoader
    
    print("\n=== Generating full dataset forecasts ===")
    model.eval()
    
    # Apply scalers to get normalized data
    df = apply_scalers(df_raw, cfg, scalers)
    
    # Create datasets for all splits
    ds_train = WindowDataset(df, cfg, train_rng)
    ds_val = WindowDataset(df, cfg, val_rng)
    ds_test = WindowDataset(df, cfg, test_rng)
    
    all_datasets = [
        ("Train", ds_train, 'blue'),
        ("Val", ds_val, 'orange'),
        ("Test", ds_test, 'green')
    ]
    
    # Get scaler params for denormalization
    y_mean = float(scalers[cfg.target].mean_[0])
    y_std = float(np.sqrt(scalers[cfg.target].var_[0]))
    
    # Collect all forecasts
    all_predictions = []
    all_timestamps = []
    all_ground_truth = []
    all_lo = []
    all_hi = []
    all_splits = []
    
    # Create collate function
    collate_fn = partial(collate_tensor_only, target=cfg.target, endo_cols=cfg.endo_cols, exo_cols=cfg.exo_cols)
    
    for split_name, dataset, color in all_datasets:
        print(f"Processing {split_name} set ({len(dataset)} samples)...")
        
        # Use DataLoader for efficient batching
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, 
                           drop_last=False, collate_fn=collate_fn)
        
        batch_idx = 0
        for batch in loader:
            target_hist = batch["target_hist"].to(device)
            endo_hists = {k: v.to(device) for k, v in batch["endo_hists"].items()}
            exo_hists = {k: v.to(device) for k, v in batch["exo_hists"].items()}
            future_cov = batch["future_cov"].to(device)
            
            # Forward pass
            out = model(target_hist, endo_hists, exo_hists, future_cov)
            
            # Denormalize to original units
            mu_orig = (out["mu"][:, 0].cpu().numpy() * y_std + y_mean)  # Take first forecast step
            sigma_orig = out["sigma"][:, 0].cpu().numpy() * y_std
            
            # Get batch indices
            batch_size = target_hist.shape[0]
            start_idx = batch_idx * 512
            end_idx = min(start_idx + batch_size, len(dataset))
            
            # Get ground truth and timestamps for this batch
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                t_center = dataset.centers[global_idx]
                y_true_orig = df_raw[cfg.target].iloc[t_center + 1]  # First forecast step
                ts = df_raw.index[t_center + 1]
                
                # 80% PI using scipy.stats.norm
                from scipy.stats import norm
                lo = norm.ppf(0.1, loc=mu_orig[i], scale=sigma_orig[i])
                hi = norm.ppf(0.9, loc=mu_orig[i], scale=sigma_orig[i])
                
                # Store
                all_predictions.append(mu_orig[i])
                all_timestamps.append(ts)
                all_ground_truth.append(y_true_orig)
                all_lo.append(lo)
                all_hi.append(hi)
                all_splits.append(split_name)
            
            batch_idx += 1
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_lo = np.array(all_lo)
    all_hi = np.array(all_hi)
    
    print(f"Total forecasts generated: {len(all_predictions)}")
    
    # Create comprehensive Plotly figure
    fig = go.Figure()
    
    # Add ground truth
    fig.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_ground_truth,
        mode='lines',
        name='Ground Truth',
        line=dict(color='black', width=1),
        opacity=0.7
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_predictions,
        mode='lines',
        name='Forecast (μ)',
        line=dict(color='red', width=1.5)
    ))
    
    # Add prediction interval as shaded region
    fig.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_hi,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=all_timestamps,
        y=all_lo,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='80% PI',
        fillcolor='rgba(255,0,0,0.2)',
        hoverinfo='skip'
    ))
    
    # Add vertical lines for split boundaries using shapes
    # Find boundaries - ensure we have valid indices
    if len(ds_train) > 0 and len(ds_val) > 0:
        train_end_ts = all_timestamps[len(ds_train) - 1]
        val_end_ts = all_timestamps[len(ds_train) + len(ds_val) - 1]
        
        # Use shapes instead of vline to avoid timestamp issues
        fig.add_shape(type="line",
                     x0=train_end_ts, x1=train_end_ts,
                     y0=0, y1=1, yref="paper",
                     line=dict(color="blue", width=2, dash="dash"))
        fig.add_annotation(x=train_end_ts, y=1, yref="paper",
                          text="Train/Val Split", showarrow=False,
                          yanchor="bottom")
        
        fig.add_shape(type="line",
                     x0=val_end_ts, x1=val_end_ts,
                     y0=0, y1=1, yref="paper",
                     line=dict(color="orange", width=2, dash="dash"))
        fig.add_annotation(x=val_end_ts, y=1, yref="paper",
                          text="Val/Test Split", showarrow=False,
                          yanchor="bottom")
    
    # Calculate MAE for each split
    train_mae = np.mean(np.abs(all_predictions[:len(ds_train)] - all_ground_truth[:len(ds_train)]))
    val_mae = np.mean(np.abs(all_predictions[len(ds_train):len(ds_train)+len(ds_val)] - 
                              all_ground_truth[len(ds_train):len(ds_train)+len(ds_val)]))
    test_mae = np.mean(np.abs(all_predictions[len(ds_train)+len(ds_val):] - 
                               all_ground_truth[len(ds_train)+len(ds_val):]))
    
    # Layout
    fig.update_layout(
        title=(f"Full Dataset Forecasts: {cfg.target} (1-step ahead)<br>" + 
               f"Train={len(ds_train)} samples (MAE={train_mae:.4f} kW) | " +
               f"Val={len(ds_val)} (MAE={val_mae:.4f} kW) | " +
               f"Test={len(ds_test)} (MAE={test_mae:.4f} kW)"),
        xaxis_title="Date",
        yaxis_title=f"{cfg.target} (kW)",
        hovermode='x unified',
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Save
    fname = out_dir / "full_dataset_forecasts_1step.html"
    fig.write_html(str(fname))
    print(f"✓ Saved full dataset forecast plot to: {fname}")
    
    # Also create a multi-step version (showing all 24 forecast horizons)
    print("\nGenerating multi-step horizon plot for test set...")
    plot_full_dataset_multistep(model, df_raw, scalers, cfg, test_rng, device, out_dir)
    
    return fname


@torch.no_grad()
def plot_full_dataset_multistep(model, df_raw, scalers, cfg, test_rng, device, out_dir, max_samples=500):
    """
    Create a detailed plot showing multi-step forecasts for test set.
    For visualization clarity, limit to max_samples.
    """
    from step1_3_data_pipeline import apply_scalers, WindowDataset
    
    model.eval()
    df = apply_scalers(df_raw, cfg, scalers)
    ds_test = WindowDataset(df, cfg, test_rng)
    
    # Limit samples for clarity
    n_samples = min(len(ds_test), max_samples)
    
    y_mean = float(scalers[cfg.target].mean_[0])
    y_std = float(np.sqrt(scalers[cfg.target].var_[0]))
    
    fig = go.Figure()
    
    # Plot every Nth sample to avoid overcrowding
    step = max(1, n_samples // 100)  # ~100 forecast trajectories max
    
    for idx in range(0, n_samples, step):
        sample = ds_test[idx]
        out = forward_single(model, device, sample, cfg.target)
        
        # Denormalize
        mu_orig = out["mu"] * y_std + y_mean
        sigma_orig = out["sigma"] * y_std
        
        # Ground truth
        t_center = ds_test.centers[idx]
        y_true_orig = df_raw[cfg.target].iloc[t_center + 1 : t_center + cfg.H + 1].to_numpy()
        ts_future = df_raw.index[t_center + 1 : t_center + cfg.H + 1]
        
        # 80% PI
        lo = normal_quantile_np(mu_orig, sigma_orig, 0.1)
        hi = normal_quantile_np(mu_orig, sigma_orig, 0.9)
        
        # Add traces (with grouping to avoid legend clutter)
        showlegend = (idx == 0)
        
        # Ground truth
        fig.add_trace(go.Scatter(
            x=ts_future,
            y=y_true_orig,
            mode='lines',
            name='Ground Truth' if showlegend else None,
            line=dict(color='green', width=1),
            opacity=0.5,
            showlegend=showlegend,
            legendgroup='truth'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=ts_future,
            y=mu_orig,
            mode='lines',
            name='Forecast' if showlegend else None,
            line=dict(color='red', width=1),
            opacity=0.6,
            showlegend=showlegend,
            legendgroup='forecast'
        ))
        
        # PI
        fig.add_trace(go.Scatter(
            x=ts_future,
            y=hi,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            legendgroup='pi'
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_future,
            y=lo,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='80% PI' if showlegend else None,
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=showlegend,
            legendgroup='pi'
        ))
    
    fig.update_layout(
        title=f"Test Set Multi-Step Forecasts (24h horizon)<br>" +
              f"Showing {n_samples // step} sample trajectories from {n_samples} test samples",
        xaxis_title="Date",
        yaxis_title=f"{cfg.target} (kW)",
        hovermode='x unified',
        height=700,
        width=1400,
        showlegend=True
    )
    
    fname = out_dir / "full_dataset_forecasts_multistep_test.html"
    fig.write_html(str(fname))
    print(f"Saved multi-step forecast plot to: {fname}")
    
    return fname


def save_forecast_with_pi(out_dir: Path, ts_list, y_true, y_pred, lo, hi, title, fname):
    fig = go.Figure()
    
    x = np.arange(len(ts_list))
    
    # PI as shaded region
    fig.add_trace(go.Scatter(x=x, y=hi, fill=None, mode='lines',
                              line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=lo, fill='tonexty', mode='lines',
                              line_color='rgba(0,0,0,0)', name='PI',
                              fillcolor='rgba(0,100,200,0.2)'))
    
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='y_true',
                              line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='mu_pred',
                              line=dict(color='red', width=2)))
    
    sparse_ticks = [0, len(x)//2, len(x)-1]
    sparse_labels = [str(ts_list[0]), str(ts_list[len(x)//2]), str(ts_list[-1])]
    
    fig.update_layout(
        title=title,
        xaxis_title="forecast horizon (timestamps shown sparsely)",
        yaxis_title="heat_consumption (original units)",
        hovermode='x unified',
        height=500,
        width=900,
    )
    fig.update_xaxes(tickvals=sparse_ticks, ticktext=sparse_labels)
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))


def plot_importance_by_horizon(out_dir: Path, imp_h: dict, title: str, fname: str):
    keys = list(imp_h.keys())
    H = imp_h[keys[0]].shape[0]
    x = np.arange(1, H + 1)

    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scatter(x=x, y=imp_h[k], mode='lines', name=k))
    
    fig.update_layout(
        title=title,
        xaxis_title="Horizon step h",
        yaxis_title="Importance",
        hovermode='x unified',
        height=500,
        width=900,
        yaxis=dict(range=[0.0, 1.0]),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.write_html(str(out_dir / fname.replace('.png', '.html')))


def effect_global(term: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> float:
    term_mean = term.mean()
    denom_mean = denom.mean()
    num = (term - term_mean).abs().sum()
    den = (denom - denom_mean).abs().sum().clamp_min(eps)
    return (num / den).item()

def effect_by_horizon(term: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    term_mean_h = term.mean(dim=0, keepdim=True)
    denom_mean_h = denom.mean(dim=0, keepdim=True)
    num_h = (term - term_mean_h).abs().sum(dim=0)
    den_h = (denom - denom_mean_h).abs().sum(dim=0).clamp_min(eps)
    return (num_h / den_h).detach().cpu().numpy()

def normalize_importances(effects: dict) -> dict:
    total = float(sum(effects.values()))
    if total <= 0:
        return {k: 0.0 for k in effects}
    return {k: float(v) / total for k, v in effects.items()}

def normalize_importances_by_horizon(effects_h: dict) -> dict:
    keys = list(effects_h.keys())
    H = effects_h[keys[0]].shape[0]
    denom = np.zeros(H, dtype=np.float64)
    for k in keys:
        denom += effects_h[k]
    denom = np.clip(denom, 1e-12, None)
    return {k: (effects_h[k] / denom).astype(np.float64) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="best_tsnamlss.pt")
    ap.add_argument("--L", type=int, default=168)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sample_idx", type=int, default=100, help="Index within TEST dataset")
    ap.add_argument("--out_dir", type=str, default="interp_out")
    ap.add_argument("--preprocessing_state", type=str, default=None,
                    help="Path to NAMLSS_preprocessing_state.pkl saved by the benchmarker. "
                         "When provided, scalers and TSConfig are loaded from this file to ensure "
                         "interpret uses the exact same preprocessing as benchmarking/eval.")
    ap.add_argument("--test_start_str", type=str, default=None,
                    help="(Optional) start date for test split, e.g. '2023-01-01'.")
    ap.add_argument("--test_end_str", type=str, default=None,
                    help="(Optional) end date for test split, e.g. '2023-06-30'.")
    ap.add_argument("--do_effects", action="store_true", help="Compute dataset-level effect/importance over the full test set")
    ap.add_argument("--effects_batch_size", type=int, default=512)
    ap.add_argument("--effects_max_batches", type=int, default=-1, help="Limit batches for quick runs; -1 uses all")
    ap.add_argument("--plot_full_dataset", action="store_true", help="Generate full dataset forecast plot (train + val + test)")

    args = ap.parse_args()
    device = torch.device(args.device)

    # Load checkpoint and detect format (train_tsnamlss vs benchmarker raw state_dict)
    raw_ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(raw_ckpt, dict) and "model_state" in raw_ckpt:
        ckpt = raw_ckpt
        print("✓ Detected train_tsnamlss.py checkpoint format")
    else:
        ckpt = {"model_state": raw_ckpt, "cfg": None}
        print("✓ Detected benchmarker checkpoint format (raw state dict)")

    # ------------------------------------------------------------------
    # Load preprocessing state (scalers + TSConfig) if provided, otherwise
    # fall back to deriving/fit scalers locally (may differ from benchmarker)
    # ------------------------------------------------------------------
    scalers = None
    if args.preprocessing_state is not None:
        print(f"✓ Loading preprocessing state from: {args.preprocessing_state}")
        with open(args.preprocessing_state, "rb") as f:
            preprocessing_state = pickle.load(f)
        cfg = preprocessing_state["cfg_obj"]
        scalers = preprocessing_state["scalers"]
        print(f"  target:           {cfg.target}")
        print(f"  endo_cols:        {cfg.endo_cols}")
        print(f"  exo_cols:         {cfg.exo_cols}")
        print(f"  future_cov_cols:  {cfg.future_cov_cols}")
    else:
        # Start from args
        cfg = TSConfig(L=args.L, H=args.H)
        # If checkpoint contains cfg, prefer its values
        if ckpt.get("cfg") is not None:
            saved_cfg = ckpt["cfg"]
            if "target" in saved_cfg:          cfg.target = saved_cfg["target"]
            if "endo_cols" in saved_cfg:       cfg.endo_cols = saved_cfg["endo_cols"]
            if "exo_cols" in saved_cfg:        cfg.exo_cols = saved_cfg["exo_cols"]
            if "future_cov_cols" in saved_cfg: cfg.future_cov_cols = saved_cfg["future_cov_cols"]

    # If no .pkl was provided and cfg has no exo_cols, try to detect from state_dict
    state_dict = ckpt["model_state"]
    has_exo_nets = any("exo_nets" in key for key in state_dict.keys())
    if args.preprocessing_state is None and not cfg.exo_cols:
        exo_cols = []
        for key in state_dict.keys():
            if key.startswith("exo_nets."):
                col = key.split(".")[1]
                if col not in exo_cols:
                    exo_cols.append(col)
        cfg.exo_cols = exo_cols

    if not has_exo_nets:
        raise ValueError("Legacy architecture no longer supported. Please use the configurable architecture.")

    print(f"\n✓ Model config:")
    print(f"  target:           {cfg.target}")
    print(f"  endo_cols:        {cfg.endo_cols}")
    print(f"  exo_cols:         {cfg.exo_cols}")
    print(f"  future_cov_cols:  {cfg.future_cov_cols}")

    # Prepare data (now that cfg is known)
    df_raw = load_and_prepare(Path(args.csv_path), cfg)
    n = len(df_raw)
    train_rng, val_rng, test_rng = chronological_split_indices(n, cfg.train_frac, cfg.val_frac)

    # Scalers: reuse from .pkl if available, otherwise refit on train
    if scalers is not None:
        print("✓ Reusing benchmarker scalers (no refitting)")
        df = apply_scalers(df_raw, cfg, scalers)
    else:
        scalers = fit_scalers_on_train(df_raw, cfg, train_rng)
        df = apply_scalers(df_raw, cfg, scalers)
        print("✓ Fitted fresh scalers (no --preprocessing_state provided)")

    ds_test = WindowDataset(df, cfg, test_rng)
    if len(ds_test) == 0:
        raise RuntimeError("Test dataset is empty.")

    sample_idx = max(0, min(args.sample_idx, len(ds_test) - 1))
    sample = ds_test[sample_idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # #==============================================================================================
    #     # --- DEBUG: print original-unit values at influential lags ---
    # t_center = ds_test.centers[sample_idx]          # index of "end of history" in df
    # hist_start = t_center - (cfg.L - 1)             # index of start of history window

    # lags_temp = [167, 166, 65, 62, 59]              # your interesting exo lags
    # lags_y    = [167, 166, 165]                     # recent endo lags

    # print("\n=== DEBUG: Original-unit values at selected lags ===")
    # print("History window:", df_raw.index[hist_start], "->", df_raw.index[t_center])
    # print("Forecast window:", df_raw.index[t_center+1], "->", df_raw.index[t_center+cfg.H])

    # def print_lag_block(lags, colname, label):
    #     print(f"\n[{label}] column={colname}")
    #     for lag in lags:
    #         idx = hist_start + lag
    #         ts = df_raw.index[idx]
    #         val = df_raw.iloc[idx][colname]
    #         hours_ago = (cfg.L - 1) - lag
    #         print(f"lag {lag:3d} | {hours_ago:3d}h ago | {ts} | {colname}={val}")

    # # ==============================================================================================



    # Load model with correct architecture
    model = TSNAMLSSNormal(L=args.L, H=args.H, target=cfg.target, endo_cols=cfg.endo_cols, 
                           exo_cols=cfg.exo_cols, future_cov_cols=cfg.future_cov_cols).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

# =============================================================================

    if args.plot_full_dataset:
        plot_full_dataset_forecasts(model, df_raw, scalers, cfg, train_rng, val_rng, test_rng, device, out_dir)
        print("\n✓ Full dataset plots generated successfully!")
        # continue to optionally compute effects and per-sample decompositions

    if args.do_effects:
        # Create collate function with target, endo and exo columns
        collate_fn = partial(collate_tensor_only, target=cfg.target, endo_cols=cfg.endo_cols, exo_cols=cfg.exo_cols)
        test_loader = DataLoader(
            ds_test,
            batch_size=args.effects_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Optional batch limiting (quick debug)
        if args.effects_max_batches > 0:
            from itertools import islice
            test_loader = list(islice(test_loader, args.effects_max_batches))

        compute_effects_over_testset(model, test_loader, device, cfg.target, out_dir)


# ==============================================================================
    # ---------
    # 8A) Decomposition per horizon
    # ---------
    out = forward_single(model, device, sample, cfg.target)

# ===============================================================================
        # --- Forecast plot: truth vs predicted + PI in ORIGINAL units ---
    y_mean = float(scalers[cfg.target].mean_[0])
    y_std  = float(np.sqrt(scalers[cfg.target].var_[0]))

    # predicted in original units
    mu_orig = out["mu"] * y_std + y_mean
    sigma_orig = out["sigma"] * y_std

    # truth in original units (use df_raw, not scaled df)
    t_center = ds_test.centers[sample_idx]
    y_true_orig = df_raw[cfg.target].iloc[t_center+1 : t_center+cfg.H+1].to_numpy()

    # timestamps for forecast horizon
    ts_future = df_raw.index[t_center+1 : t_center+cfg.H+1].to_list()

    # 80% PI (0.1, 0.9) — match your evaluation
    lo = normal_quantile_np(mu_orig, sigma_orig, 0.1)
    hi = normal_quantile_np(mu_orig, sigma_orig, 0.9)

    save_forecast_with_pi(
        out_dir,
        ts_future,
        y_true_orig,
        mu_orig,
        lo,
        hi,
        title=f"Forecast vs truth with 80% PI (sample {sample_idx})",
        fname=f"forecast_pi_sample{sample_idx}.png",
    )

# ===============================================================================

        # --- Full timeline plot: 168h history + 24h forecast (truth + PI) ---
    hist_start = t_center - (cfg.L - 1)
    hist_end = t_center

    # history truth in original units
    y_hist_orig = df_raw[cfg.target].iloc[hist_start : hist_end + 1].to_numpy()
    ts_hist = df_raw.index[hist_start : hist_end + 1].to_list()

    save_history_forecast_with_pi(
        out_dir=out_dir,
        ts_hist=ts_hist,
        y_hist=y_hist_orig,
        ts_future=ts_future,
        y_true_future=y_true_orig,
        mu_future=mu_orig,
        lo_future=lo,
        hi_future=hi,
        title=f"168h history + 24h forecast with 80% PI (sample {sample_idx})",
        fname=f"history168_forecast24_pi_sample{sample_idx}.png",
    )

# ===============================================================================
    # --- Z-score plot (plotly) ---

    z = (y_true_orig - mu_orig) / (sigma_orig + 1e-12)
    fig = go.Figure()
    
    x = np.arange(1, cfg.H+1)
    fig.add_trace(go.Scatter(x=x, y=z, mode='lines+markers', name='z = (y-mu)/sigma',
                              line=dict(color='blue', width=2)))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="black", annotation_text="0")
    fig.add_hline(y=1.96, line_dash="dash", line_color="red", annotation_text="1.96")
    fig.add_hline(y=-1.96, line_dash="dash", line_color="red", annotation_text="-1.96")
    
    fig.update_layout(
        title=f"Standardized residuals by horizon (sample {sample_idx})",
        xaxis_title="horizon step",
        yaxis_title="z-score",
        hovermode='x unified',
        height=500,
        width=900,
    )
    fig.write_html(str(out_dir / f"zscore_sample{sample_idx}.html"))

# ===============================================================================
    h = np.arange(1, cfg.H + 1)

    # μ stacked
    contribs_mu = [
        out["target"][:, 0],
        out["endo"][:, 0],
        out["exo"][:, 0],
        out["future_cov"][:, 0],
        out["beta"][:, 0],
    ]
    labels_mu = ["target_mu", "endo_mu", "exo_mu", "future_cov_mu", "beta_mu"]
    save_stackplot(
        out_dir, h, contribs_mu, labels_mu, out["mu"],
        title=f"Stacked contributions to μ (sample {sample_idx})",
        fname=f"stack_mu_sample{sample_idx}.png",
    )

    # rawσ stacked
    rawsig_total = out["raw"][:, 1]
    contribs_rs = [
        out["target"][:, 1],
        out["endo"][:, 1],
        out["exo"][:, 1],
        out["future_cov"][:, 1],
        out["beta"][:, 1],
    ]
    labels_rs = ["target_rawsig", "endo_rawsig", "exo_rawsig", "future_cov_rawsig", "beta_rawsig"]
    save_stackplot(
        out_dir, h, contribs_rs, labels_rs, rawsig_total,
        title=f"Stacked contributions to rawσ (pre-softplus) (sample {sample_idx})",
        fname=f"stack_rawsig_sample{sample_idx}.png",
    )

# ===============================================================================


    H = args.H
    h = np.arange(1, H + 1)

    # μ is additive directly (raw[:,0] == mu)
    mu_sum = (
        out["target"][:, 0]
        + out["endo"][:, 0]
        + out["exo"][:, 0]
        + out["future_cov"][:, 0]
        + out["beta"][:, 0]
    )
    mu_err = np.max(np.abs(mu_sum - out["mu"]))

    # raw σ component is additive; sigma = softplus(rawσ)+eps is NOT additive
    rawsig_sum = (
        out["target"][:, 1]
        + out["endo"][:, 1]
        + out["exo"][:, 1]
        + out["future_cov"][:, 1]
        + out["beta"][:, 1]
    )
    rawsig_err = np.max(np.abs(rawsig_sum - out["raw"][:, 1]))

    print("\n=== Selected test sample ===")
    print("sample_idx:", sample_idx)
    # Get timestamps from dataframe if available
    if "t_timestamp" in sample:
        print("t_end_history:", sample["t_timestamp"])
        print("future_start :", sample["future_timestamps"][0])
        print("future_end   :", sample["future_timestamps"][-1])
    else:
        # Compute from dataframe and center index
        center_idx = ds_test.centers[sample_idx]
        print(f"center_idx (end of history):", center_idx)
        print(f"history range: [{center_idx - cfg.L + 1}, {center_idx}]")
        print(f"forecast range: [{center_idx + 1}, {center_idx + cfg.H}]")
    print(f"reconstruction check: max|mu_sum-mu|={mu_err:.3e}, max|rawsig_sum-rawsig|={rawsig_err:.3e}")

    save_horizon_profile(
        out_dir,
        h,
        {
            "target_mu": out["target"][:, 0],
            "endo_mu": out["endo"][:, 0],
            "exo_mu": out["exo"][:, 0],
            "future_cov_mu": out["future_cov"][:, 0],
            "beta_mu": out["beta"][:, 0],
            "mu_total": out["mu"],
        },
        title="Horizon decomposition for μ (scaled)",
        ylabel="contribution (scaled)",
        fname=f"decomp_mu_sample{sample_idx}.png",
    )

    save_horizon_profile(
        out_dir,
        h,
        {
            "target_rawsig": out["target"][:, 1],
            "endo_rawsig": out["endo"][:, 1],
            "exo_rawsig": out["exo"][:, 1],
            "future_cov_rawsig": out["future_cov"][:, 1],
            "beta_rawsig": out["beta"][:, 1],
            "rawsig_total": out["raw"][:, 1],
        },
        title="Horizon decomposition for raw σ component (pre-softplus, scaled space)",
        ylabel="contribution (raw, scaled)",
        fname=f"decomp_rawsig_sample{sample_idx}.png",
    )

    save_horizon_profile(
        out_dir,
        h,
        {
            "sigma": out["sigma"],
        },
        title="Predicted σ after softplus (scaled space)",
        ylabel="sigma (scaled)",
        fname=f"sigma_sample{sample_idx}.png",
    )

    # ---------
    # 8B) Occlusion maps (lag × horizon)
    # ---------
    print("\nComputing occlusion maps (this does ~2*L forward passes)...")
    dmu_endo, drs_endo, dmu_exo, drs_exo = occlusion_maps(model, device, sample, cfg.target, baseline=0.0)

    # Use absolute deltas for clearer "importance" maps
    save_heatmap(
        out_dir,
        np.abs(dmu_endo),
        title="Occlusion |Δμ| from endo history (lag × horizon)",
        xlabel="horizon step",
        ylabel="lag index (0..L-1)",
        fname=f"occ_abs_dmu_endo_sample{sample_idx}.png",
    )
    save_heatmap(
        out_dir,
        np.abs(dmu_exo),
        title="Occlusion |Δμ| from exo history (lag × horizon)",
        xlabel="horizon step",
        ylabel="lag index (0..L-1)",
        fname=f"occ_abs_dmu_exo_sample{sample_idx}.png",
    )
    save_heatmap(
        out_dir,
        np.abs(drs_endo),
        title="Occlusion |Δrawσ| from endo history (lag × horizon)",
        xlabel="horizon step",
        ylabel="lag index (0..L-1)",
        fname=f"occ_abs_drawsig_endo_sample{sample_idx}.png",
    )
    save_heatmap(
        out_dir,
        np.abs(drs_exo),
        title="Occlusion |Δrawσ| from exo history (lag × horizon)",
        xlabel="horizon step",
        ylabel="lag index (0..L-1)",
        fname=f"occ_abs_drawsig_exo_sample{sample_idx}.png",
    )

    # Quick summary: most influential lags (average over horizon)
    endo_mu_imp = np.mean(np.abs(dmu_endo), axis=1)
    exo_mu_imp  = np.mean(np.abs(dmu_exo), axis=1)
    endo_rs_imp = np.mean(np.abs(drs_endo), axis=1)
    exo_rs_imp  = np.mean(np.abs(drs_exo), axis=1)

    topk = 10
    print("\n=== Top lags by average |Δμ| (endo) ===")
    for i in np.argsort(-endo_mu_imp)[:topk]:
        print(f"lag {i:3d}: {endo_mu_imp[i]:.6f}")
    print("\n=== Top lags by average |Δμ| (exo) ===")
    for i in np.argsort(-exo_mu_imp)[:topk]:
        print(f"lag {i:3d}: {exo_mu_imp[i]:.6f}")

    print("\n=== Top lags by average |Δrawσ| (endo) ===")
    for i in np.argsort(-endo_rs_imp)[:topk]:
        print(f"lag {i:3d}: {endo_rs_imp[i]:.6f}")
    print("\n=== Top lags by average |Δrawσ| (exo) ===")
    for i in np.argsort(-exo_rs_imp)[:topk]:
        print(f"lag {i:3d}: {exo_rs_imp[i]:.6f}")

    print(f"\nSaved plots to: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
