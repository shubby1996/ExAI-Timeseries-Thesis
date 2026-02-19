import argparse
import math
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, json

from step1_3_data_pipeline import (
    TSConfig,
    load_and_prepare,
    chronological_split_indices,
    fit_scalers_on_train,
    apply_scalers,
    WindowDataset,
)

# Import model + nll from your training file (re-use exact definitions)
from train_tsnamlss import TSNAMLSSNormal, normal_nll, collate_tensor_only


'''
We’ll now:

    - load best_tsnamlss.pt
    - run on test set
    - compute:
          - Test NLL (scaled space)
          - MAE/RMSE on point forecast (μ) in original units
          - 80% PI coverage (PICP) + width (MIW) in original units
          - Winkler score (optional but easy)
'''



@torch.no_grad()
def normal_quantile(mu, sigma, q):
    # inverse CDF via torch distributions
    dist = torch.distributions.Normal(mu, sigma)
    q_t = torch.tensor(q, device=mu.device, dtype=mu.dtype)
    return dist.icdf(q_t)

@torch.no_grad()
def evaluate_test(model, loader, device, target, y_mean, y_std, alpha=0.2, out_dir=None, exo_cols=None):
    model.eval()

    all_nll = []
    all_mae = []
    all_rmse = []
    all_picp = []
    all_miw = []
    all_winkler = []

    y_all = []
    mu_parts_all = {"target": [], "endo": [], "exo": [], "future_cov": []}
    rs_parts_all = {"target": [], "endo": [], "exo": [], "future_cov": []}
    mu_parts_per_cov = {}  # per-covariate μ contributions
    rs_parts_per_cov = {}  # per-covariate rawσ contributions
    rawsig_all = []

    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    for batch in loader:
        target_hist = batch["target_hist"].to(device)
        endo_hists = {k: v.to(device) for k, v in batch["endo_hists"].items()}
        exo_hists = {k: v.to(device) for k, v in batch["exo_hists"].items()}
        future_cov = batch["future_cov"].to(device)
        target_future = batch["target_future"].to(device)  # scaled

        out = model(target_hist, endo_hists, exo_hists, future_cov)
        
        mu_s = out["mu"]       # scaled
        sig_s = out["sigma"]   # scaled

        # collect scaled y
        y_all.append(target_future.detach().cpu())

        # Collect stream-level contributions
        mu_parts_all["target"].append(out["contrib_target"][..., 0].detach().cpu())
        mu_parts_all["endo"].append(out["contrib_endo_sum"][..., 0].detach().cpu())
        mu_parts_all["exo"].append(out["contrib_exo_sum"][..., 0].detach().cpu())
        mu_parts_all["future_cov"].append(out["contrib_future_sum"][..., 0].detach().cpu())

        rs_parts_all["target"].append(out["contrib_target"][..., 1].detach().cpu())
        rs_parts_all["endo"].append(out["contrib_endo_sum"][..., 1].detach().cpu())
        rs_parts_all["exo"].append(out["contrib_exo_sum"][..., 1].detach().cpu())
        rs_parts_all["future_cov"].append(out["contrib_future_sum"][..., 1].detach().cpu())

        # Collect per-covariate contributions (μ)
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

        # Collect per-covariate contributions (rawσ)
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

        # total rawsigma (pre-softplus) for internal normalization if we want it
        rawsig_all.append(out["raw"][..., 1].detach().cpu())

        # NLL in scaled space
        nll = normal_nll(mu_s, sig_s, target_future).mean(dim=1)  # per sample
        all_nll.append(nll.cpu().numpy())

        # Convert to original units:
        # y_orig = y_scaled * y_std + y_mean
        mu = mu_s * y_std + y_mean
        sig = sig_s * y_std
        y = target_future * y_std + y_mean

        # Point errors
        mae = (mu - y).abs().mean(dim=1)
        rmse = torch.sqrt(((mu - y) ** 2).mean(dim=1))
        all_mae.append(mae.cpu().numpy())
        all_rmse.append(rmse.cpu().numpy())

        # 80% prediction interval
        lo_s = normal_quantile(mu_s, sig_s, q_lo)
        hi_s = normal_quantile(mu_s, sig_s, q_hi)
        lo = lo_s * y_std + y_mean
        hi = hi_s * y_std + y_mean

        inside = ((y >= lo) & (y <= hi)).float().mean(dim=1)  # per sample coverage across horizon
        miw = (hi - lo).mean(dim=1)
        all_picp.append(inside.cpu().numpy())
        all_miw.append(miw.cpu().numpy())

        # Winkler score (per sample, averaged over horizon)
        # Winkler_alpha = width + (2/alpha)*(lo - y) if y<lo  OR + (2/alpha)*(y - hi) if y>hi
        width = (hi - lo)
        below = (y < lo).float()
        above = (y > hi).float()
        penalty = (2.0 / alpha) * ( (lo - y) * below + (y - hi) * above )
        winkler = (width + penalty).mean(dim=1)
        all_winkler.append(winkler.cpu().numpy())


    # Aggregate stream tensors
    y_all_t = torch.cat(y_all, dim=0)  # (N,H)

    mu_target = torch.cat(mu_parts_all["target"], dim=0)
    mu_endo = torch.cat(mu_parts_all["endo"], dim=0)
    mu_exo  = torch.cat(mu_parts_all["exo"],  dim=0)
    mu_future_cov = torch.cat(mu_parts_all["future_cov"], dim=0)

    rs_target = torch.cat(rs_parts_all["target"], dim=0)
    rs_endo = torch.cat(rs_parts_all["endo"], dim=0)
    rs_exo  = torch.cat(rs_parts_all["exo"],  dim=0)
    rs_future_cov = torch.cat(rs_parts_all["future_cov"], dim=0)

    rawsig_t = torch.cat(rawsig_all, dim=0)  # (N,H)

    # Aggregate per-covariate tensors
    mu_per_cov = {}
    rs_per_cov = {}
    for key in mu_parts_per_cov.keys():
        mu_per_cov[key] = torch.cat(mu_parts_per_cov[key], dim=0)
        rs_per_cov[key] = torch.cat(rs_parts_per_cov[key], dim=0)

    # A) Stream-level effects/importances
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

    mu_eff_h = {
        "target": effect_by_horizon(mu_target, y_all_t),
        "endo": effect_by_horizon(mu_endo, y_all_t),
        "exo":  effect_by_horizon(mu_exo,  y_all_t),
        "future_cov": effect_by_horizon(mu_future_cov, y_all_t),
    }
    mu_imp_h = normalize_importances_by_horizon(mu_eff_h)

    # Per-covariate μ effects by horizon
    mu_eff_h_per_cov = {key: effect_by_horizon(mu_per_cov[key], y_all_t) for key in mu_per_cov.keys()}
    mu_imp_h_per_cov = normalize_importances_by_horizon(mu_eff_h_per_cov)

    # B) raw-sigma effects/importances
    rs_effects_y = {
        "target": effect_global(rs_target, y_all_t),
        "endo": effect_global(rs_endo, y_all_t),
        "exo":  effect_global(rs_exo,  y_all_t),
        "future_cov": effect_global(rs_future_cov, y_all_t),
    }
    rs_imps_y = normalize_importances(rs_effects_y)

    # Per-covariate rawσ effects (normalized by y)
    rs_effects_y_per_cov = {key: effect_global(rs_per_cov[key], y_all_t) for key in rs_per_cov.keys()}
    rs_imps_y_per_cov = normalize_importances(rs_effects_y_per_cov)

    rs_eff_h_y = {
        "target": effect_by_horizon(rs_target, y_all_t),
        "endo": effect_by_horizon(rs_endo, y_all_t),
        "exo":  effect_by_horizon(rs_exo,  y_all_t),
        "future_cov": effect_by_horizon(rs_future_cov, y_all_t),
    }
    rs_imp_h_y = normalize_importances_by_horizon(rs_eff_h_y)

    # Per-covariate rawσ effects by horizon (normalized by y)
    rs_eff_h_y_per_cov = {key: effect_by_horizon(rs_per_cov[key], y_all_t) for key in rs_per_cov.keys()}
    rs_imp_h_y_per_cov = normalize_importances_by_horizon(rs_eff_h_y_per_cov)

    # Per-covariate rawσ effects (normalized by rawσ)
    rs_effects_rs = {
        "target": effect_global(rs_target, rawsig_t),
        "endo": effect_global(rs_endo, rawsig_t),
        "exo":  effect_global(rs_exo,  rawsig_t),
        "future_cov": effect_global(rs_future_cov, rawsig_t),
    }
    rs_imps_rs = normalize_importances(rs_effects_rs)

    # Per-covariate rawσ effects (normalized by rawσ)
    rs_effects_rs_per_cov = {key: effect_global(rs_per_cov[key], rawsig_t) for key in rs_per_cov.keys()}
    rs_imps_rs_per_cov = normalize_importances(rs_effects_rs_per_cov)

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

    effects_payload = {
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

    print("\n=== EFFECT / IMPORTANCE (stream-level) ===")
    print("MU effects (norm by y):", mu_effects)
    print("MU importances:", mu_imps)
    print("RAWSIG effects (norm by y):", rs_effects_y)
    print("RAWSIG importances (norm by y):", rs_imps_y)
    print("RAWSIG effects (norm by rawsig):", rs_effects_rs)
    print("RAWSIG importances (norm by rawsig):", rs_imps_rs)

    print("\n=== EFFECT / IMPORTANCE (per-covariate, μ) ===")
    print("MU importances per covariate:", mu_imps_per_cov)

    print("\n=== EFFECT / IMPORTANCE (per-covariate, rawσ norm by y) ===")
    print("RAWSIG importances per covariate (norm by y):", rs_imps_y_per_cov)

    print("\n=== EFFECT / IMPORTANCE (per-covariate, rawσ norm by rawσ) ===")
    print("RAWSIG importances per covariate (norm by rawsig):", rs_imps_rs_per_cov)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "effect_importance_stream_level.json"), "w") as f:
            json.dump(effects_payload, f, indent=2)

        np.save(os.path.join(out_dir, "mu_importance_by_horizon_norm_y.npy"), mu_imp_h)
        np.save(os.path.join(out_dir, "mu_importance_by_horizon_per_cov_norm_y.npy"), mu_imp_h_per_cov)
        np.save(os.path.join(out_dir, "rawsig_importance_by_horizon_norm_y.npy"), rs_imp_h_y)
        np.save(os.path.join(out_dir, "rawsig_importance_by_horizon_per_cov_norm_y.npy"), rs_imp_h_y_per_cov)
        np.save(os.path.join(out_dir, "rawsig_importance_by_horizon_norm_rawsig.npy"), rs_imp_h_rs)
        np.save(os.path.join(out_dir, "rawsig_importance_by_horizon_per_cov_norm_rawsig.npy"), rs_imp_h_rs_per_cov)


    # Aggregate
    nll = float(np.mean(np.concatenate(all_nll)))
    mae = float(np.mean(np.concatenate(all_mae)))
    rmse = float(np.mean(np.concatenate(all_rmse)))
    picp = float(np.mean(np.concatenate(all_picp)))
    miw = float(np.mean(np.concatenate(all_miw)))
    winkler = float(np.mean(np.concatenate(all_winkler)))

    return {
        "test_nll_scaled": nll,
        "mae_orig": mae,
        "rmse_orig": rmse,
        "picp_(1-alpha)": picp,
        "miw_orig": miw,
        "winkler_orig": winkler,
    }


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
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--alpha", type=float, default=0.2)  # 0.2 -> 80% PI
    ap.add_argument("--out_dir", type=str, default="eval_out")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load checkpoint to detect model architecture FIRST
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Detect model architecture from checkpoint state_dict
    has_exo_nets = any("exo_nets" in key for key in ckpt["model_state"].keys())
    
    # Initialize config
    cfg = TSConfig(L=args.L, H=args.H)
    
    # Restore cfg from checkpoint if available
    if "cfg" in ckpt:
        saved_cfg = ckpt["cfg"]
        if "target" in saved_cfg:
            cfg.target = saved_cfg["target"]
        if "endo_cols" in saved_cfg:
            cfg.endo_cols = saved_cfg["endo_cols"]
        if "exo_cols" in saved_cfg:
            cfg.exo_cols = saved_cfg["exo_cols"]
        if "future_cov_cols" in saved_cfg:
            cfg.future_cov_cols = saved_cfg["future_cov_cols"]
    
    if has_exo_nets:
        # New architecture: detect any remaining exo_cols from checkpoint if not in cfg
        exo_cols = []
        for key in ckpt["model_state"].keys():
            if key.startswith("exo_nets."):
                exo_col = key.split(".")[1]
                if exo_col not in exo_cols:
                    exo_cols.append(exo_col)
        if exo_cols and not cfg.exo_cols:
            cfg.exo_cols = exo_cols
        
        print(f"✓ Detected configurable model:")
        print(f"  target: {cfg.target}")
        print(f"  endo_cols: {cfg.endo_cols}")
        print(f"  exo_cols: {cfg.exo_cols}")
        print(f"  future_cov_cols: {cfg.future_cov_cols}")
    else:
        # OLD architecture not supported anymore
        raise ValueError("Legacy architecture no longer supported. Please use new configurable architecture with target.")
    
    # Load and prepare
    df_raw = load_and_prepare(Path(args.csv_path), cfg)
    n = len(df_raw)
    train_rng, val_rng, test_rng = chronological_split_indices(n, cfg.train_frac, cfg.val_frac)

    # Fit scalers on train and apply
    scalers = fit_scalers_on_train(df_raw, cfg, train_rng)
    df = apply_scalers(df_raw, cfg, scalers)

    # Test dataset with proper collate function
    ds_test = WindowDataset(df, cfg, test_rng)
    collate_fn = partial(collate_tensor_only, target=cfg.target, endo_cols=cfg.endo_cols, exo_cols=cfg.exo_cols)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, drop_last=False, collate_fn=collate_fn)

    # Load model with correct architecture
    model = TSNAMLSSNormal(L=args.L, H=args.H, target=cfg.target, endo_cols=cfg.endo_cols, 
                           exo_cols=cfg.exo_cols, future_cov_cols=cfg.future_cov_cols).to(device)
    model.load_state_dict(ckpt["model_state"])

    # y scaling constants (from fitted scaler on target)
    y_mean = float(scalers[cfg.target].mean_[0])
    y_std = float(np.sqrt(scalers[cfg.target].var_[0]))

    metrics = evaluate_test(model, test_loader, device, target=cfg.target, y_mean=y_mean, 
                           y_std=y_std, alpha=args.alpha, out_dir=args.out_dir, exo_cols=cfg.exo_cols)

    print("saved eval artifacts to:", os.path.abspath(args.out_dir))

    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("test samples:", len(ds_test))


if __name__ == "__main__":
    main()
