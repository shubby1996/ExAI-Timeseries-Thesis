import argparse
import math
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from step1_3_data_pipeline import (
    TSConfig,
    load_and_prepare,
    chronological_split_indices,
    fit_scalers_on_train,
    apply_scalers,
    WindowDataset,
)


'''

It trains for epochs, evaluates on validation each epoch, and early-stops. It uses the exact same dataset + model you already built.

python train_tsnamlss.py --csv_path nordbyen_features_engineered.csv --L 168 --H 24 --batch_size 256 --epochs 30 --lr 1e-3 --device cpu

On your HPC, you can also try:

--num_workers 2 or 4

--batch_size 512 (if memory allows)

“Why is NLL negative?”

Totally fine. NLL is -log p(y|μ,σ). For continuous distributions, the density p(.) can exceed 1 when σ is small and the prediction is very accurate, so log p can be positive → NLL negative. No issue.

'''

torch.manual_seed(7)


# ----------------------------
# Likelihood: Normal NLL
# ----------------------------
def normal_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    return 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * (z * z)  # (B,H)


# ----------------------------
# Model parts (same as before)
# ----------------------------
class WindowMLP(nn.Module):
    def __init__(self, L: int, H: int, K: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.H, self.K = H, K
        self.fc1 = nn.Linear(L, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, H * K)
        self.dropout = nn.Dropout(dropout)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(w))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out.view(-1, self.H, self.K)


class FutureCovMLP(nn.Module):
    def __init__(self, D: int, K: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, K),
        )

    def forward(self, future_cov: torch.Tensor) -> torch.Tensor:
        B, H, D = future_cov.shape
        x = future_cov.reshape(B * H, D)
        y = self.net(x)
        return y.view(B, H, -1)


class TSNAMLSSNormal(nn.Module):
    def __init__(self, L: int, H: int, target: str = "heat_consumption", endo_cols: list = None, 
                 exo_cols: list = None, future_cov_cols: list = None, hidden_window: int = 64, 
                 hidden_future_cov: int = 32, dropout: float = 0.1):
        super().__init__()
        self.L, self.H, self.K = L, H, 2
        self.target = target  # Singular target to forecast
        self.endo_cols = endo_cols if endo_cols is not None else []  # Optional AR features
        self.exo_cols = exo_cols if exo_cols is not None else ["temp"]
        self.future_cov_cols = future_cov_cols if future_cov_cols is not None else ["hour_sin", "hour_cos"]
        
        # Target stream (always present)
        self.target_net = WindowMLP(L, H, self.K, hidden=hidden_window, dropout=dropout)
        
        # Endogenous (AR) streams - one per endo feature
        self.endo_nets = nn.ModuleDict()
        for endo_col in self.endo_cols:
            self.endo_nets[endo_col] = WindowMLP(L, H, self.K, hidden=hidden_window, dropout=dropout)
        
        # Create a separate WindowMLP for each exogenous column
        if len(self.exo_cols) > 0:
            self.exo_nets = nn.ModuleDict()
            for exo_col in self.exo_cols:
                self.exo_nets[exo_col] = WindowMLP(L, H, self.K, hidden=hidden_window, dropout=dropout)
        
        # Future covariate streams - one per future feature (each sees (B,H,1))
        self.future_cov_nets = nn.ModuleDict()
        for cov_col in self.future_cov_cols:
            self.future_cov_nets[cov_col] = FutureCovMLP(D=1, K=self.K, hidden=hidden_future_cov, dropout=0.0)
        self.beta = nn.Parameter(torch.zeros(H, self.K))

    def forward(self, target_hist: torch.Tensor, endo_hists: Dict[str, torch.Tensor], 
                exo_hists: Dict[str, torch.Tensor], future_cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            target_hist: (B, L) target variable history
            endo_hists: dict of {endo_col: (B, L)} for optional AR features
            exo_hists: dict of {exo_col: (B, L)} exogenous histories
            future_cov: (B, H, num_future_cov) future covariates
        """
        B = target_hist.shape[0]
        device = target_hist.device
        
        # Target contribution
        target_contrib = self.target_net(target_hist)  # (B,H,K)
        
        # Endogenous (AR) contributions
        endo_total = None
        endo_contribs = {}
        for endo_col in self.endo_cols:
            endo_hist = endo_hists.get(endo_col)
            if endo_hist is None:
                endo_hist = torch.zeros_like(target_hist)
            endo_contrib = self.endo_nets[endo_col](endo_hist)  # (B,H,K)
            endo_contribs[endo_col] = endo_contrib
            endo_total = endo_contrib if endo_total is None else endo_total + endo_contrib
        if endo_total is None:
            endo_total = torch.zeros(B, self.H, self.K, device=device)
        
        # Sum contributions from all exogenous columns
        exo_total = None
        exo_contribs = {}
        if len(self.exo_cols) > 0:
            for exo_col in self.exo_cols:
                exo_contrib = self.exo_nets[exo_col](exo_hists[exo_col])  # (B,H,K)
                exo_contribs[exo_col] = exo_contrib
                if exo_total is None:
                    exo_total = exo_contrib
                else:
                    exo_total = exo_total + exo_contrib
        else:
            # No exogenous features
            exo_total = torch.zeros(B, self.H, self.K, device=device)
        
        # Future covariate contributions
        future_total = None
        future_contribs = {}
        for idx, cov_col in enumerate(self.future_cov_cols):
            cov_series = future_cov[:, :, idx:idx + 1]  # (B,H,1)
            cov_contrib = self.future_cov_nets[cov_col](cov_series)  # (B,H,K)
            future_contribs[cov_col] = cov_contrib
            future_total = cov_contrib if future_total is None else future_total + cov_contrib
        if future_total is None:
            future_total = torch.zeros(B, self.H, self.K, device=device)
        
        raw = target_contrib + endo_total + exo_total + future_total + self.beta.unsqueeze(0)

        mu = raw[:, :, 0]
        sigma = F.softplus(raw[:, :, 1]) + 1e-5

        ret = {
            "mu": mu,
            "sigma": sigma,
            "raw": raw,
            "contrib_target": target_contrib,
            "contrib_endo_sum": endo_total,
            "contrib_exo_sum": exo_total,
            "contrib_future_sum": future_total,
            "beta": self.beta.unsqueeze(0).expand_as(raw),
        }
        
        # Add individual endogenous contributions
        for endo_col in self.endo_cols:
            ret[f"contrib_endo_{endo_col}"] = endo_contribs[endo_col]
        
        # Add individual exogenous contributions
        for exo_col in self.exo_cols:
            ret[f"contrib_{exo_col}"] = exo_contribs[exo_col]
        
        # Add individual future cov contributions
        for cov_col in self.future_cov_cols:
            ret[f"contrib_future_{cov_col}"] = future_contribs[cov_col]
        
        return ret


# ----------------------------
# Collate (extract tensors from new dataset format)
# ----------------------------
def collate_tensor_only(batch, target: str, endo_cols, exo_cols):
    """
    Collate function for new dataset format.
    Dataset returns: target_hist, target_future, endo_col_hist (if any), exo_col, future_cov
    """
    out = {}
    
    # Stack target histories and futures
    out["target_hist"] = torch.stack([b["target_hist"] for b in batch], dim=0)
    out["target_future"] = torch.stack([b["target_future"] for b in batch], dim=0)
    
    # Stack endogenous (AR) histories if any
    endo_hists = {}
    for endo_col in endo_cols:
        endo_hists[endo_col] = torch.stack([b[f"{endo_col}_hist"] for b in batch], dim=0)
    out["endo_hists"] = endo_hists
    
    # Stack exogenous histories into dict
    exo_hists = {}
    for exo_col in exo_cols:
        exo_hists[exo_col] = torch.stack([b[exo_col] for b in batch], dim=0)
    out["exo_hists"] = exo_hists
    
    # Stack future covariates
    out["future_cov"] = torch.stack([b["future_cov"] for b in batch], dim=0)
    
    return out


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        target_hist = batch["target_hist"].to(device)
        endo_hists = {k: v.to(device) for k, v in batch["endo_hists"].items()}
        exo_hists = {k: v.to(device) for k, v in batch["exo_hists"].items()}
        future_cov = batch["future_cov"].to(device)
        target_future = batch["target_future"].to(device)

        out = model(target_hist, endo_hists, exo_hists, future_cov)
        nll = normal_nll(out["mu"], out["sigma"], target_future)
        losses.append(nll.mean().item())

    return float(np.mean(losses)) if losses else float("inf")


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        target_hist = batch["target_hist"].to(device)
        endo_hists = {k: v.to(device) for k, v in batch["endo_hists"].items()}
        exo_hists = {k: v.to(device) for k, v in batch["exo_hists"].items()}
        future_cov = batch["future_cov"].to(device)
        target_future = batch["target_future"].to(device)

        out = model(target_hist, endo_hists, exo_hists, future_cov)
        nll = normal_nll(out["mu"], out["sigma"], target_future)
        loss = nll.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # mild safety
        opt.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--L", type=int, default=168)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--patience", "--early_stopping_patience", type=int, default=5, dest="patience")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_path", type=str, default="best_tsnamlss.pt")
    args = ap.parse_args()

    cfg = TSConfig(L=args.L, H=args.H)

    # Step 1–3: load, split, scale, dataset
    df_raw = load_and_prepare(Path(args.csv_path), cfg)
    n = len(df_raw)
    train_rng, val_rng, test_rng = chronological_split_indices(n, cfg.train_frac, cfg.val_frac)

    scalers = fit_scalers_on_train(df_raw, cfg, train_rng)
    df = apply_scalers(df_raw, cfg, scalers)

    ds_train = WindowDataset(df, cfg, train_rng)
    ds_val   = WindowDataset(df, cfg, val_rng)

    # Create collate function with target, endo and exo columns
    collate_fn = partial(collate_tensor_only, target=cfg.target, endo_cols=cfg.endo_cols, exo_cols=cfg.exo_cols)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = TSNAMLSSNormal(L=args.L, H=args.H, target=cfg.target, endo_cols=cfg.endo_cols, 
                           exo_cols=cfg.exo_cols, future_cov_cols=cfg.future_cov_cols, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    bad_epochs = 0

    print(f"train samples={len(ds_train)} | val samples={len(ds_val)}")
    print(f"target={cfg.target}")
    print(f"endogenous (AR) columns={cfg.endo_cols}")
    print(f"exogenous columns={cfg.exo_cols}")
    print(f"future covariates={cfg.future_cov_cols}")
    print(f"device={device} | batch_size={args.batch_size} | lr={args.lr} | dropout={args.dropout}")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device)
        va = evaluate(model, val_loader, device)

        print(f"epoch {epoch:02d}: train_nll={tr:.6f} | val_nll={va:.6f}")

        if va < best_val - 1e-5:
            best_val = va
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": vars(cfg),
                    "scalers": {k: {"mean": float(v.mean_[0]), "std": float(np.sqrt(v.var_[0]))} for k, v in scalers.items()},
                },
                args.save_path,
            )
            print(f"  saved best -> {args.save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (patience={args.patience}). Best val_nll={best_val:.6f}")
                break


if __name__ == "__main__":
    main()
