"""
mdn_hyperparameter_search.py
=============================
Systematic hyperparameter search for the MDN model.

Extends the existing architectural ablation (Exp1–Exp5 in ablation_results.json)
with a proper grid search over continuous hyperparameters that were previously
left at default values.  Preserves ALL existing results and appends new ones.

Search dimensions
-----------------
  Phase 1 — Learning rate             : {3e-4, 1e-3, 3e-3}
  Phase 2 — Weight decay              : {0, 1e-5, 1e-4, 1e-3}
  Phase 3 — Batch size                : {128, 256, 512, 1024}
  Phase 4 — Dropout rate              : {uniform 0.1, uniform 0.3, uniform 0.5,
                                         tapered [0.4,0.3,0.1]}
  Phase 5 — Hidden dimensions         : {[128,64], [256,128,64], [512,256,128],
                                         [256,128,64,32]}
  Phase 6 — Mixture components K      : {2, 3, 5, 8, 10}
  Phase 7 — SIGMA_MIN                 : {1e-6, 1e-4, 1e-3, 1e-2}
  Phase 8 — Grad-clip max-norm        : {0.5, 1.0, 5.0, None}
  Phase 9 — Scheduler patience/factor : {(3,0.5), (5,0.5), (5,0.3), (10,0.5)}
  Phase 10 — Early-stop patience      : {8, 15, 25}
  Phase 11 — Best combo               : top params from phases 1-10

Each experiment trains on 3 rolling folds (same as existing ablation) to keep
runtime manageable while maintaining statistical validity.

Output
------
Updates results/mdn_ablation/ablation_results.json in-place (appends).

Usage
-----
    cd <project_root>
    python src/mdn_hyperparameter_search.py            # run all phases
    python src/mdn_hyperparameter_search.py --phase 1  # run only phase 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# -- project imports (no modifications to existing code) -------------------
from common.constants import (
    PROJECT_ROOT, DATA_PATH, DATE_COL, TICKER_COL, TARGET_UP, TARGET_DN,
    RANDOM_SEED, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS, PURGE_GAP_DAYS,
    EXCLUDE_COLS_NN as EXCLUDE_COLS,
    VAL_FRAC, WINSORIZE_PCT,
)
from common.data import (
    build_updown_targets, check_required_columns, build_rolling_windows,
    robust_standardize, inverse_standardize,
)
from train_mdn_rolling import mdn_nll_loss, mdn_mean, mdn_std

warnings.filterwarnings("ignore")

# -- paths -----------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "mdn_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "ablation_results.json"

# -- search defaults (same as production MDN) ------------------------------
DEVICE     = torch.device("cpu")   # CPU for reproducibility across machines
MAX_FOLDS  = 3
MAX_EPOCHS = 80                    # reduced vs full training (200) for search
SEED       = RANDOM_SEED

# -- baseline config (mirrors common/constants.py) -------------------------
BASELINE_CFG = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    hidden_dims=[256, 128, 64],
    dropout=[0.3, 0.3, 0.2],
    n_components=5,
    sigma_min=1e-4,
    grad_clip=1.0,
    sched_patience=5,
    sched_factor=0.5,
    es_patience=15,
    max_epochs=MAX_EPOCHS,
)


# ===================================================================
# CONFIGURABLE MDN (accepts hyperparams as arguments)
# ===================================================================
class MDNConfigurable(nn.Module):
    """MDN with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: List[float],
        n_components: int,
        sigma_min: float,
    ):
        super().__init__()
        self.K = n_components
        self.sigma_min = sigma_min

        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h, d in zip(hidden_dims, dropout):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.pi_head    = nn.Linear(prev, n_components)
        self.mu_head    = nn.Linear(prev, n_components)
        self.sigma_head = nn.Linear(prev, n_components)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        pi    = torch.softmax(self.pi_head(h), dim=-1)
        mu    = self.mu_head(h)
        sigma = torch.exp(self.sigma_head(h)) + self.sigma_min
        return pi, mu, sigma


# ===================================================================
# DATA PREPARATION
# ===================================================================
def load_and_prepare() -> Tuple[pd.DataFrame, List[str], list]:
    """Load dataset, build targets, one-hot encode, build rolling windows."""
    raw = pd.read_csv(DATA_PATH)
    check_required_columns(raw)

    df = raw.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])

    saved_ticker = df[TICKER_COL].copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_COLS]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    df[TICKER_COL] = saved_ticker.values

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    windows = build_rolling_windows(
        all_dates=df[DATE_COL],
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        purge_gap_days=PURGE_GAP_DAYS,
    )
    return df, feature_cols, windows


# ===================================================================
# TRAIN + EVALUATE ONE TARGET WITH GIVEN CONFIG
# ===================================================================
def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
) -> Tuple[float, float, float]:
    """
    Train MDN with the given config dict on one target.
    Returns (rmse, nll, mean_predicted_std).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_proc, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_train)

    n = len(X_train)
    n_val = max(1, int(n * VAL_FRAC))
    n_tr = n - n_val

    Xtr = torch.tensor(X_train[:n_tr], dtype=torch.float32)
    ytr = torch.tensor(y_proc[:n_tr],  dtype=torch.float32)
    Xva = torch.tensor(X_train[n_tr:], dtype=torch.float32)
    yva = torch.tensor(y_proc[n_tr:],  dtype=torch.float32)
    Xte = torch.tensor(X_test,         dtype=torch.float32)

    y_test_z = (y_test - y_center) / y_scale
    yte = torch.tensor(y_test_z, dtype=torch.float32)

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = MDNConfigurable(
        input_dim=X_train.shape[1],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        n_components=cfg["n_components"],
        sigma_min=cfg["sigma_min"],
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min",
        factor=cfg["sched_factor"],
        patience=cfg["sched_patience"],
        min_lr=1e-6,
    )

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_nll_loss(pi, mu, sigma, yb)
            loss.backward()
            if cfg["grad_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        model.eval()
        with torch.no_grad():
            pi, mu, sigma = model(Xva)
            vl = mdn_nll_loss(pi, mu, sigma, yva).item()
        if math.isnan(vl) or math.isinf(vl):
            continue
        sched.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["es_patience"]:
                break

    if best_state is None:
        return float("nan"), float("nan"), float("nan")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(Xte)
        pred_z = mdn_mean(pi, mu).numpy()
        std_z  = mdn_std(pi, mu, sigma).numpy()
        test_nll = mdn_nll_loss(pi, mu, sigma, yte).item()

    pred = inverse_standardize(pred_z, y_center, y_scale, clip_lo, clip_hi)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mean_std = float(np.mean(std_z)) * y_scale
    return rmse, test_nll, mean_std


# ===================================================================
# RUN A SINGLE EXPERIMENT ACROSS FOLDS
# ===================================================================
def _encode_and_scale(X_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.values.astype(np.float32))
    Xte = scaler.transform(X_test.values.astype(np.float32))
    return np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0)


def run_experiment(
    name: str,
    cfg: dict,
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: list,
) -> dict:
    """Run one HP config across MAX_FOLDS rolling folds."""
    up_imps, dn_imps = [], []
    up_nlls, dn_nlls = [], []
    up_stds, dn_stds = [], []
    per_fold_imp = []

    for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(windows[:MAX_FOLDS], 1):
        train_mask = (df[DATE_COL] >= tr_s) & (df[DATE_COL] <= tr_e)
        test_mask  = (df[DATE_COL] >= te_s) & (df[DATE_COL] <= te_e)
        train_df = df.loc[train_mask].dropna(
            subset=feature_cols + [TARGET_UP, TARGET_DN])
        test_df = df.loc[test_mask].dropna(
            subset=feature_cols + [TARGET_UP, TARGET_DN])

        if train_df.empty or test_df.empty:
            continue

        X_tr_raw, X_te_raw = train_df[feature_cols], test_df[feature_cols]
        X_tr_raw, X_te_raw = X_tr_raw.align(
            X_te_raw, join="left", axis=1, fill_value=0)
        Xtr, Xte = _encode_and_scale(X_tr_raw, X_te_raw)

        fold_imps = []
        for tgt, label in [(TARGET_UP, "up"), (TARGET_DN, "dn")]:
            y_tr = train_df[tgt].values.astype(np.float32)
            y_te = test_df[tgt].values.astype(np.float32)

            try:
                rmse, nll, mean_std = train_and_eval(Xtr, y_tr, Xte, y_te, cfg)
            except Exception as e:
                print(f"    Fold {fold_id} {label} FAILED: {e}")
                continue

            naive_rmse = float(np.sqrt(mean_squared_error(
                y_te, np.zeros_like(y_te))))
            imp = (1 - rmse / naive_rmse) * 100

            if label == "up":
                up_imps.append(imp)
                up_nlls.append(nll)
                up_stds.append(mean_std)
            else:
                dn_imps.append(imp)
                dn_nlls.append(nll)
                dn_stds.append(mean_std)
            fold_imps.append(imp)

        if len(fold_imps) == 2:
            avg_f = (fold_imps[0] + fold_imps[1]) / 2
            per_fold_imp.append(round(avg_f, 2))
            print(f"    Fold {fold_id}: up={fold_imps[0]:+.1f}%  "
                  f"dn={fold_imps[1]:+.1f}%  avg={avg_f:+.1f}%")

    if not up_imps or not dn_imps:
        return {"name": name, "n_folds": 0, "avg_imp": float("nan")}

    avg_imp = float(np.mean(
        [(u + d) / 2 for u, d in zip(up_imps, dn_imps)]))

    return {
        "name": name,
        "n_folds": len(per_fold_imp),
        "avg_up_imp": float(np.mean(up_imps)),
        "avg_dn_imp": float(np.mean(dn_imps)),
        "avg_imp": avg_imp,
        "avg_up_nll": float(np.mean(up_nlls)),
        "avg_dn_nll": float(np.mean(dn_nlls)),
        "avg_up_std": float(np.mean(up_stds)),
        "avg_dn_std": float(np.mean(dn_stds)),
        "per_fold_imp": per_fold_imp,
        "config": {
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "batch_size": cfg["batch_size"],
            "hidden_dims": cfg["hidden_dims"],
            "dropout": cfg["dropout"],
            "n_components": cfg["n_components"],
            "sigma_min": cfg["sigma_min"],
            "grad_clip": cfg["grad_clip"],
            "sched_patience": cfg["sched_patience"],
            "sched_factor": cfg["sched_factor"],
            "es_patience": cfg["es_patience"],
            "max_epochs": cfg["max_epochs"],
        },
    }


# ===================================================================
# PERSISTENCE: load existing + append + save
# ===================================================================
def load_existing() -> List[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(all_results: List[dict]) -> None:
    # handle NaN/Infinity for JSON serialization
    def _clean(obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"\n  Results saved -> {RESULTS_PATH}")


def already_ran(existing: List[dict], name: str) -> bool:
    return any(r["name"] == name for r in existing)


# ===================================================================
# HELPER: make config by overriding baseline defaults
# ===================================================================
def make_cfg(**overrides) -> dict:
    cfg = deepcopy(BASELINE_CFG)
    # ensure dropout list length matches hidden_dims
    if "hidden_dims" in overrides and "dropout" not in overrides:
        n = len(overrides["hidden_dims"])
        # taper dropout: start from 0.3, decrease
        base_drop = cfg["dropout"]
        if n <= len(base_drop):
            overrides["dropout"] = base_drop[:n]
        else:
            overrides["dropout"] = base_drop + [base_drop[-1]] * (n - len(base_drop))
    cfg.update(overrides)
    return cfg


# ===================================================================
# PHASE DEFINITIONS
# ===================================================================
def phase_1_learning_rate() -> List[Tuple[str, dict]]:
    """Grid search over learning rate."""
    return [
        (f"HP-LR: lr={lr}", make_cfg(lr=lr))
        for lr in [3e-4, 1e-3, 3e-3]
    ]


def phase_2_weight_decay() -> List[Tuple[str, dict]]:
    """Grid search over weight decay."""
    return [
        (f"HP-WD: wd={wd}", make_cfg(weight_decay=wd))
        for wd in [0, 1e-5, 1e-4, 1e-3]
    ]


def phase_3_batch_size() -> List[Tuple[str, dict]]:
    """Grid search over batch size."""
    return [
        (f"HP-BS: bs={bs}", make_cfg(batch_size=bs))
        for bs in [128, 256, 512, 1024]
    ]


def phase_4_dropout() -> List[Tuple[str, dict]]:
    """Grid search over dropout patterns."""
    configs = [
        ("HP-Drop: uniform=0.1", [0.1, 0.1, 0.1]),
        ("HP-Drop: uniform=0.3", [0.3, 0.3, 0.3]),
        ("HP-Drop: uniform=0.5", [0.5, 0.5, 0.5]),
        ("HP-Drop: taper=[0.4,0.3,0.1]", [0.4, 0.3, 0.1]),
        ("HP-Drop: taper=[0.5,0.3,0.1]", [0.5, 0.3, 0.1]),
    ]
    return [(name, make_cfg(dropout=d)) for name, d in configs]


def phase_5_hidden_dims() -> List[Tuple[str, dict]]:
    """Grid search over network architecture."""
    configs = [
        ("HP-Arch: [128,64]",       [128, 64]),
        ("HP-Arch: [256,128]",      [256, 128]),
        ("HP-Arch: [256,128,64]",   [256, 128, 64]),
        ("HP-Arch: [512,256,128]",  [512, 256, 128]),
        ("HP-Arch: [256,128,64,32]", [256, 128, 64, 32]),
        ("HP-Arch: [512,256,128,64]", [512, 256, 128, 64]),
    ]
    return [(name, make_cfg(hidden_dims=h)) for name, h in configs]


def phase_6_components() -> List[Tuple[str, dict]]:
    """Grid search over number of mixture components K."""
    return [
        (f"HP-K: K={k}", make_cfg(n_components=k))
        for k in [2, 3, 5, 8, 10]
    ]


def phase_7_sigma_min() -> List[Tuple[str, dict]]:
    """Grid search over sigma floor."""
    return [
        (f"HP-SigmaMin: {s}", make_cfg(sigma_min=s))
        for s in [1e-6, 1e-4, 1e-3, 1e-2]
    ]


def phase_8_grad_clip() -> List[Tuple[str, dict]]:
    """Grid search over gradient clipping."""
    return [
        (f"HP-GradClip: {g}", make_cfg(grad_clip=g))
        for g in [0.5, 1.0, 5.0, None]
    ]


def phase_9_scheduler() -> List[Tuple[str, dict]]:
    """Grid search over LR scheduler settings."""
    configs = [
        ("HP-Sched: pat=3,fac=0.5",  3, 0.5),
        ("HP-Sched: pat=5,fac=0.5",  5, 0.5),
        ("HP-Sched: pat=5,fac=0.3",  5, 0.3),
        ("HP-Sched: pat=10,fac=0.5", 10, 0.5),
    ]
    return [
        (name, make_cfg(sched_patience=p, sched_factor=f))
        for name, p, f in configs
    ]


def phase_10_es_patience() -> List[Tuple[str, dict]]:
    """Grid search over early stopping patience."""
    return [
        (f"HP-ES: patience={p}", make_cfg(es_patience=p))
        for p in [8, 15, 25]
    ]


def phase_11_best_combo(existing: List[dict]) -> List[Tuple[str, dict]]:
    """
    Combine the best value from each HP dimension into candidate configs.
    Reads results from phases 1-10 to identify winners.
    """
    phase_prefixes = {
        "lr":             "HP-LR:",
        "weight_decay":   "HP-WD:",
        "batch_size":     "HP-BS:",
        "dropout":        "HP-Drop:",
        "hidden_dims":    "HP-Arch:",
        "n_components":   "HP-K:",
        "sigma_min":      "HP-SigmaMin:",
        "grad_clip":      "HP-GradClip:",
        "sched":          "HP-Sched:",
        "es_patience":    "HP-ES:",
    }

    best_per_dim: Dict[str, dict] = {}
    for dim_key, prefix in phase_prefixes.items():
        candidates = [
            r for r in existing
            if r.get("name", "").startswith(prefix) and r.get("avg_imp") is not None
        ]
        if candidates:
            winner = max(candidates, key=lambda r: r["avg_imp"])
            best_per_dim[dim_key] = winner.get("config", {})

    if not best_per_dim:
        print("    No phase 1-10 results found; skipping phase 11.")
        return []

    # Build combined config from winners
    combo = deepcopy(BASELINE_CFG)
    for dim_key, wcfg in best_per_dim.items():
        if dim_key == "sched":
            combo["sched_patience"] = wcfg.get("sched_patience", combo["sched_patience"])
            combo["sched_factor"]   = wcfg.get("sched_factor", combo["sched_factor"])
        elif dim_key in wcfg:
            combo[dim_key] = wcfg[dim_key]

    combos = [("HP-BestCombo: all-winners", combo)]

    # Also try top-2 from the most impactful dimensions (LR, arch, K)
    for dim_key, prefix in [("lr", "HP-LR:"), ("hidden_dims", "HP-Arch:"),
                            ("n_components", "HP-K:")]:
        candidates = [
            r for r in existing
            if r.get("name", "").startswith(prefix) and r.get("avg_imp") is not None
        ]
        if len(candidates) >= 2:
            sorted_cands = sorted(candidates, key=lambda r: r["avg_imp"],
                                  reverse=True)
            runner_up = sorted_cands[1].get("config", {})
            alt = deepcopy(combo)
            if dim_key in runner_up:
                alt[dim_key] = runner_up[dim_key]
            alt_name = (f"HP-BestCombo: swap {dim_key}="
                        f"{runner_up.get(dim_key, '?')}")
            combos.append((alt_name, alt))

    return combos


ALL_PHASES = {
    1:  ("Learning Rate",     phase_1_learning_rate),
    2:  ("Weight Decay",      phase_2_weight_decay),
    3:  ("Batch Size",        phase_3_batch_size),
    4:  ("Dropout",           phase_4_dropout),
    5:  ("Hidden Dims",       phase_5_hidden_dims),
    6:  ("Mixture Components", phase_6_components),
    7:  ("Sigma Min",         phase_7_sigma_min),
    8:  ("Gradient Clipping", phase_8_grad_clip),
    9:  ("LR Scheduler",     phase_9_scheduler),
    10: ("ES Patience",       phase_10_es_patience),
    # phase 11 handled specially (needs existing results)
}


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MDN hyperparameter grid search")
    parser.add_argument(
        "--phase", type=int, nargs="*", default=None,
        help="Run only specific phase(s), e.g. --phase 1 2.  "
             "Default: run all phases 1-11.")
    args = parser.parse_args()
    phases_to_run = args.phase or list(range(1, 12))

    print("=" * 70)
    print("  MDN HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"  Device       : {DEVICE}")
    print(f"  Max folds    : {MAX_FOLDS}")
    print(f"  Max epochs   : {MAX_EPOCHS}")
    print(f"  Phases       : {phases_to_run}")
    print(f"  Results file : {RESULTS_PATH}")

    # Load data once
    print("\n  Loading dataset ...")
    df, feature_cols, windows = load_and_prepare()
    print(f"  Rows: {len(df):,}  Features: {len(feature_cols)}  "
          f"Windows: {len(windows)} (using first {MAX_FOLDS})\n")

    existing = load_existing()
    n_existing = len(existing)
    print(f"  Existing results: {n_existing} experiments")

    total_new = 0

    for phase_id in phases_to_run:
        if phase_id == 11:
            experiments = phase_11_best_combo(existing)
            phase_name = "Best Combo"
        elif phase_id in ALL_PHASES:
            phase_name, gen_fn = ALL_PHASES[phase_id]
            experiments = gen_fn()
        else:
            print(f"\n  Unknown phase {phase_id}, skipping.")
            continue

        if not experiments:
            print(f"\n  Phase {phase_id} ({phase_name}): nothing to run.")
            continue

        print(f"\n{'='*70}")
        print(f"  Phase {phase_id}: {phase_name}  "
              f"({len(experiments)} experiments)")
        print(f"{'='*70}")

        for exp_name, cfg in experiments:
            if already_ran(existing, exp_name):
                print(f"\n  SKIP (already done): {exp_name}")
                continue

            print(f"\n  {'─'*60}")
            print(f"  {exp_name}")
            # print key diffs from baseline
            diffs = {k: v for k, v in cfg.items()
                     if BASELINE_CFG.get(k) != v}
            if diffs:
                print(f"  Changed: {diffs}")
            print(f"  {'─'*60}")

            t0 = time.time()
            result = run_experiment(exp_name, cfg, df, feature_cols, windows)
            elapsed = time.time() - t0
            result["time_min"] = round(elapsed / 60, 1)

            avg = result.get("avg_imp")
            avg_str = f"{avg:+.2f}%" if avg is not None and not math.isnan(avg) else "N/A"
            print(f"\n  => avg_improvement = {avg_str}  "
                  f"({result['time_min']} min)")

            existing.append(result)
            total_new += 1

            # save after each experiment (crash-safe)
            save_results(existing)

    # -- Final summary table -----------------------------------------------
    print(f"\n\n{'='*90}")
    print(f"  FULL RESULTS TABLE  ({len(existing)} experiments)")
    print(f"{'='*90}")
    print(f"{'#':>3s}  {'Name':<42s} {'Avg%':>8s} {'Up%':>8s} "
          f"{'Dn%':>8s} {'UpNLL':>7s} {'DnNLL':>7s} {'Min':>5s}")
    print("-" * 90)

    # sort by avg_imp descending
    ranked = sorted(
        enumerate(existing),
        key=lambda x: x[1].get("avg_imp") or float("-inf"),
        reverse=True,
    )
    for rank, (idx, r) in enumerate(ranked, 1):
        avg = r.get("avg_imp")
        avg_s = f"{avg:+.2f}" if avg is not None else "N/A"
        up = r.get("avg_up_imp")
        up_s = f"{up:+.2f}" if up is not None else "N/A"
        dn = r.get("avg_dn_imp")
        dn_s = f"{dn:+.2f}" if dn is not None else "N/A"
        unll = r.get("avg_up_nll")
        unll_s = f"{unll:.3f}" if unll is not None else "N/A"
        dnll = r.get("avg_dn_nll")
        dnll_s = f"{dnll:.3f}" if dnll is not None else "N/A"
        t = r.get("time_min", 0)
        name = r.get("name", "?")[:42]

        marker = " *" if rank == 1 else ""
        print(f"{rank:>3d}  {name:<42s} {avg_s:>8s} {up_s:>8s} "
              f"{dn_s:>8s} {unll_s:>7s} {dnll_s:>7s} {t:>5.1f}{marker}")

    print("-" * 90)
    print(f"  * = best config overall")
    print(f"  New experiments added this run: {total_new}")
    print(f"  Results saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
