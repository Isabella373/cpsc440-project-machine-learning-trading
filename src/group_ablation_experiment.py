"""
group_ablation_experiment.py
============================
Feature-group ablation: remove one group at a time to measure each group's
contribution.  Does NOT modify any existing code.

Groups (raw column names – before one-hot):
  1. RETURN / MOMENTUM  : ret_1d, ret_5d, ret_20d, momentum_20d, momentum_60d
  2. MA / TREND          : ma_ratio_5, ma_ratio_20, rsi_14, macd
  3. VOLATILITY / DIST   : vol_20d, kurt_20d, skew_20d, hl_spread, oc_return
  4. VOLUME / LIQUIDITY  : volume, volume_zscore, dollar_volume
  5. MACRO / RATES       : vix, vix_ma20, vix_slope, vix_change_9d, vvix,
                           dxy, dxy_ma20, dxy_ret_1d,
                           bond_yield_3m, bond_yield_5y,
                           gld_ret, ief_ret, tlt_ret, wti_ret,
                           initial_claims, nfp, unrate
  6. CALENDAR            : is_friday, is_month_end, is_opex_day, is_opex_week,
                           is_pre_holiday, is_pre_long_weekend, is_quarter_end
  7. SECTOR (one-hot)    : sector  (becomes ~18 dummies)

Additional experiments:
  8. DROP REDUNDANT ONLY : remove momentum_20d (=ret_20d, r=1.0), and one of
     each highly-correlated pair to test minimal-redundancy set
  9. KEEP BEST ONLY      : if results suggest some groups are useless, test
     removing multiple groups at once

Methodology: 3 folds, MDN K=5, CPU, 80 epochs, patience 10.
"""

from __future__ import annotations
import time, warnings, json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.constants import (
    PROJECT_ROOT, DATA_PATH, DATE_COL, TICKER_COL, TARGET_UP, TARGET_DN,
    RANDOM_SEED, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS, PURGE_GAP_DAYS,
    EXCLUDE_COLS_NN as EXCLUDE_COLS,
    HIDDEN_DIMS, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, VAL_FRAC,
)
from common.data import (
    build_updown_targets, check_required_columns, build_rolling_windows,
    robust_standardize as _robust_standardize,
    inverse_standardize as _inverse_standardize,
)
from train_mdn_rolling import (
    SIGMA_MIN, N_COMPONENTS,
    mdn_nll_loss, mdn_mean,
    MDN,
)

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")
MAX_FOLDS = 3
MAX_EPOCHS = 80
PATIENCE = 10

# ===================================================================
# FEATURE GROUPS  (raw column names, before one-hot)
# ===================================================================
GROUPS: Dict[str, List[str]] = {
    "return_momentum": [
        "ret_1d", "ret_5d", "ret_20d", "momentum_20d", "momentum_60d",
    ],
    "ma_trend": [
        "ma_ratio_5", "ma_ratio_20", "rsi_14", "macd",
    ],
    "volatility_dist": [
        "vol_20d", "kurt_20d", "skew_20d", "hl_spread", "oc_return",
    ],
    "volume_liquidity": [
        "volume", "volume_zscore", "dollar_volume",
    ],
    "macro_rates": [
        "vix", "vix_ma20", "vix_slope", "vix_change_9d", "vvix",
        "dxy", "dxy_ma20", "dxy_ret_1d",
        "bond_yield_3m", "bond_yield_5y",
        "gld_ret", "ief_ret", "tlt_ret", "wti_ret",
        "initial_claims", "nfp", "unrate",
    ],
    "calendar": [
        "is_friday", "is_month_end", "is_opex_day", "is_opex_week",
        "is_pre_holiday", "is_pre_long_weekend", "is_quarter_end",
    ],
    "sector": [
        "sector",  # one-hot expanded later
    ],
}

# Redundant columns to drop in the "drop-redundant" experiment
# Based on correlation analysis:
#   ret_20d vs momentum_20d  r=1.000  → drop momentum_20d
#   dxy     vs dxy_ma20      r=0.981  → drop dxy_ma20
#   bond_yield_3m vs bond_yield_5y r=0.934 → drop bond_yield_5y
#   ief_ret vs tlt_ret       r=0.914  → drop tlt_ret
#   vix     vs vix_ma20      r=0.839  → drop vix_ma20
#   ret_20d vs ma_ratio_20   r=0.828  → keep both (different groups)
#   ma_ratio_20 vs rsi_14    r=0.811  → drop rsi_14 (redundant with ma_ratio_20)
#   ret_5d  vs ma_ratio_5    r=0.783  → keep both (different groups)
REDUNDANT_COLS = [
    "momentum_20d",   # r=1.000 with ret_20d
    "dxy_ma20",       # r=0.981 with dxy
    "bond_yield_5y",  # r=0.934 with bond_yield_3m
    "tlt_ret",        # r=0.914 with ief_ret
    "vix_ma20",       # r=0.839 with vix
    "rsi_14",         # r=0.811 with ma_ratio_20
]


# ===================================================================
# DATA PREPARATION
# ===================================================================
def prepare_data(
    raw: pd.DataFrame,
    drop_raw_cols: List[str] | None = None,
    drop_sector_dummies: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare dataset, optionally dropping columns before feature inference."""
    df = raw.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    # Build targets
    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])

    # Drop specified raw columns BEFORE one-hot
    if drop_raw_cols:
        existing = [c for c in drop_raw_cols if c in df.columns and c != "sector"]
        if existing:
            df = df.drop(columns=existing)
        # Handle sector separately (it's a categorical → one-hot)
        if "sector" in drop_raw_cols and "sector" in df.columns:
            df = df.drop(columns=["sector"])

    # Save ticker
    saved_ticker = df[TICKER_COL].copy()

    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_COLS]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    df[TICKER_COL] = saved_ticker.values

    # Drop sector dummies if requested (for testing sector removal post one-hot)
    if drop_sector_dummies:
        sector_cols = [c for c in df.columns if c.startswith("sector_")]
        if sector_cols:
            df = df.drop(columns=sector_cols)

    # Infer feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    return df, feature_cols


# ===================================================================
# TRAIN + EVAL  (identical to feature_ablation_experiment)
# ===================================================================
def _encode_and_scale(X_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.values.astype(np.float32))
    Xte = scaler.transform(X_test.values.astype(np.float32))
    return np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0), scaler


def train_and_eval_target(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Tuple[float, float]:
    """Train MDN on one target, return (rmse, nll)."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_proc, clip_lo, clip_hi, y_center, y_scale = _robust_standardize(y_train)

    n = len(X_train)
    n_val = max(1, int(n * VAL_FRAC))
    n_tr = n - n_val

    Xtr = torch.tensor(X_train[:n_tr], dtype=torch.float32)
    ytr = torch.tensor(y_proc[:n_tr], dtype=torch.float32)
    Xva = torch.tensor(X_train[n_tr:], dtype=torch.float32)
    yva = torch.tensor(y_proc[n_tr:], dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    y_test_z = (y_test - y_center) / y_scale
    yte = torch.tensor(y_test_z, dtype=torch.float32)

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MDN(X_train.shape[1], N_COMPONENTS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_nll_loss(pi, mu, sigma, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pi, mu, sigma = model(Xva)
            vl = mdn_nll_loss(pi, mu, sigma, yva).item()
        sched.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(Xte)
        pred_z = mdn_mean(pi, mu).numpy()
        test_nll = mdn_nll_loss(pi, mu, sigma, yte).item()

    pred = _inverse_standardize(pred_z, y_center, y_scale, clip_lo, clip_hi)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return rmse, test_nll


def run_experiment(
    name: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: list,
) -> dict:
    """Run MDN across folds, return summary."""
    up_imps, dn_imps = [], []
    up_nlls, dn_nlls = [], []

    for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(windows[:MAX_FOLDS], 1):
        train_mask = (df[DATE_COL] >= tr_s) & (df[DATE_COL] <= tr_e)
        test_mask = (df[DATE_COL] >= te_s) & (df[DATE_COL] <= te_e)
        train_df = df.loc[train_mask].dropna(subset=feature_cols + [TARGET_UP, TARGET_DN])
        test_df = df.loc[test_mask].dropna(subset=feature_cols + [TARGET_UP, TARGET_DN])

        if train_df.empty or test_df.empty:
            continue

        X_tr_raw, X_te_raw = train_df[feature_cols], test_df[feature_cols]
        X_tr_raw, X_te_raw = X_tr_raw.align(X_te_raw, join="left", axis=1, fill_value=0)
        Xtr, Xte, _ = _encode_and_scale(X_tr_raw, X_te_raw)

        for tgt, label in [(TARGET_UP, "up"), (TARGET_DN, "dn")]:
            y_tr = train_df[tgt].values.astype(np.float32)
            y_te = test_df[tgt].values.astype(np.float32)

            try:
                rmse, nll = train_and_eval_target(Xtr, y_tr, Xte, y_te)
            except Exception as e:
                print(f"    Fold {fold_id} {label} FAILED: {e}")
                continue

            naive_rmse = float(np.sqrt(mean_squared_error(y_te, np.zeros_like(y_te))))
            imp = (1 - rmse / naive_rmse) * 100

            if label == "up":
                up_imps.append(imp)
                up_nlls.append(nll)
            else:
                dn_imps.append(imp)
                dn_nlls.append(nll)

        avg_imp_fold = (up_imps[-1] + dn_imps[-1]) / 2
        print(f"    Fold {fold_id}: up={up_imps[-1]:+.1f}%  dn={dn_imps[-1]:+.1f}%  "
              f"avg={avg_imp_fold:+.1f}%")

    if not up_imps:
        return {"name": name, "n_features": len(feature_cols),
                "avg_imp": float("nan"), "avg_up_imp": float("nan"),
                "avg_dn_imp": float("nan"), "avg_up_nll": float("nan"),
                "avg_dn_nll": float("nan")}

    avg_imp = float(np.mean([(u + d) / 2 for u, d in zip(up_imps, dn_imps)]))
    return {
        "name": name,
        "n_features": len(feature_cols),
        "avg_imp": avg_imp,
        "avg_up_imp": float(np.mean(up_imps)),
        "avg_dn_imp": float(np.mean(dn_imps)),
        "avg_up_nll": float(np.mean(up_nlls)),
        "avg_dn_nll": float(np.mean(dn_nlls)),
    }


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 75)
    print("  GROUP ABLATION: Remove One Feature Group at a Time")
    print("=" * 75)
    print(f"  Folds: {MAX_FOLDS}  |  Epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Device: {DEVICE}")

    raw = pd.read_csv(DATA_PATH)
    check_required_columns(raw)

    # ---------------------------------------------------------------
    # Phase 1:  BASELINE + drop-one-group experiments
    # ---------------------------------------------------------------
    experiments = []

    # A: Baseline (all features)
    experiments.append(("A: BASELINE (all 63 feats)", None, False))

    # B–H: Remove one group at a time
    for i, (grp_name, grp_cols) in enumerate(GROUPS.items()):
        letter = chr(ord("B") + i)
        experiments.append((
            f"{letter}: DROP {grp_name}",
            grp_cols,
            grp_name == "sector",  # special handling for sector dummies
        ))

    # I: Drop redundant columns only
    experiments.append(("I: DROP redundant (6 cols)", REDUNDANT_COLS, False))

    # ---------------------------------------------------------------
    # Run all experiments
    # ---------------------------------------------------------------
    results = []
    windows = None

    for exp_name, drop_cols, drop_sector_flag in experiments:
        print(f"\n{'─' * 65}")
        print(f"  {exp_name}")
        print(f"{'─' * 65}")

        t0 = time.time()

        # For group drops, we drop the raw columns; for sector, we drop dummies
        if drop_sector_flag:
            # Sector is categorical – drop the one-hot dummies
            df, feature_cols = prepare_data(raw, drop_raw_cols=["sector"])
        else:
            df, feature_cols = prepare_data(raw, drop_raw_cols=drop_cols)

        print(f"    Features: {len(feature_cols)}")

        if drop_cols:
            actually_dropped = [c for c in (drop_cols or [])
                                if c not in [cc for cc in df.columns] or c == "sector"]
            print(f"    Dropped: {drop_cols}")

        # Build windows once
        if windows is None:
            windows = build_rolling_windows(
                all_dates=df[DATE_COL],
                train_months=TRAIN_MONTHS,
                test_months=TEST_MONTHS,
                step_months=STEP_MONTHS,
                purge_gap_days=PURGE_GAP_DAYS,
            )
            print(f"    Windows: {len(windows)}, running first {MAX_FOLDS}")

        res = run_experiment(exp_name, df, feature_cols, windows)
        elapsed = time.time() - t0
        res["time_min"] = round(elapsed / 60, 1)
        results.append(res)
        print(f"\n  => avg_improvement = {res['avg_imp']:+.2f}%  "
              f"({res['n_features']} feats, {res['time_min']} min)")

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    baseline_imp = results[0]["avg_imp"]

    print("\n" + "=" * 95)
    print("  GROUP ABLATION RESULTS")
    print("=" * 95)
    print(f"{'Experiment':<35s} {'#Feats':>6s} {'Avg Imp%':>9s} {'Δ vs Base':>10s} "
          f"{'Up Imp%':>9s} {'Dn Imp%':>9s} {'Time':>6s} {'Verdict':>12s}")
    print("-" * 95)

    for r in results:
        delta = r["avg_imp"] - baseline_imp
        if r is results[0]:
            verdict = "BASELINE"
            delta_str = "—"
        else:
            delta_str = f"{delta:+.2f}%"
            if delta > 1.0:
                verdict = "✅ BETTER"
            elif delta > 0.3:
                verdict = "📈 Slightly+"
            elif delta > -0.3:
                verdict = "≈ Same"
            elif delta > -1.0:
                verdict = "📉 Slightly-"
            else:
                verdict = "❌ WORSE"

        print(f"{r['name']:<35s} {r['n_features']:>6d} {r['avg_imp']:>+8.2f}% "
              f"{delta_str:>10s} {r['avg_up_imp']:>+8.2f}% {r['avg_dn_imp']:>+8.2f}% "
              f"{r['time_min']:>5.1f}m {verdict:>12s}")

    print("-" * 95)

    # ---------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------
    print("\n  INTERPRETATION GUIDE:")
    print("  • If dropping a group makes the score BETTER → the group is NOISE (remove it!)")
    print("  • If dropping a group makes the score WORSE  → the group is USEFUL (keep it)")
    print("  • If dropping a group has NO effect          → the group is REDUNDANT (can remove)")

    # Sort by delta (which groups hurt most when removed = most valuable)
    ranked = sorted(results[1:], key=lambda r: r["avg_imp"] - baseline_imp)
    print("\n  FEATURE GROUPS RANKED (most valuable → least valuable):")
    for r in ranked:
        delta = r["avg_imp"] - baseline_imp
        print(f"    {delta:+.2f}%  {r['name']}")

    # ---------------------------------------------------------------
    # Phase 2: If any groups were noise (dropping improved score),
    # test dropping ALL noise groups together
    # ---------------------------------------------------------------
    noise_groups = []
    for r in results[1:-1]:  # skip baseline and redundant experiment
        delta = r["avg_imp"] - baseline_imp
        if delta > 0.3:  # dropping this group helped
            # Figure out which group this was
            idx = results.index(r) - 1  # offset by baseline
            grp_name = list(GROUPS.keys())[idx]
            noise_groups.append(grp_name)

    if noise_groups:
        print(f"\n{'─' * 65}")
        print(f"  BONUS: Drop ALL noise groups: {noise_groups}")
        print(f"{'─' * 65}")

        all_noise_cols = []
        drop_sector = False
        for gn in noise_groups:
            if gn == "sector":
                drop_sector = True
            else:
                all_noise_cols.extend(GROUPS[gn])

        t0 = time.time()
        if drop_sector:
            df, feature_cols = prepare_data(raw, drop_raw_cols=all_noise_cols + ["sector"])
        else:
            df, feature_cols = prepare_data(raw, drop_raw_cols=all_noise_cols)
        print(f"    Features: {len(feature_cols)}")
        print(f"    Dropped groups: {noise_groups}")
        print(f"    Dropped cols: {all_noise_cols}")

        res = run_experiment("BONUS: Drop all noise", df, feature_cols, windows)
        elapsed = time.time() - t0
        res["time_min"] = round(elapsed / 60, 1)

        delta = res["avg_imp"] - baseline_imp
        print(f"\n  => BONUS avg_improvement = {res['avg_imp']:+.2f}%  "
              f"(Δ={delta:+.2f}% vs baseline, {res['n_features']} feats)")
    else:
        print("\n  No noise groups detected (all groups contribute or are neutral).")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    save_dir = os.path.join(PROJECT_ROOT, "results", "group_ablation")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "group_ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {save_path}")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plot_group_ablation(save_dir, results)


# ===================================================================
# PLOTTING
# ===================================================================
_PRETTY = {
    "sector": "Sector\n(one-hot)", "macro_rates": "Macro /\nRates",
    "volatility_dist": "Volatility /\nDistribution",
    "redundant": "Drop\nRedundant", "volume_liquidity": "Volume /\nLiquidity",
    "return_momentum": "Return /\nMomentum",
    "ma_trend": "MA / Trend", "calendar": "Calendar",
}

def _group_key(name: str) -> str:
    return name.split("DROP ")[-1].split(" (")[0]


def plot_group_ablation(save_dir: str, results: list | None = None):
    """Generate 3 figures from group-ablation results."""
    out = save_dir if isinstance(save_dir, str) else str(save_dir)

    if results is None:
        with open(os.path.join(out, "group_ablation_results.json")) as f:
            results = json.load(f)

    bl   = next(r for r in results if "BASELINE" in r["name"])
    exps = [r for r in results if "BASELINE" not in r["name"]]
    BL, BL_UP, BL_DN = bl["avg_imp"], bl["avg_up_imp"], bl["avg_dn_imp"]
    BL_NLL_UP, BL_NLL_DN = bl["avg_up_nll"], bl["avg_dn_nll"]

    ranked = sorted(exps, key=lambda r: BL - r["avg_imp"], reverse=True)
    keys   = [_group_key(r["name"]) for r in ranked]
    labels = [_PRETTY.get(k, k) for k in keys]
    deltas = [BL - r["avg_imp"] for r in ranked]
    x      = np.arange(len(ranked))

    def _severity_color(d):
        if d > 5:   return "#C62828"
        if d > 2:   return "#EF6C00"
        if d > 0:   return "#FFA726"
        return "#2E7D32"

    colors = [_severity_color(d) for d in deltas]

    # ── Fig 1: Importance waterfall ───────────────────────────────
    fig1, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x, deltas, color=colors, edgecolor="white", width=0.65, zorder=3)
    for i, (d, r) in enumerate(zip(deltas, ranked)):
        if d >= 0:
            ax.text(i, d + 0.25, f"−{d:.2f} pp", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colors[i])
            ax.text(i, d / 2, f"→ {r['avg_imp']:.1f}%", ha="center",
                    va="center", fontsize=7.5, color="white", fontweight="bold")
        else:
            ax.text(i, d - 0.25, f"+{abs(d):.2f} pp", ha="center", va="top",
                    fontsize=9, fontweight="bold", color=colors[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Performance Drop (pp)", fontsize=11)
    ax.set_title("Feature-Group Ablation: How Much Does Each Group Matter?",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[
        Patch(facecolor="#C62828", label="Critical (> 5 pp)"),
        Patch(facecolor="#EF6C00", label="Important (2–5 pp)"),
        Patch(facecolor="#FFA726", label="Moderate (0–2 pp)"),
        Patch(facecolor="#2E7D32", label="Safe to remove (≤ 0 pp)"),
    ], loc="upper right", fontsize=8, framealpha=0.9,
       title="Impact Severity", title_fontsize=9)
    ax.annotate(f"Baseline = {BL:.2f}% avg improvement",
                xy=(len(x) - 1, -0.8), fontsize=8.5, fontstyle="italic",
                color="#555")
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(min(deltas) - 1, max(deltas) + 2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out, "fig1_group_importance.png"),
                 dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved fig1_group_importance.png")

    # ── Fig 2: Up vs Down breakdown ──────────────────────────────
    up_d = [BL_UP - r["avg_up_imp"] for r in ranked]
    dn_d = [BL_DN - r["avg_dn_imp"] for r in ranked]
    w = 0.25

    fig2, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x - w, deltas, w, label="Average",        color="#424242",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x,     up_d,   w, label="y_up (upside)",  color="#1565C0",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + w, dn_d,   w, label="y_down (downside)", color="#C62828",
           edgecolor="white", linewidth=0.5, zorder=3)
    for i in range(len(ranked)):
        for val, xo in [(deltas[i], -w), (up_d[i], 0), (dn_d[i], w)]:
            s = "−" if val > 0 else "+"
            ax.text(i + xo, val + (0.25 if val >= 0 else -0.25),
                    f"{s}{abs(val):.1f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=6.5, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Performance Drop (pp)", fontsize=10)
    ax.set_title("Asymmetric Impact: Upside vs Downside Forecasting",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    si = keys.index("sector")
    ax.annotate("Sector removal\ndestroys downside\nforecasting",
                xy=(si + w, dn_d[si]),
                xytext=(si + 1.3, dn_d[si] - 1.5),
                fontsize=8, fontstyle="italic", color="#C62828",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))
    fig2.tight_layout()
    fig2.savefig(os.path.join(out, "fig2_updown_breakdown.png"),
                 dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved fig2_updown_breakdown.png")

    # ── Fig 3: NLL degradation + dimensionality scatter ──────────
    up_nll = [r["avg_up_nll"] - BL_NLL_UP for r in ranked]
    dn_nll = [r["avg_dn_nll"] - BL_NLL_DN for r in ranked]

    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={"width_ratios": [2.2, 1]})

    hw = 0.25
    axL.bar(x - hw/2, up_nll, hw, label="y_up NLL change",
            color="#1565C0", edgecolor="white", linewidth=0.5, zorder=3)
    axL.bar(x + hw/2, dn_nll, hw, label="y_down NLL change",
            color="#C62828", edgecolor="white", linewidth=0.5, zorder=3)
    for i in range(len(ranked)):
        for val, xo, c in [(up_nll[i], -hw/2, "#1565C0"),
                            (dn_nll[i], hw/2, "#C62828")]:
            if abs(val) > 0.01:
                axL.text(i + xo, val + (0.01 if val >= 0 else -0.01),
                         f"{'+' if val > 0 else ''}{val:.3f}", ha="center",
                         va="bottom" if val >= 0 else "top",
                         fontsize=6.5, color=c, fontweight="bold")
    axL.axhline(0, color="black", linewidth=0.8)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("NLL Change vs Baseline (higher = worse)", fontsize=9.5)
    axL.set_title("Distributional Fit Degradation (NLL)",
                  fontsize=11, fontweight="bold")
    axL.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axL.grid(axis="y", alpha=0.15, zorder=0)

    # Right: feature count vs improvement
    all_r = [bl] + list(ranked)
    fc = [r["n_features"] for r in all_r]
    ai = [r["avg_imp"]    for r in all_r]
    sc = ["#2E7D32"] + [_severity_color(BL - r["avg_imp"]) for r in ranked]
    el = ["Baseline"] + [_PRETTY.get(_group_key(r["name"]),
          _group_key(r["name"])).replace("\n", " ") for r in ranked]

    axR.scatter(fc, ai, c=sc, s=120, edgecolors="white", linewidth=1.2, zorder=3)
    for i, (fx, ay, lb) in enumerate(zip(fc, ai, el)):
        xo, yo, ha = 0.8, 0.0, "left"
        if "Sector" in lb:  yo = 1.0
        elif "Baseline" in lb: xo, ha = -0.8, "right"
        elif "Calendar" in lb or "MA" in lb: yo = -0.6
        axR.annotate(lb, (fx, ay), xytext=(fx + xo, ay + yo), fontsize=7,
                     ha=ha, va="center",
                     arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5)
                     if abs(xo) > 0.5 or abs(yo) > 0.5 else None)
    axR.axhline(BL, color="#2E7D32", ls="--", lw=1, alpha=0.5)
    axR.text(max(fc) + 0.5, BL + 0.3, f"Baseline\n{BL:.1f}%",
             fontsize=7, color="#2E7D32", va="bottom")
    axR.set_xlabel("Number of Features", fontsize=9.5)
    axR.set_ylabel("Avg Improvement (%)", fontsize=9.5)
    axR.set_title("Dimensionality vs Performance",
                  fontsize=11, fontweight="bold")
    axR.grid(alpha=0.15, zorder=0)

    fig3.suptitle("Probability Calibration & Feature Efficiency",
                  fontsize=13, fontweight="bold", y=1.02)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out, "fig3_nll_and_dimensionality.png"),
                 dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved fig3_nll_and_dimensionality.png")


if __name__ == "__main__":
    main()
