"""
feature_ablation_experiment.py
==============================
Test whether cross-sectional rank features and relative-to-market/sector
features improve MDN performance.  Does NOT modify any existing code.

Experiments (3 folds, MDN K=5, CPU):
  A. BASELINE       — current 63 features
  B. + RANK         — add 12 cross-sectional percentile rank features
  C. + RELATIVE     — add relative-to-market & relative-to-sector features
  D. + RANK + REL   — add both
"""

from __future__ import annotations
import time, warnings, math
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    mdn_nll_loss, mdn_mean, mdn_std,
    MDN,
)

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")
MAX_FOLDS = 3


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================

# Columns to compute cross-sectional rank
RANK_COLS = [
    "ret_1d", "ret_5d", "ret_20d",
    "momentum_20d", "momentum_60d",
    "rsi_14", "macd",
    "vol_20d", "volume_zscore", "dollar_volume",
    "hl_spread", "oc_return",
]

# Columns for relative-to-market and relative-to-sector
RELATIVE_RET_COLS = ["ret_1d", "ret_5d", "ret_20d"]
RELATIVE_VOL_COLS = ["vol_20d"]


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional percentile rank features (per date)."""
    df = df.copy()
    for col in RANK_COLS:
        if col not in df.columns:
            continue
        rank_col = f"{col}_rank"
        df[rank_col] = df.groupby(DATE_COL)[col].rank(pct=True, method="average")
        df[rank_col] = df[rank_col].fillna(0.5)
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative-to-market and relative-to-sector features."""
    df = df.copy()

    # --- Relative to market (cross-sectional mean of all stocks that day) ---
    for col in RELATIVE_RET_COLS:
        if col not in df.columns:
            continue
        mkt_mean = df.groupby(DATE_COL)[col].transform("mean")
        df[f"{col}_vs_mkt"] = df[col] - mkt_mean

    for col in RELATIVE_VOL_COLS:
        if col not in df.columns:
            continue
        mkt_mean = df.groupby(DATE_COL)[col].transform("mean")
        # ratio (avoid div by zero)
        df[f"{col}_vs_mkt_ratio"] = df[col] / (mkt_mean + 1e-10)

    # --- Relative to sector ---
    if "sector" in df.columns:
        for col in RELATIVE_RET_COLS:
            if col not in df.columns:
                continue
            sec_mean = df.groupby([DATE_COL, "sector"])[col].transform("mean")
            df[f"{col}_vs_sector"] = df[col] - sec_mean

        for col in RELATIVE_VOL_COLS:
            if col not in df.columns:
                continue
            sec_mean = df.groupby([DATE_COL, "sector"])[col].transform("mean")
            df[f"{col}_vs_sector_ratio"] = df[col] / (sec_mean + 1e-10)

    return df


# ===================================================================
# DATA PREP (mirrors train_mdn_rolling.prepare_data)
# ===================================================================
def prepare_data_with_features(
    raw: pd.DataFrame,
    add_ranks: bool = False,
    add_relative: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data and optionally add new features."""
    df = raw.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    # Build targets
    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])

    # Add new features BEFORE one-hot encoding (need ticker/sector/date)
    if add_ranks:
        df = add_rank_features(df)
    if add_relative:
        df = add_relative_features(df)

    # Save ticker
    saved_ticker = df[TICKER_COL].copy()

    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_COLS]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    df[TICKER_COL] = saved_ticker.values

    # Infer feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    return df, feature_cols


# ===================================================================
# TRAIN + EVAL (same as MDN ablation)
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
    for epoch in range(80):
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
            if wait >= 10:
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


# ===================================================================
# RUN EXPERIMENT
# ===================================================================
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
    print("=" * 70)
    print("  FEATURE ABLATION: Cross-sectional Rank + Relative Features")
    print("=" * 70)

    raw = pd.read_csv(DATA_PATH)
    check_required_columns(raw)

    # Prepare 4 variants
    configs = [
        ("A: BASELINE (current)", False, False),
        ("B: + RANK features", True, False),
        ("C: + RELATIVE features", False, True),
        ("D: + RANK + RELATIVE", True, True),
    ]

    results = []
    windows = None

    for name, add_ranks, add_rel in configs:
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")

        t0 = time.time()
        df, feature_cols = prepare_data_with_features(raw, add_ranks=add_ranks, add_relative=add_rel)
        print(f"    Features: {len(feature_cols)}")

        # Only build windows once (they're the same for all)
        if windows is None:
            windows = build_rolling_windows(
                all_dates=df[DATE_COL],
                train_months=TRAIN_MONTHS,
                test_months=TEST_MONTHS,
                step_months=STEP_MONTHS,
                purge_gap_days=PURGE_GAP_DAYS,
            )
            print(f"    Windows: {len(windows)}, running first {MAX_FOLDS}")

        # Show new features added
        if add_ranks or add_rel:
            new_feats = [f for f in feature_cols if "_rank" in f or "_vs_" in f]
            print(f"    New features ({len(new_feats)}): {new_feats[:8]}...")

        res = run_experiment(name, df, feature_cols, windows)
        elapsed = time.time() - t0
        res["time_min"] = round(elapsed / 60, 1)
        results.append(res)
        print(f"\n  => avg_improvement = {res['avg_imp']:+.2f}%  "
              f"({res['n_features']} feats, {res['time_min']} min)")

    # Summary
    print("\n" + "=" * 85)
    print("  FEATURE ABLATION RESULTS")
    print("=" * 85)
    print(f"{'Experiment':<30s} {'#Feats':>6s} {'Avg Imp%':>9s} {'Up Imp%':>9s} "
          f"{'Dn Imp%':>9s} {'Up NLL':>8s} {'Dn NLL':>8s} {'Time':>6s}")
    print("-" * 85)

    baseline_imp = results[0]["avg_imp"]
    for r in results:
        delta = r["avg_imp"] - baseline_imp
        marker = "" if r is results[0] else f"  ({delta:+.2f}%)"
        print(f"{r['name']:<30s} {r['n_features']:>6d} {r['avg_imp']:>+8.2f}% "
              f"{r['avg_up_imp']:>+8.2f}% {r['avg_dn_imp']:>+8.2f}% "
              f"{r['avg_up_nll']:>8.3f} {r['avg_dn_nll']:>8.3f} "
              f"{r['time_min']:>5.1f}m{marker}")
    print("-" * 85)

    # Interpretation
    print("\n  KEY FINDINGS:")
    for r in results[1:]:
        delta = r["avg_imp"] - baseline_imp
        if delta > 1.0:
            verdict = "✅ SIGNIFICANT IMPROVEMENT"
        elif delta > 0.3:
            verdict = "📈 Moderate improvement"
        elif delta > -0.3:
            verdict = "≈ No meaningful change"
        else:
            verdict = "❌ Worse"
        print(f"    {r['name']:<30s}: {delta:+.2f}%  {verdict}")


if __name__ == "__main__":
    main()
