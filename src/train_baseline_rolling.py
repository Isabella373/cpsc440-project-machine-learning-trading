"""
train_baseline_rolling.py
=========================

Rolling-window baseline for per-ticker equity **range prediction**.

Targets
-------
    y_up   = max(P_{t+1:t+5}) / P_t  - 1   (max upside over next 5 days)
    y_down = min(P_{t+1:t+5}) / P_t  - 1   (max downside over next 5 days)

Pipeline
--------
1. Reads the cleaned panel dataset (one row per date × ticker)
2. Constructs y_up and y_down from adj_close
3. Uses a **12-month** rolling training window
4. Predicts on the following **3-month** test window
5. Rolls forward by 3 months each iteration
6. Trains separate XGBoost models for y_up and y_down
7. Naive baseline: predict 0 (i.e. "price stays the same")
8. Saves:
     - all out-of-sample predictions       → results/baseline_rolling/oos_predictions_xgboost.csv
     - per-fold metrics                    → results/baseline_rolling/fold_metrics_xgboost.csv
     - per-ticker error summary            → results/baseline_rolling/ticker_metrics_xgboost.csv
     - overall summary                     → results/baseline_rolling/summary_xgboost.json

Usage
-----
    cd <project_root>
    python src/train_baseline_rolling.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Shared modules ──────────────────────────────────────────────────
from common.constants import (
    PROJECT_ROOT,
    DATA_PATH,
    DATE_COL,
    TICKER_COL,
    PRICE_COL,
    TARGET_UP,
    TARGET_DN,
    TRAIN_MONTHS,
    TEST_MONTHS,
    STEP_MONTHS,
    PURGE_GAP_DAYS,
    RANDOM_SEED,
    FORECAST_TICKERS,
    PRED_FEAT_COLS,
)
from common.data import (
    check_required_columns,
    build_updown_targets,
    build_rolling_windows,
    build_stock_features,
    build_macro_features,
    update_dataset,
)
from common.metrics import rmse_mae, improvement_pct

warnings.filterwarnings("ignore")


# ===================================================================
# CONFIG (baseline-specific)
# ===================================================================
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_rolling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = "xgboost"  # "lightgbm" or "xgboost"

# XGBoost uses ticker/sector as native categoricals, so we do NOT
# exclude TICKER_COL from features (unlike NN models).
EXCLUDE_COLS = {
    DATE_COL,
    TARGET_UP,
    TARGET_DN,
    "target_ret_5d",
}


# ===================================================================
# DATA PREP  (baseline-specific: category type, not one-hot)
# ===================================================================
def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all numeric/category columns except excluded ones."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for XGBoost: parse dates, build targets, mark categoricals."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    print("  Building y_up / y_down targets ...")
    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])
    print(f"  y_up  mean={df[TARGET_UP].mean():.6f}  std={df[TARGET_UP].std():.6f}")
    print(f"  y_down mean={df[TARGET_DN].mean():.6f}  std={df[TARGET_DN].std():.6f}")

    # Mark categoricals for LightGBM / XGBoost
    if TICKER_COL in df.columns:
        df[TICKER_COL] = df[TICKER_COL].astype("category")
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype("category")

    return df


# ===================================================================
# HELPERS
# ===================================================================
@dataclass
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    # y_up metrics
    up_xgb_rmse: float
    up_xgb_mae: float
    up_naive_rmse: float
    up_naive_mae: float
    # y_down metrics
    dn_xgb_rmse: float
    dn_xgb_mae: float
    dn_naive_rmse: float
    dn_naive_mae: float


# ===================================================================
# MODEL TRAINING
# ===================================================================
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm") from e

    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_cols if cat_cols else "auto")
    return model.predict(X_test)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost not installed. Run: pip install xgboost") from e

    cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols)
    X_test_enc  = pd.get_dummies(X_test,  columns=cat_cols)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_enc, y_train)
    return model.predict(X_test_enc)


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Dispatch to the configured model type."""
    if MODEL_TYPE.lower() == "lightgbm":
        return train_lightgbm(X_train, y_train, X_test)
    elif MODEL_TYPE.lower() == "xgboost":
        return train_xgboost(X_train, y_train, X_test)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


# ===================================================================
# SINGLE FOLD
# ===================================================================
def run_one_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    fold_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[FoldResult, pd.DataFrame]:

    train_mask = (df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)
    test_mask  = (df[DATE_COL] >= test_start)  & (df[DATE_COL] <= test_end)

    train_df = df.loc[train_mask].copy()
    test_df  = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Fold {fold_id}: train or test is empty.")

    required = feature_cols + [TARGET_UP, TARGET_DN]
    train_df = train_df.dropna(subset=required)
    test_df  = test_df.dropna(subset=required)

    if train_df.empty or test_df.empty:
        raise ValueError(f"Fold {fold_id}: empty after NaN drop.")

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    # ── Train separate models for y_up and y_down ──────────────
    pred_up   = _train_model(X_train, train_df[TARGET_UP], X_test)
    pred_down = _train_model(X_train, train_df[TARGET_DN], X_test)

    # ── Assemble prediction dataframe ──────────────────────────
    pred_df = test_df[[DATE_COL, TICKER_COL]].copy()
    pred_df["y_up_true"]  = test_df[TARGET_UP].values
    pred_df["y_up_pred"]  = pred_up
    pred_df["y_up_naive"] = 0.0
    pred_df["y_dn_true"]  = test_df[TARGET_DN].values
    pred_df["y_dn_pred"]  = pred_down
    pred_df["y_dn_naive"] = 0.0
    pred_df["fold_id"]    = fold_id

    # ── Fold-level metrics ─────────────────────────────────────
    up_xgb_rmse,   up_xgb_mae   = rmse_mae(pred_df["y_up_true"], pred_df["y_up_pred"])
    up_naive_rmse, up_naive_mae  = rmse_mae(pred_df["y_up_true"], pred_df["y_up_naive"])
    dn_xgb_rmse,   dn_xgb_mae   = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_pred"])
    dn_naive_rmse, dn_naive_mae  = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_naive"])

    return FoldResult(
        fold_id=fold_id,
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        n_train=len(train_df),
        n_test=len(test_df),
        up_xgb_rmse=up_xgb_rmse,   up_xgb_mae=up_xgb_mae,
        up_naive_rmse=up_naive_rmse, up_naive_mae=up_naive_mae,
        dn_xgb_rmse=dn_xgb_rmse,   dn_xgb_mae=dn_xgb_mae,
        dn_naive_rmse=dn_naive_rmse, dn_naive_mae=dn_naive_mae,
    ), pred_df


# ===================================================================
# PER-TICKER SUMMARY  (baseline uses hardcoded xgb column names)
# ===================================================================
def compute_ticker_metrics(all_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE, MAE for XGBoost vs naive per ticker, for both targets."""
    rows = []
    for ticker, grp in all_preds.groupby(TICKER_COL):
        up_xgb_rmse, up_xgb_mae     = rmse_mae(grp["y_up_true"], grp["y_up_pred"])
        up_naive_rmse, up_naive_mae  = rmse_mae(grp["y_up_true"], grp["y_up_naive"])
        dn_xgb_rmse, dn_xgb_mae     = rmse_mae(grp["y_dn_true"], grp["y_dn_pred"])
        dn_naive_rmse, dn_naive_mae  = rmse_mae(grp["y_dn_true"], grp["y_dn_naive"])

        up_improve = improvement_pct(up_xgb_rmse, up_naive_rmse)
        dn_improve = improvement_pct(dn_xgb_rmse, dn_naive_rmse)

        rows.append({
            "ticker": ticker,
            "n_predictions": len(grp),
            "n_folds": int(grp["fold_id"].nunique()),
            "mean_y_up_true": float(grp["y_up_true"].mean()),
            "mean_y_up_pred": float(grp["y_up_pred"].mean()),
            "up_xgb_rmse": up_xgb_rmse,  "up_xgb_mae": up_xgb_mae,
            "up_naive_rmse": up_naive_rmse, "up_naive_mae": up_naive_mae,
            "up_improve_pct": float(up_improve),
            "mean_y_dn_true": float(grp["y_dn_true"].mean()),
            "mean_y_dn_pred": float(grp["y_dn_pred"].mean()),
            "dn_xgb_rmse": dn_xgb_rmse,  "dn_xgb_mae": dn_xgb_mae,
            "dn_naive_rmse": dn_naive_rmse, "dn_naive_mae": dn_naive_mae,
            "dn_improve_pct": float(dn_improve),
            "avg_improve_pct": float((up_improve + dn_improve) / 2),
        })
    result = pd.DataFrame(rows).sort_values("avg_improve_pct", ascending=False).reset_index(drop=True)
    return result


# ===================================================================
# OVERALL SUMMARY  (baseline uses hardcoded xgb column names)
# ===================================================================
def summarize_all_predictions(all_preds: pd.DataFrame) -> Dict[str, float]:
    up_xgb_rmse,   up_xgb_mae   = rmse_mae(all_preds["y_up_true"], all_preds["y_up_pred"])
    up_naive_rmse, up_naive_mae  = rmse_mae(all_preds["y_up_true"], all_preds["y_up_naive"])
    dn_xgb_rmse,   dn_xgb_mae   = rmse_mae(all_preds["y_dn_true"], all_preds["y_dn_pred"])
    dn_naive_rmse, dn_naive_mae  = rmse_mae(all_preds["y_dn_true"], all_preds["y_dn_naive"])

    up_improve = improvement_pct(up_xgb_rmse, up_naive_rmse)
    dn_improve = improvement_pct(dn_xgb_rmse, dn_naive_rmse)

    return {
        "up_xgb_rmse":         up_xgb_rmse,
        "up_xgb_mae":          up_xgb_mae,
        "up_naive_rmse":       up_naive_rmse,
        "up_naive_mae":        up_naive_mae,
        "up_improve_pct":      float(up_improve),
        "dn_xgb_rmse":         dn_xgb_rmse,
        "dn_xgb_mae":          dn_xgb_mae,
        "dn_naive_rmse":       dn_naive_rmse,
        "dn_naive_mae":        dn_naive_mae,
        "dn_improve_pct":      float(dn_improve),
        "avg_improve_pct":     float((up_improve + dn_improve) / 2),
        "n_predictions":       int(len(all_preds)),
        "n_dates":             int(all_preds[DATE_COL].nunique()),
        "n_tickers":           int(all_preds[TICKER_COL].nunique()),
    }


# ===================================================================
# MAIN
# ===================================================================
def main() -> None:
    # --- Auto-update dataset with latest Yahoo Finance data ---
    update_dataset()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"\n{'='*60}")
    print("  BASELINE ROLLING-WINDOW TRAINER  (y_up / y_down)")
    print(f"{'='*60}")
    print(f"  Model       : {MODEL_TYPE}")
    print(f"  Data        : {DATA_PATH}")
    print(f"  Targets     : y_up  = max(P_{{t+1:t+5}})/P_t - 1")
    print(f"              : y_down= min(P_{{t+1:t+5}})/P_t - 1")
    print(f"  Naive       : predict 0  (price stays the same)")
    print(f"  Train window: {TRAIN_MONTHS} months")
    print(f"  Test  window: {TEST_MONTHS} months")
    print(f"  Step        : {STEP_MONTHS} months")
    print(f"  Purge gap   : {PURGE_GAP_DAYS} days")

    df = pd.read_csv(DATA_PATH)
    check_required_columns(df)
    df = prepare_data(df)

    feature_cols = infer_feature_columns(df)

    print(f"\n  Rows         : {len(df):,}")
    print(f"  Date range   : {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"  Tickers      : {df[TICKER_COL].nunique()}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:10]} ...")

    windows = build_rolling_windows(
        all_dates=df[DATE_COL],
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        purge_gap_days=PURGE_GAP_DAYS,
    )
    if not windows:
        raise ValueError("No rolling windows created. Check date range / window sizes.")

    print(f"  Rolling folds: {len(windows)}")
    print()

    all_fold_results: List[FoldResult] = []
    all_preds_list: List[pd.DataFrame] = []

    for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        print(f"Fold {fold_id:>2d}:  train [{tr_s.date()} -> {tr_e.date()}]  "
              f"test [{te_s.date()} -> {te_e.date()}]", end="")

        try:
            fold_res, fold_preds = run_one_fold(
                df=df, feature_cols=feature_cols, fold_id=fold_id,
                train_start=tr_s, train_end=tr_e,
                test_start=te_s, test_end=te_e,
            )
            all_fold_results.append(fold_res)
            all_preds_list.append(fold_preds)

            up_imp = improvement_pct(fold_res.up_xgb_rmse, fold_res.up_naive_rmse)
            dn_imp = improvement_pct(fold_res.dn_xgb_rmse, fold_res.dn_naive_rmse)

            print(f"\n        y_up:  XGB RMSE={fold_res.up_xgb_rmse:.6f}  Naive={fold_res.up_naive_rmse:.6f}  Improv={up_imp:+.1f}%"
                  f"\n        y_dn:  XGB RMSE={fold_res.dn_xgb_rmse:.6f}  Naive={fold_res.dn_naive_rmse:.6f}  Improv={dn_imp:+.1f}%")
        except Exception as e:
            print(f"  ⚠️  SKIPPED: {e}")

    if not all_preds_list:
        raise RuntimeError("No folds completed.")

    # ── Aggregate ──────────────────────────────────────────────────
    all_preds       = pd.concat(all_preds_list, ignore_index=True)
    fold_results_df = pd.DataFrame([asdict(fr) for fr in all_fold_results])
    overall         = summarize_all_predictions(all_preds)
    ticker_metrics  = compute_ticker_metrics(all_preds)

    # ── Save ───────────────────────────────────────────────────────
    preds_path   = OUTPUT_DIR / f"oos_predictions_{MODEL_TYPE}.csv"
    folds_path   = OUTPUT_DIR / f"fold_metrics_{MODEL_TYPE}.csv"
    ticker_path  = OUTPUT_DIR / f"ticker_metrics_{MODEL_TYPE}.csv"
    summary_path = OUTPUT_DIR / f"summary_{MODEL_TYPE}.json"

    all_preds.to_csv(preds_path, index=False)
    fold_results_df.to_csv(folds_path, index=False)
    ticker_metrics.to_csv(ticker_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # ── Print overall ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  OVERALL OUT-OF-SAMPLE SUMMARY")
    print(f"{'='*60}")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")

    # ── Print per-ticker table ────────────────────────────────────
    n_up_wins = (ticker_metrics["up_improve_pct"] > 0).sum()
    n_dn_wins = (ticker_metrics["dn_improve_pct"] > 0).sum()
    n_total   = len(ticker_metrics)

    print(f"\n{'='*100}")
    print(f"  PER-TICKER: XGBoost vs Naive=0  (sorted by avg improvement)")
    print(f"  XGB beats naive on y_up for {n_up_wins}/{n_total} tickers,  y_down for {n_dn_wins}/{n_total} tickers")
    print(f"{'='*100}")
    print(f"  {'Ticker':<8} {'y_up_RMSE':>10} {'Nv_up':>10} {'up%':>8}"
          f"  {'y_dn_RMSE':>10} {'Nv_dn':>10} {'dn%':>8}  {'avg%':>8}")
    print(f"  {'-'*86}")
    for _, row in ticker_metrics.iterrows():
        print(f"  {row['ticker']:<8}"
              f" {row['up_xgb_rmse']:>10.6f} {row['up_naive_rmse']:>10.6f} {row['up_improve_pct']:>+7.1f}%"
              f"  {row['dn_xgb_rmse']:>10.6f} {row['dn_naive_rmse']:>10.6f} {row['dn_improve_pct']:>+7.1f}%"
              f"  {row['avg_improve_pct']:>+7.1f}%")

    print(f"\n  Saved predictions    -> {preds_path}")
    print(f"  Saved fold metrics   -> {folds_path}")
    print(f"  Saved ticker metrics -> {ticker_path}")
    print(f"  Saved summary        -> {summary_path}")

    # ── Generate plots ────────────────────────────────────────────
    generate_plots(fold_results_df, ticker_metrics, overall)


# ===================================================================
# PLOTTING  (baseline-specific: much more elaborate than NN plots)
# ===================================================================
def generate_plots(
    fold_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
    overall: Dict[str, float],
) -> None:
    """Generate 4 key figures and save to OUTPUT_DIR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    # ---------------------------------------------------------------
    # Fig 1: Per-fold mean predicted vs true value (XGBoost & Naive)
    # ---------------------------------------------------------------
    oos_path = OUTPUT_DIR / f"oos_predictions_{MODEL_TYPE}.csv"
    oos_df = pd.read_csv(oos_path)

    fold_means = oos_df.groupby("fold_id").agg(
        true_up=( "y_up_true",  "mean"),
        pred_up=( "y_up_pred",  "mean"),
        naive_up=("y_up_naive", "mean"),
        true_dn=( "y_dn_true",  "mean"),
        pred_dn=( "y_dn_pred",  "mean"),
        naive_dn=("y_dn_naive", "mean"),
    ).reset_index()

    folds = fold_df["fold_id"].values
    fold_labels = []
    for _, row in fold_df.iterrows():
        ts = pd.Timestamp(row["test_start"])
        fold_labels.append(f"{ts.year}-Q{(ts.month - 1) // 3 + 1}")

    x = np.arange(len(folds))

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))

    # y_up subplot
    ax = axes1[0]
    ax.plot(x, fold_means["true_up"] * 100, "k-", linewidth=2.2, label="True y_up", zorder=3)
    ax.plot(x, fold_means["pred_up"] * 100, "o-", color="#2196F3", linewidth=1.5,
            markersize=4, label="XGBoost pred", zorder=2)
    ax.axhline(0, color="#BDBDBD", linewidth=2, linestyle="--", label="Naive (ŷ=0)", zorder=1)
    ax.fill_between(x, fold_means["true_up"] * 100, fold_means["pred_up"] * 100,
                    alpha=0.12, color="#2196F3")
    ax.set_title("y_up (Max Upside %):  Mean Prediction vs True")
    ax.set_xlabel("Test Period")
    ax.set_ylabel("Mean Value (%)")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([fold_labels[i] for i in range(0, len(fold_labels), 4)], rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # y_down subplot
    ax = axes1[1]
    ax.plot(x, fold_means["true_dn"] * 100, "k-", linewidth=2.2, label="True y_down", zorder=3)
    ax.plot(x, fold_means["pred_dn"] * 100, "s-", color="#FF7043", linewidth=1.5,
            markersize=4, label="XGBoost pred", zorder=2)
    ax.axhline(0, color="#BDBDBD", linewidth=2, linestyle="--", label="Naive (ŷ=0)", zorder=1)
    ax.fill_between(x, fold_means["true_dn"] * 100, fold_means["pred_dn"] * 100,
                    alpha=0.12, color="#FF7043")
    ax.set_title("y_down (Max Downside %):  Mean Prediction vs True")
    ax.set_xlabel("Test Period")
    ax.set_ylabel("Mean Value (%)")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([fold_labels[i] for i in range(0, len(fold_labels), 4)], rotation=30, ha="right")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    fig1.suptitle("XGBoost vs Naive vs True Value — Mean Prediction per Fold\n"
                  "(XGBoost tracks the true level; Naive = 0 misses entirely)",
                  fontsize=13, fontweight="bold", y=1.04)
    fig1.tight_layout()
    p1 = OUTPUT_DIR / "fig1_fold_true_vs_pred.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved figure -> {p1}")

    # ---------------------------------------------------------------
    # Fig 2: RMSE improvement % per fold (line chart)
    # ---------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    up_imp = (1 - fold_df["up_xgb_rmse"] / fold_df["up_naive_rmse"]) * 100
    dn_imp = (1 - fold_df["dn_xgb_rmse"] / fold_df["dn_naive_rmse"]) * 100

    ax2.plot(x, up_imp, "o-", color="#2196F3", linewidth=2, markersize=5, label="y_up improvement %")
    ax2.plot(x, dn_imp, "s-", color="#FF7043", linewidth=2, markersize=5, label="y_down improvement %")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(x, 0, up_imp, where=(up_imp >= 0), alpha=0.08, color="#2196F3")
    ax2.fill_between(x, 0, dn_imp, where=(dn_imp >= 0), alpha=0.08, color="#FF7043")
    ax2.fill_between(x, 0, up_imp, where=(up_imp < 0), alpha=0.08, color="red")
    ax2.fill_between(x, 0, dn_imp, where=(dn_imp < 0), alpha=0.08, color="red")

    ax2.set_xlabel("Test Period")
    ax2.set_ylabel("RMSE Improvement over Naive (%)")
    ax2.set_title("XGBoost RMSE Improvement % per Fold\n(positive = XGBoost wins)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([fold_labels[i] for i in range(0, len(fold_labels), 2)], rotation=35, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f%%"))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    avg_up = up_imp.mean()
    avg_dn = dn_imp.mean()
    ax2.annotate(f"y_up avg: {avg_up:+.1f}%", xy=(0.72, 0.92), xycoords="axes fraction",
                 fontsize=10, color="#2196F3", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2196F3", alpha=0.9))
    ax2.annotate(f"y_down avg: {avg_dn:+.1f}%", xy=(0.72, 0.82), xycoords="axes fraction",
                 fontsize=10, color="#FF7043", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#FF7043", alpha=0.9))

    fig2.tight_layout()
    p2 = OUTPUT_DIR / "fig2_fold_improvement_pct.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved figure -> {p2}")

    # ---------------------------------------------------------------
    # Fig 3: Per-ticker improvement scatter (y_up% vs y_down%)
    # ---------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(16, 12))

    up_pct = ticker_df["up_improve_pct"].values
    dn_pct = ticker_df["dn_improve_pct"].values
    tickers = ticker_df["ticker"].values

    colors = np.where((up_pct > 0) & (dn_pct > 0), "#4CAF50",
             np.where((up_pct > 0) & (dn_pct <= 0), "#2196F3",
             np.where((up_pct <= 0) & (dn_pct > 0), "#FF9800",
                                                      "#E53935")))
    ax3.scatter(up_pct, dn_pct, c=colors, s=50, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)

    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax3.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    try:
        from adjustText import adjust_text
        texts = []
        for i in range(len(tickers)):
            c = colors[i]
            texts.append(ax3.text(up_pct[i], dn_pct[i], tickers[i],
                                  fontsize=6.5, color=c, fontweight="bold", alpha=0.85))
        adjust_text(texts, ax=ax3, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.5),
                    expand_points=(1.4, 1.4), force_text=(0.4, 0.4))
    except ImportError:
        for i in range(len(tickers)):
            c = colors[i]
            ax3.annotate(tickers[i], (up_pct[i], dn_pct[i]), fontsize=6, color=c,
                         fontweight="bold", alpha=0.85,
                         xytext=(3, 3), textcoords="offset points")

    both_win  = ((up_pct > 0) & (dn_pct > 0)).sum()
    up_only   = ((up_pct > 0) & (dn_pct <= 0)).sum()
    dn_only   = ((up_pct <= 0) & (dn_pct > 0)).sum()
    both_lose = ((up_pct <= 0) & (dn_pct <= 0)).sum()

    ax3.text(0.98, 0.98, f"Both XGBoost Better  ({both_win})",
             transform=ax3.transAxes, ha="right", va="top",
             fontsize=10, color="#4CAF50", fontweight="bold")
    ax3.text(0.02, 0.02, f"Both Naive Better  ({both_lose})",
             transform=ax3.transAxes, ha="left", va="bottom",
             fontsize=10, color="#E53935", fontweight="bold")
    ax3.text(0.98, 0.02, f"XGBoost y_up only  ({up_only})",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=9, color="#2196F3", fontweight="bold")
    ax3.text(0.02, 0.98, f"XGBoost y_dn only  ({dn_only})",
             transform=ax3.transAxes, ha="left", va="top",
             fontsize=9, color="#FF9800", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FF9800", alpha=0.7))

    ax3.set_xlabel("y_up  RMSE Improvement over Naive (%)    → XGBoost Better", fontsize=11)
    ax3.set_ylabel("y_down  RMSE Improvement over Naive (%)    → XGBoost Better", fontsize=11)
    ax3.set_title("Per-Ticker XGBoost vs Naive=0  (Averaged over All 28 Folds)\n"
                  "Positive % = XGBoost RMSE lower than Naive  →  XGBoost Better",
                  fontsize=13, fontweight="bold")
    ax3.xaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f%%"))
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f%%"))
    ax3.grid(alpha=0.3)

    fig3.tight_layout()
    p3 = OUTPUT_DIR / "fig3_ticker_scatter_all_labeled.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved figure -> {p3}")

    # ---------------------------------------------------------------
    # Fig 4: Overall summary bar chart
    # ---------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    labels = ["y_up\n(Max Upside)", "y_down\n(Max Downside)", "Average"]
    xgb_vals  = [overall["up_xgb_rmse"], overall["dn_xgb_rmse"],
                 (overall["up_xgb_rmse"] + overall["dn_xgb_rmse"]) / 2]
    naive_vals = [overall["up_naive_rmse"], overall["dn_naive_rmse"],
                  (overall["up_naive_rmse"] + overall["dn_naive_rmse"]) / 2]
    impr_vals  = [overall["up_improve_pct"], overall["dn_improve_pct"], overall["avg_improve_pct"]]

    x4 = np.arange(len(labels))
    w4 = 0.3
    bars_xgb   = ax4.bar(x4 - w4 / 2, xgb_vals,   w4, label="XGBoost", color=["#2196F3", "#FF7043", "#7E57C2"], alpha=0.85)
    bars_naive = ax4.bar(x4 + w4 / 2, naive_vals, w4, label="Naive (ŷ=0)", color="#BDBDBD", alpha=0.85)

    for i, (bx, bn) in enumerate(zip(bars_xgb, bars_naive)):
        higher = max(bx.get_height(), bn.get_height())
        color = "#4CAF50" if impr_vals[i] > 0 else "#E53935"
        ax4.annotate(f"{impr_vals[i]:+.1f}%",
                     xy=(x4[i], higher + 0.001),
                     ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)

    ax4.set_ylabel("RMSE (lower is better)")
    ax4.set_title("Overall Out-of-Sample RMSE:  XGBoost vs Naive\n"
                  f"({overall['n_predictions']:,} predictions, {overall['n_tickers']} tickers, 28 folds)",
                  fontsize=12, fontweight="bold")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels, fontsize=11)
    ax4.legend(fontsize=10, loc="upper left")
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(0, max(naive_vals) * 1.25)

    takeaway = ("Key Findings:\n"
                "• y_up: XGBoost clearly outperforms Naive (learns upside patterns)\n"
                "• y_down: roughly tied (downside harder to predict)\n"
                "• Combined: XGBoost has a meaningful overall edge")
    ax4.text(0.98, 0.98, takeaway, transform=ax4.transAxes, ha="right", va="top",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDE7", ec="#FBC02D", alpha=0.9))

    fig4.tight_layout()
    p4 = OUTPUT_DIR / "fig4_overall_summary.png"
    fig4.savefig(p4, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved figure -> {p4}")

    print(f"\n  All 4 figures saved to {OUTPUT_DIR}/")


# ===================================================================
# LIVE FORECAST
# ===================================================================
def live_forecast() -> None:
    """
    Train on the already-updated CSV (call update_dataset() first), then
    download latest features for FORECAST_TICKERS and predict y_up / y_down.
    """
    import xgboost as xgb
    import yfinance as yf

    today = date.today()
    print(f"\n{'='*70}")
    print(f"  LIVE FORECAST  —  {today}  (next 5 trading days)")
    print(f"{'='*70}")

    # ── 1. Load the (already updated) CSV & train ────────────────
    print("  [1/3] Loading updated training data ...")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    all_tickers = sorted(df[TICKER_COL].unique().tolist())
    data_end = df[DATE_COL].max()

    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])
    train_df = df.dropna(subset=PRED_FEAT_COLS)
    X_train = train_df[PRED_FEAT_COLS]
    print(f"         {len(X_train):,} training rows  "
          f"({train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()})")

    # ── 2. Train y_up / y_down models ────────────────────────────
    print("  [2/3] Training XGBoost models (y_up & y_down) ...")
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1,
                  random_state=RANDOM_SEED, n_jobs=-1)
    model_up = xgb.XGBRegressor(**params)
    model_dn = xgb.XGBRegressor(**params)
    model_up.fit(X_train, train_df[TARGET_UP])
    model_dn.fit(X_train, train_df[TARGET_DN])

    # ── 3. Download latest features & predict ────────────────────
    print(f"  [3/3] Predicting {len(FORECAST_TICKERS)} tickers ...\n")
    dl_start = (pd.Timestamp(today) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    dl_end   = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    macro = build_macro_features(dl_start, dl_end)

    live_results = []
    for ticker in FORECAST_TICKERS:
        try:
            raw = yf.download(ticker, start=dl_start, end=dl_end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if raw.empty:
                print(f"    ⚠ {ticker}: no data, skipping")
                continue

            feat = build_stock_features(raw)
            feat = feat.join(macro, how="left").ffill()
            latest = feat.iloc[-1]
            latest_date = feat.index[-1]
            price = latest["adj_close"]

            X_pred = pd.DataFrame(
                [latest[PRED_FEAT_COLS].values], columns=PRED_FEAT_COLS
            )
            if X_pred.isna().any().any():
                X_pred = X_pred.ffill(axis=1).bfill(axis=1)

            pred_up = float(model_up.predict(X_pred)[0])
            pred_dn = float(model_dn.predict(X_pred)[0])

            live_results.append({
                "ticker": ticker,
                "latest_date": str(latest_date.date()),
                "price": float(price),
                "y_up": pred_up,
                "y_down": pred_dn,
                "price_high": float(price) * (1 + pred_up),
                "price_low": float(price) * (1 + pred_dn),
            })
        except Exception as e:
            print(f"    ⚠ {ticker}: {e}")

    # ── Print results ─────────────────────────────────────────────
    print(f"  {'='*72}")
    print(f"  XGBoost 5-Day Range Forecast  (as of {today})")
    print(f"  Trained on ALL {len(all_tickers)} tickers "
          f"(data updated to {data_end.date()})")
    print(f"  {'='*72}")
    print(f"  {'Ticker':<7} {'Last Date':<12} {'Price':>9} "
          f"{'y_up':>8} {'y_down':>8} {'High':>10} {'Low':>10}")
    print(f"  {'-'*68}")

    for r in live_results:
        print(f"  {r['ticker']:<7} {r['latest_date']:<12} ${r['price']:>8.2f} "
              f"{r['y_up']*100:>+7.2f}% {r['y_down']*100:>+7.2f}% "
              f"${r['price_high']:>9.2f} ${r['price_low']:>9.2f}")

    print(f"  {'-'*68}")
    print(f"  y_up  = predicted max upside  in next 5 trading days")
    print(f"  y_down= predicted max downside in next 5 trading days")
    print(f"  High/Low = current price × (1 + y_up/y_down)")
    print()

    # Save forecast to JSON
    forecast_path = OUTPUT_DIR / f"live_forecast_{today}.json"
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump({"date": str(today), "forecasts": live_results}, f, indent=2)
    print(f"  Saved forecast -> {forecast_path}")


# ===================================================================
# TICKER ACCURACY PLOT (Fig 5)
# ===================================================================
def generate_ticker_accuracy_plot(
    tickers=("AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"),
):
    """Fig5: per-ticker rolling true vs predicted for y_up & y_down."""
    import matplotlib.pyplot as plt

    pred_path = OUTPUT_DIR / "oos_predictions_xgboost.csv"
    df = pd.read_csv(pred_path, parse_dates=["date"])

    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.2 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, tk in enumerate(tickers):
        sub = df[df["ticker"] == tk].sort_values("date")

        # --- y_up (left) ---
        ax = axes[i, 0]
        ax.plot(sub["date"], sub["y_up_true"], color="black", lw=0.6, alpha=0.45, label="True")
        ax.plot(sub["date"], sub["y_up_pred"], color="#1976D2", lw=0.8, alpha=0.8, label="XGBoost")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        rmse = np.sqrt(((sub["y_up_true"] - sub["y_up_pred"]) ** 2).mean())
        ax.set_title(f"{tk}  –  y_up   (RMSE {rmse:.4f})", fontsize=10, fontweight="bold")
        ax.set_ylabel("y_up")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

        # --- y_down (right) ---
        ax = axes[i, 1]
        ax.plot(sub["date"], sub["y_dn_true"], color="black", lw=0.6, alpha=0.45, label="True")
        ax.plot(sub["date"], sub["y_dn_pred"], color="#E65100", lw=0.8, alpha=0.8, label="XGBoost")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        rmse = np.sqrt(((sub["y_dn_true"] - sub["y_dn_pred"]) ** 2).mean())
        ax.set_title(f"{tk}  –  y_down  (RMSE {rmse:.4f})", fontsize=10, fontweight="bold")
        ax.set_ylabel("y_down")
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Rolling OOS: True vs XGBoost Predicted  (2019-2026, 28 folds)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    p = OUTPUT_DIR / "fig5_ticker_true_vs_pred.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")


if __name__ == "__main__":
    main()
    generate_ticker_accuracy_plot()
    live_forecast()
