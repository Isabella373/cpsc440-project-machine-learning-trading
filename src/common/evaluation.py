"""
common.evaluation
=================
Shared evaluation orchestration: fold splitting, main loop, result saving.

Provides:
  - split_fold()          — split dataframe into train/test for a fold
  - save_results()        — save predictions, metrics, summary to disk
  - print_overall()       — pretty-print overall summary
  - print_ticker_table()  — pretty-print per-ticker table
  - run_rolling_eval()    — complete main() orchestrator
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import (
    DATA_PATH,
    DATE_COL,
    TICKER_COL,
    TARGET_UP,
    TARGET_DN,
    TRAIN_MONTHS,
    TEST_MONTHS,
    STEP_MONTHS,
    PURGE_GAP_DAYS,
)
from .data import (
    check_required_columns,
    build_rolling_windows,
    update_dataset,
    encode_and_scale,
)
from .metrics import improvement_pct


# ===================================================================
# FOLD SPLITTING
# ===================================================================
def split_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    fold_id: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/test for a single fold.

    Returns (train_df, test_df, X_train_raw, X_test_raw)
    where X_train_raw and X_test_raw are aligned feature DataFrames.
    """
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

    X_train_raw = train_df[feature_cols]
    X_test_raw  = test_df[feature_cols]

    # Align one-hot columns (test may have missing dummies)
    X_train_raw, X_test_raw = X_train_raw.align(
        X_test_raw, join="left", axis=1, fill_value=0
    )

    return train_df, test_df, X_train_raw, X_test_raw


# ===================================================================
# RESULT SAVING & PRINTING
# ===================================================================
def save_results(
    output_dir: Path,
    model_name: str,
    all_preds: pd.DataFrame,
    fold_results_df: pd.DataFrame,
    ticker_metrics: pd.DataFrame,
    overall: Dict[str, Any],
) -> None:
    """Save all result artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds.to_csv(
        output_dir / f"oos_predictions_{model_name}.csv", index=False
    )
    fold_results_df.to_csv(
        output_dir / f"fold_metrics_{model_name}.csv", index=False
    )
    ticker_metrics.to_csv(
        output_dir / f"ticker_metrics_{model_name}.csv", index=False
    )
    with open(output_dir / f"summary_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)


def print_overall(overall: Dict[str, Any], model_label: str) -> None:
    """Pretty-print overall summary."""
    print(f"\n{'='*60}")
    print(f"  {model_label} OVERALL OUT-OF-SAMPLE SUMMARY")
    print(f"{'='*60}")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")


def print_ticker_table(
    ticker_metrics: pd.DataFrame,
    model_name: str,
    model_label: str,
    *,
    has_std: bool = False,
) -> None:
    """Pretty-print per-ticker comparison table."""
    n_up = (ticker_metrics["up_improve_pct"] > 0).sum()
    n_dn = (ticker_metrics["dn_improve_pct"] > 0).sum()
    n_t  = len(ticker_metrics)

    print(f"\n{'='*100}")
    print(f"  PER-TICKER: {model_label} vs Naive=0  (sorted by avg improvement)")
    print(f"  {model_label} beats naive on y_up for {n_up}/{n_t} tickers, "
          f"y_down for {n_dn}/{n_t} tickers")
    print(f"{'='*100}")

    header = (f"  {'Ticker':<8} {'y_up_RMSE':>10} {'Nv_up':>10} {'up%':>8}"
              f"  {'y_dn_RMSE':>10} {'Nv_dn':>10} {'dn%':>8}  {'avg%':>8}")
    if has_std:
        header += f"  {'σ_up':>6} {'σ_dn':>6}"
    print(header)
    print(f"  {'-'*100}")

    up_rmse_col = f"up_{model_name}_rmse"
    dn_rmse_col = f"dn_{model_name}_rmse"

    for _, row in ticker_metrics.iterrows():
        line = (f"  {row['ticker']:<8}"
                f" {row[up_rmse_col]:>10.6f} {row['up_naive_rmse']:>10.6f} "
                f"{row['up_improve_pct']:>+7.1f}%"
                f"  {row[dn_rmse_col]:>10.6f} {row['dn_naive_rmse']:>10.6f} "
                f"{row['dn_improve_pct']:>+7.1f}%"
                f"  {row['avg_improve_pct']:>+7.1f}%")
        if has_std:
            line += f"  {row['mean_up_std']:>6.4f} {row['mean_dn_std']:>6.4f}"
        print(line)


def print_saved_paths(output_dir: Path, model_name: str) -> None:
    """Print paths of saved files."""
    print(f"\n  Saved predictions    -> {output_dir / f'oos_predictions_{model_name}.csv'}")
    print(f"  Saved fold metrics   -> {output_dir / f'fold_metrics_{model_name}.csv'}")
    print(f"  Saved ticker metrics -> {output_dir / f'ticker_metrics_{model_name}.csv'}")
    print(f"  Saved summary        -> {output_dir / f'summary_{model_name}.json'}")
