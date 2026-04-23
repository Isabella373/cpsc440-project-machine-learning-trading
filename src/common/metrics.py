"""
common.metrics
==============
Shared metric computation used by all model trainers.

Provides:
  - rmse_mae()              — compute RMSE + MAE for a single pair
  - compute_ticker_metrics() — per-ticker RMSE/MAE for model vs naive
  - summarize_all_predictions() — overall aggregate metrics
  - improvement_pct()        — RMSE improvement percentage
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .constants import DATE_COL, TICKER_COL


# ===================================================================
# LOW-LEVEL HELPERS
# ===================================================================
def rmse_mae(y_true, y_pred) -> tuple[float, float]:
    """Return (RMSE, MAE) as plain floats."""
    return (
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
    )


def improvement_pct(model_rmse: float, naive_rmse: float) -> float:
    """RMSE improvement % of model over naive (positive = model wins)."""
    if naive_rmse <= 0:
        return 0.0
    return (1.0 - model_rmse / naive_rmse) * 100.0


# ===================================================================
# PER-TICKER METRICS
# ===================================================================
def compute_ticker_metrics(
    all_preds: pd.DataFrame,
    model_name: str,
    *,
    has_std: bool = False,
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, improvement % for model vs naive, per ticker.

    Parameters
    ----------
    all_preds : DataFrame with columns:
        y_up_true, y_up_pred, y_up_naive, y_dn_true, y_dn_pred, y_dn_naive,
        fold_id, ticker.  Optionally y_up_std, y_dn_std.
    model_name : str
        Label for column prefixes (e.g. "xgb", "fnn", "mdn", "cvae").
    has_std : bool
        If True, also report mean predictive std per ticker.

    Returns
    -------
    DataFrame sorted by avg_improve_pct descending.
    """
    rows = []
    for ticker, grp in all_preds.groupby(TICKER_COL):
        up_model_rmse, up_model_mae = rmse_mae(grp["y_up_true"], grp["y_up_pred"])
        up_naive_rmse, up_naive_mae = rmse_mae(grp["y_up_true"], grp["y_up_naive"])
        dn_model_rmse, dn_model_mae = rmse_mae(grp["y_dn_true"], grp["y_dn_pred"])
        dn_naive_rmse, dn_naive_mae = rmse_mae(grp["y_dn_true"], grp["y_dn_naive"])

        up_imp = improvement_pct(up_model_rmse, up_naive_rmse)
        dn_imp = improvement_pct(dn_model_rmse, dn_naive_rmse)

        row = {
            "ticker": ticker,
            "n_predictions": len(grp),
            "n_folds": int(grp["fold_id"].nunique()),
            f"up_{model_name}_rmse": up_model_rmse,
            f"up_{model_name}_mae": up_model_mae,
            "up_naive_rmse": up_naive_rmse,
            "up_naive_mae": up_naive_mae,
            "up_improve_pct": float(up_imp),
            f"dn_{model_name}_rmse": dn_model_rmse,
            f"dn_{model_name}_mae": dn_model_mae,
            "dn_naive_rmse": dn_naive_rmse,
            "dn_naive_mae": dn_naive_mae,
            "dn_improve_pct": float(dn_imp),
            "avg_improve_pct": float((up_imp + dn_imp) / 2),
        }
        if has_std:
            row["mean_up_std"] = float(grp["y_up_std"].mean())
            row["mean_dn_std"] = float(grp["y_dn_std"].mean())
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("avg_improve_pct", ascending=False)
        .reset_index(drop=True)
    )


# ===================================================================
# OVERALL SUMMARY
# ===================================================================
def summarize_all_predictions(
    all_preds: pd.DataFrame,
    model_name: str,
    *,
    fold_nll_values: Optional[Dict[str, List[float]]] = None,
    has_std: bool = False,
) -> Dict[str, float]:
    """
    Compute overall aggregate metrics.

    Parameters
    ----------
    all_preds : DataFrame with standard prediction columns.
    model_name : str
        Label for metric keys.
    fold_nll_values : dict, optional
        {"up": [nll_fold1, ...], "dn": [nll_fold1, ...]}
        If provided, adds averaged NLL to the summary.
    has_std : bool
        If True, adds mean predictive std to summary.
    """
    up_model_rmse, up_model_mae = rmse_mae(all_preds["y_up_true"], all_preds["y_up_pred"])
    up_naive_rmse, up_naive_mae = rmse_mae(all_preds["y_up_true"], all_preds["y_up_naive"])
    dn_model_rmse, dn_model_mae = rmse_mae(all_preds["y_dn_true"], all_preds["y_dn_pred"])
    dn_naive_rmse, dn_naive_mae = rmse_mae(all_preds["y_dn_true"], all_preds["y_dn_naive"])

    up_imp = improvement_pct(up_model_rmse, up_naive_rmse)
    dn_imp = improvement_pct(dn_model_rmse, dn_naive_rmse)

    summary: Dict[str, float] = {
        f"up_{model_name}_rmse": up_model_rmse,
        f"up_{model_name}_mae": up_model_mae,
        "up_naive_rmse": up_naive_rmse,
        "up_naive_mae": up_naive_mae,
        "up_improve_pct": float(up_imp),
    }

    if fold_nll_values and "up" in fold_nll_values:
        summary["up_avg_nll"] = float(np.mean(fold_nll_values["up"]))

    summary.update({
        f"dn_{model_name}_rmse": dn_model_rmse,
        f"dn_{model_name}_mae": dn_model_mae,
        "dn_naive_rmse": dn_naive_rmse,
        "dn_naive_mae": dn_naive_mae,
        "dn_improve_pct": float(dn_imp),
    })

    if fold_nll_values and "dn" in fold_nll_values:
        summary["dn_avg_nll"] = float(np.mean(fold_nll_values["dn"]))

    summary["avg_improve_pct"] = float((up_imp + dn_imp) / 2)

    if has_std:
        summary["mean_up_pred_std"] = float(all_preds["y_up_std"].mean())
        summary["mean_dn_pred_std"] = float(all_preds["y_dn_std"].mean())

    summary.update({
        "n_predictions": int(len(all_preds)),
        "n_dates": int(all_preds[DATE_COL].nunique()),
        "n_tickers": int(all_preds[TICKER_COL].nunique()),
    })

    return summary
