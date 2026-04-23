"""
common.plotting
===============
Shared plotting functions for NN model trainers (FNN, MDN, CVAE).

Provides:
  - plot_fold_rmse()         — Fig 1: RMSE per fold (model vs naive)
  - plot_fold_improvement()  — Fig 2: Improvement % bars per fold
  - plot_ticker_scatter()    — Fig 3: Per-ticker scatter
  - plot_overall_bar()       — Fig 4: Overall summary bar chart
  - plot_fold_nll()          — Fig 5: NLL per fold (MDN/CVAE)
  - plot_ticker_accuracy()   — Fig N: Per-ticker true vs predicted

Note: The baseline (XGBoost) has a substantially different, more
elaborate plotting style and is NOT covered by these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def _setup_mpl():
    """Import matplotlib with Agg backend and return (plt, mtick)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    return plt, mtick


# ── Fig 1 — Fold RMSE ────────────────────────────────────────────
def plot_fold_rmse(
    fold_df: pd.DataFrame,
    model_label: str,
    model_prefix: str,
    output_dir: Path,
    *,
    model_color: str = "tab:blue",
) -> None:
    """
    Dual-panel line plot of model RMSE vs naive RMSE per fold.

    Parameters
    ----------
    fold_df      : DataFrame with columns up_{prefix}_rmse, dn_{prefix}_rmse,
                   up_naive_rmse, dn_naive_rmse.
    model_label  : Display name, e.g. "FNN", "MDN", "CVAE".
    model_prefix : Column prefix, e.g. "fnn", "mdn", "cvae".
    output_dir   : Directory to save fig1_fold_rmse.png.
    model_color  : Colour for the model line.
    """
    plt, _ = _setup_mpl()
    n_folds = len(fold_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model_col, naive_col, label in [
        (axes[0], f"up_{model_prefix}_rmse", "up_naive_rmse",
         "y_up (max upside)"),
        (axes[1], f"dn_{model_prefix}_rmse", "dn_naive_rmse",
         "y_down (max downside)"),
    ]:
        x = range(1, n_folds + 1)
        ax.plot(x, fold_df[model_col], "o-", ms=4,
                label=model_label, color=model_color)
        ax.plot(x, fold_df[naive_col], "s--", ms=3,
                label="Naive=0", color="tab:red", alpha=0.6)
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{model_label} Rolling OOS: RMSE per Fold  ({n_folds} folds)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_fold_rmse.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Fig 2 — Fold improvement % ───────────────────────────────────
def plot_fold_improvement(
    fold_df: pd.DataFrame,
    model_label: str,
    model_prefix: str,
    overall: Dict[str, float],
    output_dir: Path,
) -> None:
    """Grouped bar chart of per-fold RMSE improvement %."""
    plt, mtick = _setup_mpl()

    fold_df = fold_df.copy()
    fold_df["up_imp"] = (
        1 - fold_df[f"up_{model_prefix}_rmse"] / fold_df["up_naive_rmse"]
    ) * 100
    fold_df["dn_imp"] = (
        1 - fold_df[f"dn_{model_prefix}_rmse"] / fold_df["dn_naive_rmse"]
    ) * 100

    n_folds = len(fold_df)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(1, n_folds + 1)
    ax.bar([i - 0.17 for i in x], fold_df["up_imp"], width=0.34,
           label="y_up", color="tab:green")
    ax.bar([i + 0.17 for i in x], fold_df["dn_imp"], width=0.34,
           label="y_down", color="tab:orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE Improvement vs Naive (%)")
    ax.set_title(
        f"{model_label}: Per-Fold Improvement  "
        f"(avg={overall['avg_improve_pct']:+.1f}%)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f%%"))
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_fold_improvement.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Fig 3 — Ticker scatter ───────────────────────────────────────
def plot_ticker_scatter(
    ticker_df: pd.DataFrame,
    model_label: str,
    output_dir: Path,
    *,
    scatter_color: str = "tab:blue",
) -> None:
    """Scatter plot of per-ticker improvement (y_up % vs y_down %)."""
    plt, _ = _setup_mpl()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        ticker_df["up_improve_pct"], ticker_df["dn_improve_pct"],
        s=20, alpha=0.7, c=scatter_color,
    )
    for _, row in ticker_df.iterrows():
        ax.annotate(
            row["ticker"],
            (row["up_improve_pct"], row["dn_improve_pct"]),
            fontsize=5, alpha=0.7,
        )
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("y_up improvement %")
    ax.set_ylabel("y_down improvement %")
    ax.set_title(f"{model_label}: Per-Ticker Improvement vs Naive")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_ticker_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Fig 4 — Overall bar chart ────────────────────────────────────
def plot_overall_bar(
    overall: Dict[str, float],
    model_label: str,
    output_dir: Path,
) -> None:
    """Bar chart of overall RMSE improvement (y_up, y_down, avg)."""
    plt, _ = _setup_mpl()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ["y_up", "y_down", "average"]
    vals = [
        overall["up_improve_pct"],
        overall["dn_improve_pct"],
        overall["avg_improve_pct"],
    ]
    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]
    ax.bar(bars, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("RMSE Improvement vs Naive (%)")
    ax.set_title(f"{model_label} Overall: RMSE Improvement vs Naive=0")
    for i, v in enumerate(vals):
        ax.text(
            i, v + (0.3 if v > 0 else -0.6),
            f"{v:+.1f}%", ha="center", fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_overall_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Fig 5 — NLL per fold (MDN/CVAE) ──────────────────────────────
def plot_fold_nll(
    fold_df: pd.DataFrame,
    model_label: str,
    model_prefix: str,
    output_dir: Path,
) -> None:
    """Line plot of per-fold NLL for models with uncertainty output."""
    plt, _ = _setup_mpl()
    n_folds = len(fold_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(1, n_folds + 1)
    ax.plot(x, fold_df[f"up_{model_prefix}_nll"], "o-", ms=4,
            label="y_up NLL", color="tab:blue")
    ax.plot(x, fold_df[f"dn_{model_prefix}_nll"], "s-", ms=4,
            label="y_down NLL", color="tab:red")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Mean NLL")
    ax.set_title(f"{model_label}: Negative Log-Likelihood per Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_fold_nll.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── Ticker accuracy (true vs predicted per ticker) ────────────────
def plot_ticker_accuracy(
    output_dir: Path,
    model_name: str,
    model_label: str,
    fig_num: int,
    tickers: Sequence[str] = (
        "AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK",
    ),
    *,
    up_color: str = "#1976D2",
    dn_color: str = "#E65100",
    has_std: bool = False,
) -> None:
    """
    Per-ticker rolling true vs predicted for y_up & y_down.

    Parameters
    ----------
    output_dir   : Results directory.
    model_name   : File stem, e.g. "fnn", "mdn", "cvae".
    model_label  : Display label, e.g. "FNN", "MDN", "CVAE".
    fig_num      : Figure number for the filename, e.g. 5 → fig5_...
    tickers      : Ticker symbols to plot.
    up_color     : Colour for y_up prediction line.
    dn_color     : Colour for y_down prediction line.
    has_std      : If True, plot ±1σ uncertainty bands.
    """
    plt, _ = _setup_mpl()

    pred_path = output_dir / f"oos_predictions_{model_name}.csv"
    if not pred_path.exists():
        print(f"  ⚠ {pred_path} not found, skipping fig{fig_num}")
        return
    df = pd.read_csv(pred_path, parse_dates=["date"])

    # Clip extreme std values (e.g. COVID-era sigma blowup)
    if has_std:
        STD_CAP = 1.0
        for std_col in ("y_up_std", "y_dn_std"):
            if std_col in df.columns:
                df[std_col] = (
                    df[std_col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(STD_CAP)
                    .clip(upper=STD_CAP)
                )

    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.2 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, tk in enumerate(tickers):
        sub = df[df["ticker"] == tk].sort_values("date")
        if sub.empty:
            for ax in (axes[i, 0], axes[i, 1]):
                ax.set_title(f"{tk}  –  no data")
            continue

        # --- y_up (left) ---
        ax = axes[i, 0]
        ax.plot(sub["date"], sub["y_up_true"], color="black",
                lw=0.6, alpha=0.45, label="True")
        ax.plot(sub["date"], sub["y_up_pred"], color=up_color,
                lw=0.8, alpha=0.8, label=model_label)
        if has_std and "y_up_std" in sub.columns:
            ax.fill_between(
                sub["date"],
                sub["y_up_pred"] - sub["y_up_std"],
                sub["y_up_pred"] + sub["y_up_std"],
                alpha=0.15, color=up_color, label="±1σ",
            )
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        rmse = np.sqrt(
            ((sub["y_up_true"] - sub["y_up_pred"]) ** 2).mean()
        )
        ax.set_title(f"{tk}  –  y_up   (RMSE {rmse:.4f})",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("y_up")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

        # --- y_down (right) ---
        ax = axes[i, 1]
        ax.plot(sub["date"], sub["y_dn_true"], color="black",
                lw=0.6, alpha=0.45, label="True")
        ax.plot(sub["date"], sub["y_dn_pred"], color=dn_color,
                lw=0.8, alpha=0.8, label=model_label)
        if has_std and "y_dn_std" in sub.columns:
            ax.fill_between(
                sub["date"],
                sub["y_dn_pred"] - sub["y_dn_std"],
                sub["y_dn_pred"] + sub["y_dn_std"],
                alpha=0.15, color=dn_color, label="±1σ",
            )
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        rmse = np.sqrt(
            ((sub["y_dn_true"] - sub["y_dn_pred"]) ** 2).mean()
        )
        ax.set_title(f"{tk}  –  y_down  (RMSE {rmse:.4f})",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("y_down")
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Rolling OOS: True vs {model_label} Predicted  (2019-2026)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    p = output_dir / f"fig{fig_num}_ticker_true_vs_pred.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")
