"""
common.live
===========
Shared live forecast pipeline used by all model trainers.

Provides:
  - load_and_prepare_live_data() — load CSV, build targets, scale features
  - download_latest_features()   — fetch latest Yahoo data for a ticker
  - print_live_results()         — formatted forecast table
  - save_live_forecast()         — save forecast JSON
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import (
    DATA_PATH,
    DATE_COL,
    TICKER_COL,
    TARGET_UP,
    TARGET_DN,
    PRED_FEAT_COLS,
    FORECAST_TICKERS,
)
from .data import build_updown_targets, build_stock_features, build_macro_features


# ===================================================================
# DATA LOADING FOR LIVE FORECAST
# ===================================================================
def load_and_prepare_live_data() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list, pd.Timestamp
]:
    """
    Load the updated CSV, build targets, extract & scale features.

    Returns
    -------
    train_df : DataFrame — full training data with targets
    X_all_sc : ndarray   — scaled feature matrix
    y_up_all : ndarray   — y_up target values
    y_dn_all : ndarray   — y_down target values
    scaler   : StandardScaler — fitted scaler
    all_tickers : list    — unique tickers in dataset
    data_end : Timestamp  — last date in dataset
    """
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    all_tickers = sorted(df[TICKER_COL].unique().tolist())
    data_end = df[DATE_COL].max()

    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])
    train_df = df.dropna(subset=PRED_FEAT_COLS)

    X_all = train_df[PRED_FEAT_COLS].values.astype(np.float32)
    y_up_all = train_df[TARGET_UP].values.astype(np.float32)
    y_dn_all = train_df[TARGET_DN].values.astype(np.float32)

    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)
    X_all_sc = np.nan_to_num(X_all_sc, nan=0.0)

    print(f"         {len(X_all):,} training rows  "
          f"({train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()})")

    return train_df, X_all_sc, y_up_all, y_dn_all, scaler, all_tickers, data_end


def download_latest_features(
    ticker: str,
    macro: pd.DataFrame,
    scaler: StandardScaler,
    dl_start: str,
    dl_end: str,
) -> tuple[np.ndarray, float, pd.Timestamp] | None:
    """
    Download latest data for a ticker and return scaled feature vector.

    Returns (X_scaled, price, latest_date) or None on failure.
    """
    import yfinance as yf

    try:
        raw = yf.download(
            ticker, start=dl_start, end=dl_end,
            auto_adjust=True, progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty:
            print(f"    ⚠ {ticker}: no data, skipping")
            return None

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

        X_sc = scaler.transform(X_pred.values.astype(np.float32))
        X_sc = np.nan_to_num(X_sc, nan=0.0)

        return X_sc, float(price), latest_date
    except Exception as e:
        print(f"    ⚠ {ticker}: {e}")
        return None


# ===================================================================
# RESULT PRINTING & SAVING
# ===================================================================
def print_live_results(
    live_results: List[Dict],
    model_label: str,
    all_tickers: list,
    data_end: pd.Timestamp,
    *,
    has_std: bool = False,
) -> None:
    """Pretty-print live forecast table."""
    today = date.today()

    header_cols = (f"  {'Ticker':<7} {'Last Date':<12} {'Price':>9} "
                   f"{'y_up':>8}")
    if has_std:
        header_cols += f" {'±std':>7}"
    header_cols += f" {'y_down':>8}"
    if has_std:
        header_cols += f" {'±std':>7}"
    header_cols += f" {'High':>10} {'Low':>10}"

    width = 80 if has_std else 68
    print(f"  {'='*width}")
    print(f"  {model_label} 5-Day Range Forecast  (as of {today})")
    print(f"  Trained on ALL {len(all_tickers)} tickers "
          f"(data updated to {data_end.date()})")
    print(f"  {'='*width}")
    print(header_cols)
    print(f"  {'-'*width}")

    for r in live_results:
        line = (f"  {r['ticker']:<7} {r['latest_date']:<12} ${r['price']:>8.2f} "
                f"{r['y_up']*100:>+7.2f}%")
        if has_std:
            line += f" {r['y_up_std']*100:>6.2f}%"
        line += f" {r['y_down']*100:>+7.2f}%"
        if has_std:
            line += f" {r['y_down_std']*100:>6.2f}%"
        line += f" ${r['price_high']:>9.2f} ${r['price_low']:>9.2f}"
        print(line)

    print(f"  {'-'*width}")
    print(f"  y_up  = predicted max upside  in next 5 trading days")
    print(f"  y_down= predicted max downside in next 5 trading days")
    if has_std:
        print(f"  ±std  = predictive uncertainty")
    print(f"  High/Low = current price × (1 + y_up/y_down)")
    print()


def save_live_forecast(
    output_dir: Path,
    model_name: str,
    live_results: List[Dict],
) -> None:
    """Save forecast to JSON."""
    today = date.today()
    forecast_path = output_dir / f"live_forecast_{today}.json"
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(
            {"date": str(today), "model": model_name, "forecasts": live_results},
            f, indent=2,
        )
    print(f"  Saved forecast -> {forecast_path}")
