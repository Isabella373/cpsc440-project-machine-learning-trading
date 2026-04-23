"""
common.data
============
Shared data loading, preparation, feature scaling, and target preprocessing.

Provides:
  - check_required_columns()  — validate input dataframe
  - build_updown_targets()    — construct y_up / y_down from adj_close
  - prepare_data_nn()         — full data prep for NN models (one-hot encode)
  - infer_feature_columns()   — list numeric feature columns
  - encode_and_scale()        — StandardScaler fit/transform
  - robust_standardize()      — median/IQR target standardization
  - inverse_standardize()     — undo robust_standardize
  - build_rolling_windows()   — rolling train/test window construction
  - update_dataset()          — auto-update CSV from Yahoo Finance
  - build_stock_features()    — per-ticker OHLCV → technical features
  - build_macro_features()    — download and build macro features
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import (
    DATE_COL,
    TICKER_COL,
    PRICE_COL,
    TARGET_UP,
    TARGET_DN,
    WINSORIZE_PCT,
    DATA_PATH,
    EXCLUDE_COLS_NN,
)

# For sector labels (used by update_dataset)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))
from ticker_candidates import SUBSECTOR_MAP  # noqa: E402

warnings.filterwarnings("ignore")


# ===================================================================
# VALIDATION
# ===================================================================
def check_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if essential columns are missing."""
    needed = [DATE_COL, TICKER_COL, PRICE_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ===================================================================
# TARGET CONSTRUCTION
# ===================================================================
def build_updown_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Construct y_up and y_down per ticker from adj_close.

    y_up   = max(P_{t+1:t+5}) / P_t - 1   (max upside over next 5 days)
    y_down = min(P_{t+1:t+5}) / P_t - 1   (max downside over next 5 days)
    """
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    y_up_list, y_dn_list = [], []

    for _ticker, grp in df.groupby(TICKER_COL, sort=False):
        p = grp[PRICE_COL].values
        n = len(p)
        y_up = np.full(n, np.nan)
        y_dn = np.full(n, np.nan)
        for i in range(n - 5):
            fwd = p[i + 1 : i + 6]
            y_up[i] = fwd.max() / p[i] - 1
            y_dn[i] = fwd.min() / p[i] - 1
        y_up_list.append(pd.Series(y_up, index=grp.index))
        y_dn_list.append(pd.Series(y_dn, index=grp.index))

    df[TARGET_UP] = pd.concat(y_up_list)
    df[TARGET_DN] = pd.concat(y_dn_list)
    return df


# ===================================================================
# DATA PREPARATION
# ===================================================================
def infer_feature_columns(
    df: pd.DataFrame,
    exclude_cols: set | None = None,
) -> List[str]:
    """Return all columns except excluded ones."""
    excl = exclude_cols or EXCLUDE_COLS_NN
    return [c for c in df.columns if c not in excl]


def prepare_data_nn(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for NN models: parse dates, build targets, one-hot encode.

    Used by FNN, MDN, and CVAE trainers.  The XGBoost baseline has its
    own prepare_data() because it handles categoricals differently.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    print("  Building y_up / y_down targets ...")
    df = build_updown_targets(df)
    df = df.dropna(subset=[TARGET_UP, TARGET_DN])
    print(f"  y_up  mean={df[TARGET_UP].mean():.6f}  std={df[TARGET_UP].std():.6f}")
    print(f"  y_down mean={df[TARGET_DN].mean():.6f}  std={df[TARGET_DN].std():.6f}")

    # Save ticker before one-hot encoding consumes it
    saved_ticker = df[TICKER_COL].copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_COLS_NN]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    df[TICKER_COL] = saved_ticker.values

    return df


# ===================================================================
# FEATURE SCALING
# ===================================================================
def encode_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, transform both. Fill NaN with 0."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.values.astype(np.float32))
    Xte = scaler.transform(X_test.values.astype(np.float32))
    Xtr = np.nan_to_num(Xtr, nan=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0)
    return Xtr, Xte, scaler


# ===================================================================
# TARGET PREPROCESSING (robust: median / IQR)
# ===================================================================
def robust_standardize(
    y_train: np.ndarray,
    winsorize_pct: tuple[float, float] = WINSORIZE_PCT,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Robust target preprocessing:
      1. Compute clip bounds at winsorize_pct percentiles
      2. Standardize using median / IQR (robust to outliers)

    Preserves extreme training targets so the model can learn tail behaviour.

    Returns (y_standardized, clip_lo, clip_hi, y_center, y_scale)
    """
    p_lo, p_hi = winsorize_pct
    clip_lo = float(np.percentile(y_train, p_lo))
    clip_hi = float(np.percentile(y_train, p_hi))

    y_center = float(np.median(y_train))
    q75, q25 = np.percentile(y_train, [75, 25])
    iqr = float(q75 - q25)
    y_scale = iqr / 1.3489 + 1e-8

    y_out = (y_train - y_center) / y_scale
    return y_out, clip_lo, clip_hi, y_center, y_scale


def inverse_standardize(
    y: np.ndarray,
    y_center: float,
    y_scale: float,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> np.ndarray:
    """Undo robust standardization, optionally clip to training bounds."""
    y_raw = y * y_scale + y_center
    if clip_lo is not None and clip_hi is not None:
        y_raw = np.clip(y_raw, clip_lo, clip_hi)
    return y_raw


# ===================================================================
# ROLLING WINDOWS
# ===================================================================
def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def build_rolling_windows(
    all_dates: pd.Series,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_gap_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Returns list of (train_start, train_end, test_start, test_end)."""
    min_date = pd.Timestamp(all_dates.min())
    max_date = pd.Timestamp(all_dates.max())

    first_train_start = _month_floor(min_date)
    windows: list = []
    current = first_train_start

    while True:
        train_end_excl = current + pd.DateOffset(months=train_months)
        train_end      = train_end_excl - pd.Timedelta(days=1)

        test_start     = train_end_excl + pd.Timedelta(days=purge_gap_days)
        test_end_excl  = _month_floor(test_start) + pd.DateOffset(months=test_months)
        test_end       = test_end_excl - pd.Timedelta(days=1)

        if test_start > max_date or train_end > max_date:
            break

        windows.append((current, train_end, test_start, test_end))
        current += pd.DateOffset(months=step_months)

        if current > max_date:
            break

    return windows


# ===================================================================
# YAHOO FINANCE HELPERS (used by live forecast + dataset update)
# ===================================================================
def build_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features from a single-ticker OHLCV DataFrame."""
    close, volume = df["Close"], df["Volume"]
    high, low, opn = df["High"], df["Low"], df["Open"]
    feat = pd.DataFrame(index=df.index)
    feat["adj_close"] = close
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_20d"] = close.pct_change(20)
    log_ret = np.log(close / close.shift(1))
    feat["vol_20d"] = log_ret.rolling(20).std()
    feat["skew_20d"] = log_ret.rolling(20).skew()
    feat["kurt_20d"] = log_ret.rolling(20).kurt()
    feat["momentum_20d"] = close.pct_change(20)
    feat["momentum_60d"] = close.pct_change(60)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    feat["volume_zscore"] = (volume - vol_mean) / vol_std
    feat["hl_spread"] = (high - low) / close
    feat["oc_return"] = (close - opn) / opn
    return feat


def build_macro_features(start: str, end: str) -> pd.DataFrame:
    """Download macro data and build features."""
    import yfinance as yf

    def _dl(ticker):
        d = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
        return d

    vix = _dl("^VIX")[["Close"]].rename(columns={"Close": "vix"})
    vix["vix_change_9d"] = vix["vix"] - vix["vix"].shift(9)
    vix["vix_ma20"] = vix["vix"].rolling(20).mean()
    vix["vix_slope"] = (vix["vix"] - vix["vix"].shift(5)) / 5

    macro = vix[["vix", "vix_change_9d", "vix_ma20", "vix_slope"]]
    macro = macro.join(
        _dl("^VVIX")[["Close"]].rename(columns={"Close": "vvix"}), how="outer"
    )

    for tkr, col, div in [("^IRX", "bond_yield_3m", 100), ("^FVX", "bond_yield_5y", 100)]:
        tmp = _dl(tkr)[["Close"]].rename(columns={"Close": col})
        tmp[col] = tmp[col] / div
        macro = macro.join(tmp[col], how="outer")

    for tkr, col in [("GLD", "gld_ret"), ("CL=F", "wti_ret"),
                      ("TLT", "tlt_ret"), ("IEF", "ief_ret")]:
        tmp = _dl(tkr)[["Close"]]
        macro = macro.join(tmp["Close"].pct_change().rename(col), how="outer")

    dxy = _dl("DX-Y.NYB")[["Close"]].rename(columns={"Close": "dxy"})
    dxy["dxy_ret_1d"] = dxy["dxy"].pct_change()
    dxy["dxy_ma20"] = dxy["dxy"].rolling(20).mean()
    macro = macro.join(dxy[["dxy", "dxy_ret_1d", "dxy_ma20"]], how="outer")

    return macro.ffill()


# ===================================================================
# AUTO-UPDATE DATASET
# ===================================================================
def update_dataset() -> None:
    """
    Read the existing CSV, download new OHLCV + macro data for ALL training
    tickers from Yahoo Finance up to today, rebuild per-ticker features and
    macro features for the new rows, append them, and overwrite the CSV.
    """
    import yfinance as yf

    today = date.today()
    print(f"\n{'='*60}")
    print(f"  AUTO-UPDATE DATASET  —  fetching data up to {today}")
    print(f"{'='*60}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {DATA_PATH}\n"
            "  Run  python src/data_processing/build_dataset.py  first."
        )

    # ── 1. Load existing data ─────────────────────────────────────
    orig = pd.read_csv(DATA_PATH)
    orig[DATE_COL] = pd.to_datetime(orig[DATE_COL])
    orig = orig.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    orig_end = orig[DATE_COL].max()
    all_tickers = sorted(orig[TICKER_COL].unique().tolist())
    print(f"  Existing CSV : {len(orig):,} rows, {len(all_tickers)} tickers, "
          f"ends {orig_end.date()}")

    today_ts = pd.Timestamp(today)
    if (today_ts - orig_end).days <= 1:
        print("  ✓ Data is already up-to-date, skipping download.")
        return

    # ── 2. Download fresh OHLCV for ALL tickers ───────────────────
    dl_start = (orig_end - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    dl_end   = (today_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  Downloading {len(all_tickers)} tickers "
          f"({dl_start} → {today}) ...")

    raw_all = yf.download(
        all_tickers, start=dl_start, end=dl_end,
        auto_adjust=True, progress=False, threads=True, group_by="column",
    )

    # ── 3. Download macro data ────────────────────────────────────
    print("  Downloading macro data ...")
    macro = build_macro_features(dl_start, dl_end)

    # ── 4. Build per-ticker features for NEW rows ─────────────────
    new_rows = []
    for ticker in all_tickers:
        try:
            if isinstance(raw_all.columns, pd.MultiIndex):
                tk_df = raw_all.xs(ticker, level=1, axis=1).copy()
            else:
                tk_df = raw_all.copy()
            tk_df = tk_df.dropna(subset=["Close"])
            if tk_df.empty:
                continue

            close  = tk_df["Close"]
            high   = tk_df["High"]
            low    = tk_df["Low"]
            opn    = tk_df["Open"]
            volume = tk_df["Volume"]

            feat = pd.DataFrame(index=tk_df.index)
            feat["open"]      = opn
            feat["high"]      = high
            feat["low"]       = low
            feat["adj_close"] = close
            feat["volume"]    = volume

            feat["ret_1d"]  = close.pct_change(1)
            feat["ret_5d"]  = close.pct_change(5)
            feat["ret_20d"] = close.pct_change(20)
            log_ret = np.log(close / close.shift(1))
            feat["vol_20d"]  = log_ret.rolling(20).std()
            feat["skew_20d"] = log_ret.rolling(20).skew()
            feat["kurt_20d"] = log_ret.rolling(20).kurt()
            feat["momentum_20d"] = close / close.shift(20) - 1
            feat["momentum_60d"] = close / close.shift(60) - 1
            vol_mean = volume.rolling(20).mean()
            vol_std  = volume.rolling(20).std()
            feat["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-9)
            feat["dollar_volume"] = close * volume
            feat["hl_spread"] = (high - low) / close
            feat["oc_return"] = (close - opn) / opn
            feat["sector"]    = SUBSECTOR_MAP.get(ticker, "Other")
            feat["target_ret_5d"] = np.log(close.shift(-5) / close)

            feat = feat.join(macro, how="left").ffill()
            feat[TICKER_COL] = ticker
            feat[DATE_COL]   = feat.index

            feat = feat[feat[DATE_COL] > orig_end]
            if not feat.empty:
                new_rows.append(feat.reset_index(drop=True))
        except Exception:
            continue

    if not new_rows:
        print("  ✓ No new trading days beyond existing data.")
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    n_new_dates = new_df[DATE_COL].nunique()
    print(f"  Fetched {len(new_df):,} new rows "
          f"({n_new_dates} new trading days)")

    # ── 5. Append & save ──────────────────────────────────────────
    orig_cols = orig.columns.tolist()
    for col in orig_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[orig_cols]

    required_feats = [
        "ret_1d", "ret_5d", "ret_20d", "vol_20d",
        "momentum_20d", "momentum_60d",
    ]
    n0 = len(new_df)
    new_df = new_df.dropna(subset=required_feats)
    if n0 != len(new_df):
        print(f"  Dropped {n0 - len(new_df)} warm-up rows from new data")

    if new_df.empty:
        print("  ✓ No new complete rows after warm-up filtering.")
        return

    merged = pd.concat([orig, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=[DATE_COL, TICKER_COL], keep="last")
    merged = merged.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    merged.to_csv(DATA_PATH, index=False)
    new_end = merged[DATE_COL].max()
    print(f"  ✓ Updated CSV: {len(merged):,} rows "
          f"({orig_end.date()} → {pd.Timestamp(new_end).date()})")
    print(f"  Saved -> {DATA_PATH}")
