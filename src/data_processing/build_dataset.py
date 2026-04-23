"""
build_dataset.py
================
End-to-end pipeline: download -> engineer features -> merge -> single CSV.

Output
------
data/processed/dataset_final.csv

Run
---
    cd <project_root>
    python src/data_processing/build_dataset.py
"""

# -- Imports ----------------------------------------------------------------
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar

from ticker_candidates import SP500_CANDIDATES, SUBSECTOR_MAP

warnings.filterwarnings("ignore")

# Optional: pandas_datareader for FRED data (BLS macro series).
# If not installed the pipeline still runs; BLS columns will be absent.
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

# -- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants --------------------------------------------------------------
START = "2018-01-02"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")   # always up-to-date
SECTOR_MAP = SUBSECTOR_MAP


# ===========================================================================
#  STEP 1 -- Stock Universe
# ===========================================================================

def _ticker_hash(tickers: list[str], length: int = 8) -> str:
    """Short deterministic hash of a ticker list (for cache-key safety)."""
    return hashlib.md5("_".join(sorted(tickers)).encode()).hexdigest()[:length]


def get_universe(
    n: int = 100,
    lookback_start: str = "2018-03-10",
    lookback_end: str = "2023-03-10",
) -> list[str]:
    """
    Select top-*n* tickers from SP500_CANDIDATES by 5-year average
    daily dollar volume (Close x Volume).
    Results are cached so the download only happens once.
    """
    print(f"\n{'='*60}")
    print("STEP 1 -- Selecting universe by 5-year avg dollar volume")
    print(f"{'='*60}")

    unique_tickers = list(dict.fromkeys(SP500_CANDIDATES))
    print(f"  Candidate pool: {len(unique_tickers)} tickers")

    thash = _ticker_hash(unique_tickers)
    cache = RAW_DIR / f"universe_liquidity_{lookback_start}_{lookback_end}_{thash}.csv"

    if cache.exists():
        print(f"  Loading cached ranking from {cache}")
        ranking = pd.read_csv(cache)
    else:
        print(f"  Downloading {lookback_start} -> {lookback_end} for ranking ...")
        raw = yf.download(
            unique_tickers,
            start=lookback_start,
            end=pd.Timestamp(lookback_end) + pd.Timedelta(days=1),
            auto_adjust=True, progress=True, threads=True,
            group_by="column",
        )
        close_df = raw["Close"]
        vol_df   = raw["Volume"]
        dollar_vol = close_df * vol_df

        # Require >= 80% non-NaN coverage to avoid tickers that listed late
        coverage = dollar_vol.notna().mean()
        avg_dv   = dollar_vol.mean(skipna=True)
        avg_dv   = avg_dv[coverage >= 0.8]

        ranking = (
            avg_dv
            .rename_axis("ticker")
            .reset_index(name="avg_dollar_volume")
            .sort_values("avg_dollar_volume", ascending=False)
            .reset_index(drop=True)
        )
        ranking.to_csv(cache, index=False)
        print(f"  Saved ranking -> {cache}")

    selected = ranking["ticker"].head(n).tolist()
    print(f"  Selected {len(selected)} tickers.  Top-10: {selected[:10]}")
    return selected


# ===========================================================================
#  STEP 2 -- OHLCV Download
# ===========================================================================

def download_ohlcv(tickers: list[str]) -> pd.DataFrame:
    """Download daily OHLCV and return a long-format DataFrame."""
    print(f"\n{'='*60}")
    print("STEP 2 -- Downloading OHLCV")
    print(f"{'='*60}")

    tickers_sorted = sorted(tickers)
    thash = _ticker_hash(tickers_sorted)
    cache = RAW_DIR / f"ohlcv_{START}_{END}_{thash}.parquet"

    # Re-download if cache is missing OR was created on a previous day
    use_cache = False
    if cache.exists():
        import datetime as _dt
        cache_date = _dt.date.fromtimestamp(cache.stat().st_mtime)
        if cache_date >= _dt.date.today():
            use_cache = True

    if use_cache:
        print(f"  Loading today's cached OHLCV from {cache}")
        raw = pd.read_parquet(cache)
    else:
        print(f"  Downloading {len(tickers_sorted)} tickers ({START} -> {END}) ...")
        raw = yf.download(
            tickers_sorted,
            start=START,
            end=pd.Timestamp(END) + pd.Timedelta(days=1),
            auto_adjust=True, progress=True, threads=True,
            group_by="column",
        )
        raw.to_parquet(cache)
        print(f"  Saved -> {cache}")

    # Wide -> long pivot
    fields = ["Open", "High", "Low", "Close", "Volume"]
    frames = []
    for fld in fields:
        tmp = (
            raw[fld]
            .rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name=fld.lower())
        )
        tmp = tmp[tmp["ticker"].isin(tickers_sorted)]
        frames.append(tmp.set_index(["date", "ticker"]))

    df = pd.concat(frames, axis=1).reset_index()
    df = df.dropna(subset=["close"])
    df["adj_close"] = df["close"]           # auto_adjust=True -> already adjusted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"  Rows: {len(df):,}   Tickers: {df['ticker'].nunique()}")
    return df


# ===========================================================================
#  STEP 3 -- Price-Volume Feature Engineering  (per ticker)
# ===========================================================================

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-ticker price/volume features + target."""
    print(f"\n{'='*60}")
    print("STEP 3 -- Price-volume feature engineering")
    print(f"{'='*60}")

    results = []
    for ticker, grp in df.groupby("ticker"):
        g = grp.sort_values("date").copy()
        p = g["adj_close"]

        # Returns
        g["ret_1d"]  = p.pct_change(1)
        g["ret_5d"]  = p.pct_change(5)
        g["ret_20d"] = p.pct_change(20)

        # Volatility / higher moments (on log returns)
        log_ret = np.log(p / p.shift(1))
        g["vol_20d"]  = log_ret.rolling(20).std()
        g["skew_20d"] = log_ret.rolling(20).skew()
        g["kurt_20d"] = log_ret.rolling(20).kurt()

        # Momentum
        g["momentum_20d"] = p / p.shift(20) - 1
        g["momentum_60d"] = p / p.shift(60) - 1

        # NOTE: rsi_14, macd, ma_ratio_5, ma_ratio_20 removed after
        # group-ablation experiment showed they are redundant with
        # return/momentum features (Δ = +0.14% when dropped).

        # Volume features
        vol_mean = g["volume"].rolling(20).mean()
        vol_std  = g["volume"].rolling(20).std()
        g["volume_zscore"] = (g["volume"] - vol_mean) / (vol_std + 1e-9)
        g["dollar_volume"] = g["adj_close"] * g["volume"]

        # Intraday spread / return (normalised by adj_close for consistency)
        g["hl_spread"] = (g["high"] - g["low"]) / g["adj_close"]
        g["oc_return"] = (g["adj_close"] - g["open"]) / g["open"]

        # Sector label
        g["sector"] = SECTOR_MAP.get(ticker, "Other")

        # Target: 5-day forward log return
        g["target_ret_5d"] = np.log(p.shift(-5) / p)

        results.append(g)

    out = pd.concat(results, ignore_index=True)
    print(f"  Shape: {out.shape}")
    return out


# ===========================================================================
#  STEP 4 -- Macro / Regime Data
# ===========================================================================

def _yf_close(ticker: str) -> pd.Series:
    """Download a single ticker's Close via yfinance (with consistent END)."""
    raw = yf.download(
        ticker, start=START,
        end=pd.Timestamp(END) + pd.Timedelta(days=1),
        auto_adjust=True, progress=False,
    )
    close = raw["Close"]
    # New yfinance returns a DataFrame with ticker as column; squeeze to Series
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close.index = pd.to_datetime(close.index)
    return close


def _download_vix() -> pd.DataFrame:
    """VIX level, 9-day change, VVIX, MA-20, slope."""
    print("  . VIX / VVIX")
    vix  = _yf_close("^VIX").rename("vix")
    vvix = _yf_close("^VVIX").rename("vvix")

    df = pd.concat([vix, vvix], axis=1)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    df["vix_change_9d"] = df["vix"].diff(9)
    df["vix_ma20"]      = df["vix"].rolling(20).mean()
    df["vix_slope"]     = df["vix"].diff(5) / 5
    return df


def _download_bond_yields() -> pd.DataFrame:
    """^IRX (~ 3-month T-bill) and ^FVX (5-year note) as yield proxies."""
    print("  . Treasury yields (^IRX, ^FVX)")
    t3m = _yf_close("^IRX").rename("bond_yield_3m")
    t5y = _yf_close("^FVX").rename("bond_yield_5y")

    df = pd.concat([t3m, t5y], axis=1)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _download_cross_assets() -> pd.DataFrame:
    """Gold (GLD), Oil (CL=F), Long bond (TLT), Mid bond (IEF) -- log returns."""
    print("  . Cross-asset (GLD, CL=F, TLT, IEF)")
    mapping = {"GLD": "gld_ret", "CL=F": "wti_ret", "TLT": "tlt_ret", "IEF": "ief_ret"}
    frames = []
    for tkr, col in mapping.items():
        try:
            close = _yf_close(tkr)
            frames.append(np.log(close / close.shift(1)).rename(col))
        except Exception as e:
            print(f"    WARNING: {tkr} failed: {e}")
    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _download_dxy() -> pd.DataFrame:
    """USD Dollar Index (DX-Y.NYB) + 1-day return + MA-20."""
    print("  . DXY")
    dxy = _yf_close("DX-Y.NYB").rename("dxy")

    df = dxy.to_frame()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["dxy_ret_1d"] = np.log(df["dxy"] / df["dxy"].shift(1))
    df["dxy_ma20"]   = df["dxy"].rolling(20).mean()
    return df


def _download_bls_macro():
    """FRED series: NFP (PAYEMS), Unemployment (UNRATE), Initial Claims (ICSA).
    Returns None if pandas_datareader is not installed.
    """
    if not HAS_PDR:
        print("  . BLS/FRED -- SKIPPED (install pandas-datareader to enable)")
        return None

    print("  . BLS/FRED (PAYEMS, UNRATE, ICSA)")
    series = {"PAYEMS": "nfp", "UNRATE": "unrate", "ICSA": "initial_claims"}
    frames = []
    for code, col in series.items():
        try:
            s = pdr.DataReader(code, "fred", START, END).rename(columns={code: col})
            frames.append(s)
        except Exception as e:
            print(f"    WARNING: {code} failed: {e}")

    if not frames:
        return None
    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_macro_df() -> pd.DataFrame:
    """Merge all macro sources into one daily DataFrame (ffill over gaps)."""
    print(f"\n{'='*60}")
    print("STEP 4 -- Macro / regime data")
    print(f"{'='*60}")

    parts = [
        _download_vix(),
        _download_bond_yields(),
        _download_cross_assets(),
        _download_dxy(),
    ]

    bls = _download_bls_macro()
    if bls is not None:
        parts.append(bls)

    macro = parts[0]
    for p in parts[1:]:
        macro = macro.merge(p, on="date", how="outer")

    macro = macro.sort_values("date").reset_index(drop=True)
    macro = macro.set_index("date").ffill().bfill().reset_index()
    print(f"  Macro shape: {macro.shape}")
    return macro


# ===========================================================================
#  STEP 5 -- Calendar Features
# ===========================================================================

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Third Friday of a given month (monthly options expiration day)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    fri = first + pd.offsets.Week(weekday=4)
    if fri.month != month:
        fri += pd.Timedelta(days=7)
    return fri + pd.Timedelta(days=14)


def _holiday_set(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Approximate US market holidays via USFederalHolidayCalendar."""
    cal = USFederalHolidayCalendar()
    return set(pd.to_datetime(cal.holidays(start=start, end=end)).normalize())


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calendar flags on *unique trading dates* first, then merge
    back -- avoids slow per-row apply on the full (date x ticker) frame.
    """
    print(f"\n{'='*60}")
    print("STEP 5 -- Calendar features")
    print(f"{'='*60}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Work on unique dates only
    dates = pd.Series(sorted(df["date"].unique())).reset_index(drop=True)
    cal = pd.DataFrame({"date": dates})
    cal["date_norm"] = cal["date"].dt.normalize()

    # -- Basic flags --
    cal["is_friday"] = (cal["date"].dt.dayofweek == 4).astype(int)

    # Month-end / quarter-end = last *trading* day in that period
    cal["_ym"] = cal["date"].dt.to_period("M")
    cal["_yq"] = cal["date"].dt.to_period("Q")
    month_last   = cal.groupby("_ym")["date"].transform("max")
    quarter_last = cal.groupby("_yq")["date"].transform("max")
    cal["is_month_end"]   = (cal["date"] == month_last).astype(int)
    cal["is_quarter_end"] = (cal["date"] == quarter_last).astype(int)

    # -- Pre-holiday flag --
    #   "Next trading day is > 1 calendar day away AND it's not a regular
    #    Friday -> Monday gap."
    next_td = dates.shift(-1)
    gap     = (next_td - dates).dt.days
    cal["is_pre_holiday"] = (
        (gap > 1) & ~((cal["date"].dt.dayofweek == 4) & (gap == 3))
    ).astype(int).fillna(0).astype(int)

    # -- Pre-long-weekend --
    cal["is_pre_long_weekend"] = (gap >= 3).astype(int).fillna(0).astype(int)

    # -- Options expiration (OpEx) --
    opex_set = set()
    for y in cal["date"].dt.year.unique():
        for m in range(1, 13):
            opex_set.add(_third_friday(int(y), m).normalize())

    cal["is_opex_day"] = cal["date_norm"].isin(opex_set).astype(int)

    # OpEx week = week containing the third Friday
    def _in_opex_week(d):
        mon = (d - pd.Timedelta(days=d.weekday())).normalize()
        return int(any((mon + pd.Timedelta(days=i)).normalize() in opex_set
                       for i in range(5)))
    cal["is_opex_week"] = cal["date"].apply(_in_opex_week)

    # Drop helper columns and merge back
    cal = cal.drop(columns=["date_norm", "_ym", "_yq"])
    df = df.merge(cal, on="date", how="left")

    added = [c for c in cal.columns if c != "date"]
    print(f"  Added: {added}")
    return df


# ===========================================================================
#  STEP 6 -- Merge & Clean
# ===========================================================================

def merge_and_clean(stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro onto stock rows, forward-fill, drop warm-up NaNs, winsorise."""
    print(f"\n{'='*60}")
    print("STEP 6 -- Merge & clean")
    print(f"{'='*60}")

    df = stock_df.merge(macro_df, on="date", how="left")

    # Forward-fill macro within each ticker
    macro_cols = [c for c in macro_df.columns if c != "date"]
    df = df.sort_values(["ticker", "date"])
    df[macro_cols] = df.groupby("ticker")[macro_cols].transform(
        lambda s: s.ffill().bfill()
    )

    # Drop warm-up rows (first ~60 per ticker lack momentum_60d etc.)
    required = [
        "ret_1d", "ret_5d", "ret_20d", "vol_20d",
        "momentum_20d", "momentum_60d",
    ]
    n0 = len(df)
    df = df.dropna(subset=required)
    print(f"  Dropped {n0 - len(df):,} warm-up rows ({n0:,} -> {len(df):,})")

    # Drop rows without target (last 5 per ticker)
    n0 = len(df)
    df = df.dropna(subset=["target_ret_5d"])
    print(f"  Dropped {n0 - len(df):,} target-NaN rows ({n0:,} -> {len(df):,})")

    # Winsorise extreme tails at 1st / 99th percentile
    # NOTE: hl_spread and oc_return are derived directly from raw OHLCV prices,
    # so we do NOT winsorise them here -- doing so would create inconsistencies
    # between the feature column and the underlying open/high/low/close columns.
    # They are winsorised later in validate_dataset.py after being recomputed
    # from the (already-validated) raw prices.
    clip_cols = [
        "ret_1d", "ret_5d", "ret_20d", "target_ret_5d",
        "vol_20d",
        "momentum_20d", "momentum_60d",
    ]
    for col in clip_cols:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Canonical column order (auto-filters to columns that actually exist)
    order = [
        # identifiers
        "date", "ticker",
        # raw OHLCV (close dropped; adj_close is already split-/dividend-adjusted)
        "open", "high", "low", "adj_close", "volume",
        # price-volume features
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "skew_20d", "kurt_20d",
        "momentum_20d", "momentum_60d",
        "volume_zscore", "dollar_volume",
        "hl_spread", "oc_return",
        "sector",
        # macro / regime
        "vix", "vix_change_9d", "vvix", "vix_ma20", "vix_slope",
        "bond_yield_3m", "bond_yield_5y",
        "gld_ret", "wti_ret", "tlt_ret", "ief_ret",
        "dxy", "dxy_ret_1d", "dxy_ma20",
        "nfp", "unrate", "initial_claims",
        # calendar
        "is_friday", "is_month_end", "is_quarter_end",
        "is_pre_holiday", "is_pre_long_weekend",
        "is_opex_week", "is_opex_day",
        # target
        "target_ret_5d",
    ]
    order = [c for c in order if c in df.columns]
    df = df[order]

    print(f"  Final shape : {df.shape}")
    print(f"  Date range  : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Tickers     : {df['ticker'].nunique()}")
    print(f"  Columns     : {len(df.columns)}")
    return df


# ===========================================================================
#  MAIN
# ===========================================================================

def main() -> None:
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  EQUITY RETURN DATASET BUILDER")
    print("=" * 60)

    tickers  = get_universe(n=100)                     # Step 1
    ohlcv_df = download_ohlcv(tickers)                 # Step 2
    stock_df = engineer_price_features(ohlcv_df)       # Step 3
    macro_df = build_macro_df()                        # Step 4
    stock_df = add_calendar_features(stock_df)         # Step 5
    final_df = merge_and_clean(stock_df, macro_df)     # Step 6

    # Save
    out = PROC_DIR / "dataset_final.csv"
    final_df.to_csv(out, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  DONE -- saved {out}  ({elapsed:.1f} min)")
    print(f"{'='*60}\n")

    # Quick sanity check
    print(final_df.head(3).to_string())
    print("\nColumn dtypes:\n", final_df.dtypes)
    missing = final_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nRemaining NaN counts:\n", missing)
    else:
        print("\nNo missing values.")


if __name__ == "__main__":
    main()
