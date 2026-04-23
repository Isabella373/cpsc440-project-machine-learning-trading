"""
validate_dataset.py
===================
Validate and clean the dataset produced by build_dataset.py.

Output
------
data/processed/dataset_final_cleaned.csv
data/processed/dataset_final_validation_report.json

Run
---
    cd <project_root>
    python src/data_processing/validate_dataset.py
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column taxonomy
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "date", "ticker", "open", "high", "low", "adj_close", "volume",
    "ret_1d", "ret_5d", "ret_20d", "vol_20d", "skew_20d", "kurt_20d",
    "momentum_20d", "momentum_60d",
    "volume_zscore", "dollar_volume", "hl_spread", "oc_return",
    "sector", "vix", "vix_change_9d", "vvix", "vix_ma20", "vix_slope",
    "bond_yield_3m", "bond_yield_5y", "gld_ret", "wti_ret", "tlt_ret", "ief_ret",
    "dxy", "dxy_ret_1d", "dxy_ma20", "nfp", "unrate", "initial_claims",
    "is_friday", "is_month_end", "is_quarter_end", "is_pre_holiday",
    "is_pre_long_weekend", "is_opex_week", "is_opex_day", "target_ret_5d",
]

RAW_PRICE_COLS = ["open", "high", "low", "adj_close", "volume"]

RETURN_LIKE_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "momentum_20d", "momentum_60d",
    "vix_change_9d", "vix_slope", "gld_ret", "wti_ret", "tlt_ret",
    "ief_ret", "dxy_ret_1d", "target_ret_5d", "oc_return",
]

NONNEGATIVE_COLS = [
    "volume", "vol_20d", "dollar_volume", "hl_spread", "vix", "vvix",
    "bond_yield_3m", "bond_yield_5y", "dxy", "nfp", "unrate", "initial_claims",
]

BINARY_FLAG_COLS = [
    "is_friday", "is_month_end", "is_quarter_end", "is_pre_holiday",
    "is_pre_long_weekend", "is_opex_week", "is_opex_day",
]

MACRO_COLS = [
    "vix", "vix_change_9d", "vvix", "vix_ma20", "vix_slope",
    "bond_yield_3m", "bond_yield_5y", "gld_ret", "wti_ret", "tlt_ret",
    "ief_ret", "dxy", "dxy_ret_1d", "dxy_ma20", "nfp", "unrate", "initial_claims",
]

CRITICAL_FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "vol_20d", "momentum_20d", "momentum_60d",
    "target_ret_5d",
]

WINSORIZE_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "target_ret_5d", "vol_20d", "hl_spread",
    "oc_return", "momentum_20d", "momentum_60d", "volume_zscore",
    "skew_20d", "kurt_20d", "vix_change_9d", "dxy_ret_1d",
    "gld_ret", "wti_ret", "tlt_ret", "ief_ret",
]


# ---------------------------------------------------------------------------
# Validator / Cleaner
# ---------------------------------------------------------------------------

class DatasetValidatorCleaner:
    """Load, validate, clean, and save dataset_final.csv."""

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self.report: dict = {
            "file": str(self.csv_path),
            "rows_before": 0,
            "rows_after": 0,
            "issues": {},
            "actions": {},
            "summary": {},
        }

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------

    def load(self) -> "DatasetValidatorCleaner":
        print(f"\n{'='*60}")
        print("LOAD -- Reading CSV")
        print(f"{'='*60}")
        self.df = pd.read_csv(self.csv_path)
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.report["rows_before"] = len(self.df)
        print(f"  Rows      : {len(self.df):,}")
        print(f"  Columns   : {self.df.shape[1]}")
        return self

    # ------------------------------------------------------------------
    # 2. VALIDATE  (read-only inspection, fills self.report["issues"])
    # ------------------------------------------------------------------

    def validate(self) -> dict:
        if self.df is None:
            self.load()
        df = self.df
        issues: dict = {}

        print(f"\n{'='*60}")
        print("VALIDATE -- Inspecting data quality")
        print(f"{'='*60}")

        # 1) Schema
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        issues["missing_columns"] = missing_cols
        issues["extra_columns"]   = extra_cols
        print(f"  Schema    : {len(missing_cols)} missing cols, {len(extra_cols)} extra cols")

        # 2) Date / ticker validity
        issues["bad_date_rows"]         = int(df["date"].isna().sum())
        issues["missing_ticker_rows"]   = int(df["ticker"].isna().sum())
        print(f"  Bad dates : {issues['bad_date_rows']}   Missing ticker: {issues['missing_ticker_rows']}")

        # 3) Duplicate (date, ticker) keys
        dup_count = int(df.duplicated(subset=["date", "ticker"], keep=False).sum())
        issues["duplicate_date_ticker_rows"] = dup_count
        print(f"  Duplicates: {dup_count} rows with duplicate (date, ticker)")

        # 4) Missing values
        issues["missing_by_column"] = {
            col: int(val)
            for col, val in df.isna().sum().items()
            if int(val) > 0
        }
        total_missing = sum(issues["missing_by_column"].values())
        print(f"  NaN total : {total_missing:,}  (across {len(issues['missing_by_column'])} columns)")

        # 5) Infinite values
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            inf_counts = np.isinf(df[num_cols]).sum()
            issues["inf_by_column"] = {
                col: int(v) for col, v in inf_counts.items() if int(v) > 0
            }
        else:
            issues["inf_by_column"] = {}
        print(f"  Inf total : {sum(issues['inf_by_column'].values()):,}  (across {len(issues['inf_by_column'])} columns)")

        # 6) Raw OHLCV sanity
        raw_price_issues: dict = {}
        for col in [c for c in RAW_PRICE_COLS if c in df.columns]:
            raw_price_issues[f"{col}_nonpositive"] = int((df[col] <= 0).sum())
        if all(c in df.columns for c in ["open", "high", "low", "adj_close"]):
            OHLC_TOL_PCT = 0.001
            hl_inv   = df["high"] < df["low"]
            tol_mask = (df["low"] - df["high"]).abs() / df["adj_close"] <= OHLC_TOL_PCT
            raw_price_issues["ohlc_inversion_fixable"] = int((hl_inv &  tol_mask).sum())
            raw_price_issues["ohlc_inversion_drop"]    = int((hl_inv & ~tol_mask).sum())
        issues["raw_price_issues"] = raw_price_issues
        issues["bad_ohlc_rows"] = (
            raw_price_issues.get("ohlc_inversion_fixable", 0)
            + raw_price_issues.get("ohlc_inversion_drop", 0)
        )
        bad_ohlcv = sum(raw_price_issues.values())
        print(f"  OHLCV bad : {bad_ohlcv} problematic cells/rows  "
              f"(fixable inversions: {raw_price_issues.get('ohlc_inversion_fixable', 0)}, "
              f"drop: {raw_price_issues.get('ohlc_inversion_drop', 0)})")

        # 7) Feature-range sanity
        feature_issues: dict = {}
        if "rsi_14" in df.columns:
            feature_issues["rsi_out_of_range"] = int(
                ((df["rsi_14"] < 0) | (df["rsi_14"] > 100)).sum()
            )
        if "hl_spread" in df.columns:
            feature_issues["negative_hl_spread"] = int((df["hl_spread"] < 0).sum())
        if "volume_zscore" in df.columns:
            feature_issues["extreme_volume_zscore_abs_gt_10"] = int(
                (df["volume_zscore"].abs() > 10).sum()
            )
        if "skew_20d" in df.columns:
            feature_issues["extreme_skew_abs_gt_10"] = int(
                (df["skew_20d"].abs() > 10).sum()
            )
        if "kurt_20d" in df.columns:
            feature_issues["extreme_kurt_abs_gt_50"] = int(
                (df["kurt_20d"].abs() > 50).sum()
            )
        issues["feature_issues"] = feature_issues
        print(f"  Feature   : {feature_issues}")

        # 8) Nonnegative column violations
        neg_nonnegs: dict = {}
        for col in [c for c in NONNEGATIVE_COLS if c in df.columns]:
            cnt = int((df[col] < 0).sum())
            if cnt > 0:
                neg_nonnegs[col] = cnt
        issues["negative_nonnegative_cols"] = neg_nonnegs

        # 9) Binary flag violations
        bad_flags: dict = {}
        for col in [c for c in BINARY_FLAG_COLS if c in df.columns]:
            cnt = int((~df[col].isin([0, 1])).sum())
            if cnt > 0:
                bad_flags[col] = cnt
        issues["bad_binary_flags"] = bad_flags

        # 10) Per-ticker date ordering
        not_sorted   = 0
        has_dup_date = 0
        for _, g in df.groupby("ticker", dropna=False):
            d = g["date"]
            if not d.is_monotonic_increasing:
                not_sorted += 1
            if d.duplicated().any():
                has_dup_date += 1
        issues["tickers_not_sorted_by_date"]    = int(not_sorted)
        issues["tickers_with_duplicate_dates"]  = int(has_dup_date)
        print(f"  Ordering  : {not_sorted} tickers unsorted, {has_dup_date} tickers with dup dates")

        # 11) Target consistency spot-check
        if all(c in df.columns for c in ["ticker", "date", "adj_close", "target_ret_5d"]):
            chk     = df[["ticker", "date", "adj_close", "target_ret_5d"]].copy()
            chk     = chk.sort_values(["ticker", "date"])
            implied = chk.groupby("ticker")["adj_close"].transform(
                lambda s: np.log(s.shift(-5) / s)
            )
            diff = (implied - chk["target_ret_5d"]).abs()
            issues["target_mismatch_rows_abs_gt_1e_8"] = int((diff > 1e-8).sum())
        else:
            issues["target_mismatch_rows_abs_gt_1e_8"] = None
        print(f"  Target    : {issues['target_mismatch_rows_abs_gt_1e_8']} mismatched rows")

        # 12) Outlier summary
        outlier_summary: dict = {}
        for col in [c for c in RETURN_LIKE_COLS if c in df.columns]:
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            outlier_summary[col] = {
                "q01": None if pd.isna(q01) else round(float(q01), 6),
                "q99": None if pd.isna(q99) else round(float(q99), 6),
                "abs_gt_0_5": int((df[col].abs() > 0.5).sum()),
            }
        issues["outlier_summary"] = outlier_summary

        self.report["issues"] = issues

        # Print summary of issues found
        total_issues = (
            len(missing_cols) + total_missing +
            sum(issues["inf_by_column"].values()) + bad_ohlcv +
            dup_count + issues["bad_date_rows"]
        )
        print(f"\n  {'✅ No critical issues found.' if total_issues == 0 else f'⚠️  Total potential issues: {total_issues:,}'}")

        return self.report

    # ------------------------------------------------------------------
    # 3. CLEAN  (modifies self.df in-place, logs actions)
    # ------------------------------------------------------------------

    def clean(self, winsorize: bool = True) -> pd.DataFrame:
        if self.df is None:
            self.load()
        df = self.df.copy()
        actions: dict = {}

        print(f"\n{'='*60}")
        print("CLEAN -- Applying fixes")
        print(f"{'='*60}")

        # --- 1. Date coercion ---
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        actions["date_parse"] = "converted date column to datetime"

        # --- 2. Drop rows with missing primary keys ---
        before = len(df)
        df = df.dropna(subset=["date", "ticker"])
        dropped = before - len(df)
        actions["drop_missing_date_or_ticker"] = dropped
        if dropped:
            print(f"  Dropped {dropped} rows missing date/ticker")

        # --- 3. Sort + deduplicate (date, ticker) ---
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        before = len(df)
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
        dropped = before - len(df)
        actions["drop_duplicate_date_ticker"] = dropped
        if dropped:
            print(f"  Dropped {dropped} duplicate (date, ticker) rows")

        # --- 4. Replace inf -> NaN ---
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            inf_total = int(np.isinf(df[num_cols]).sum().sum())
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        else:
            inf_total = 0
        actions["replace_inf_with_nan"] = inf_total
        if inf_total:
            print(f"  Replaced {inf_total} inf values with NaN")

        # --- 5. Fix / drop impossible OHLCV rows ---
        # Threshold: if high/low inversion is within 0.1% of close, treat as
        # a data-vendor rounding artefact and correct it; otherwise drop.
        OHLC_TOL_PCT = 0.001   # 0.1 %

        # 5a) Hard-impossible: NaN or non-positive prices/volume -> always drop
        hard_impossible = pd.Series(False, index=df.index)
        for col in [c for c in RAW_PRICE_COLS if c in df.columns]:
            hard_impossible |= df[col].isna() | (df[col] <= 0)
        before = len(df)
        df = df.loc[~hard_impossible].copy()
        dropped_hard = before - len(df)
        if dropped_hard:
            print(f"  Dropped {dropped_hard} rows with NaN / non-positive OHLCV")

        # 5b) Soft OHLC inversion: high < low (or slightly below open/close)
        #     Strategy:
        #       - If |high - low| / close <= TOL  -> swap high/low to fix
        #       - Else (large inversion)           -> drop the row
        corrected_ohlc = 0
        dropped_ohlc   = 0
        if all(c in df.columns for c in ["open", "high", "low", "adj_close"]):
            hl_inv   = df["high"] < df["low"]
            tol_mask = (df["low"] - df["high"]).abs() / df["adj_close"] <= OHLC_TOL_PCT

            # Fix: swap high <-> low for small inversions
            fix_mask = hl_inv & tol_mask
            if fix_mask.any():
                df.loc[fix_mask, ["high", "low"]] = (
                    df.loc[fix_mask, ["low", "high"]].values
                )
                corrected_ohlc = int(fix_mask.sum())
                print(f"  Corrected {corrected_ohlc} small high<low inversions (swapped high/low)")

            # Drop: large inversion that cannot be safely fixed
            drop_mask = hl_inv & ~tol_mask
            before = len(df)
            df = df.loc[~drop_mask].copy()
            dropped_ohlc = before - len(df)
            if dropped_ohlc:
                print(f"  Dropped {dropped_ohlc} rows with large irrecoverable OHLC inversion")

        actions["drop_impossible_ohlcv_rows"] = dropped_hard + dropped_ohlc
        actions["corrected_ohlc_inversion"]   = corrected_ohlc

        # --- 6. Recompute derived OHLCV features from raw prices ---
        if all(c in df.columns for c in ["high", "low", "adj_close"]):
            df["hl_spread"] = (df["high"] - df["low"]) / df["adj_close"]
            df["hl_spread"] = df["hl_spread"].clip(lower=0)
            actions["recompute_hl_spread"] = True

        if all(c in df.columns for c in ["open", "adj_close"]):
            df["oc_return"] = (df["adj_close"] - df["open"]) / df["open"]
            actions["recompute_oc_return"] = True

        if all(c in df.columns for c in ["adj_close", "volume"]):
            df["dollar_volume"] = df["adj_close"] * df["volume"]
            actions["recompute_dollar_volume"] = True

        print("  Recomputed: hl_spread, oc_return, dollar_volume from raw prices")

        # --- 7. Clip RSI into [0, 100] ---
        if "rsi_14" in df.columns:
            bad = int(((df["rsi_14"] < 0) | (df["rsi_14"] > 100)).sum())
            df["rsi_14"] = df["rsi_14"].clip(0, 100)
            actions["clip_rsi_to_0_100"] = bad
            if bad:
                print(f"  Clipped {bad} RSI values to [0, 100]")

        # --- 8. Normalise binary flag columns ---
        for col in [c for c in BINARY_FLAG_COLS if c in df.columns]:
            bad = int((~df[col].isin([0, 1])).sum())
            df[col] = df[col].fillna(0)
            df[col] = (df[col] > 0).astype(int)
            actions[f"normalize_{col}"] = bad

        # --- 9. Fill missing sector ---
        if "sector" in df.columns:
            missing_sector = int(df["sector"].isna().sum())
            df["sector"] = df["sector"].fillna("Other")
            actions["fill_missing_sector"] = missing_sector
            if missing_sector:
                print(f"  Filled {missing_sector} missing sector values with 'Other'")

        # --- 10. Forward/back fill macro columns within each ticker ---
        macro_present = [c for c in MACRO_COLS if c in df.columns]
        if macro_present:
            na_before = int(df[macro_present].isna().sum().sum())
            df[macro_present] = (
                df.groupby("ticker", group_keys=False)[macro_present]
                .apply(lambda x: x.ffill().bfill())
            )
            na_after = int(df[macro_present].isna().sum().sum())
            actions["fill_macro_missing_values"] = {
                "before": na_before, "after": na_after
            }
            if na_before:
                print(f"  Macro NaN: {na_before} -> {na_after} after ffill/bfill")

        # --- 11. Drop rows still missing critical training features ---
        critical_present = [c for c in CRITICAL_FEATURE_COLS if c in df.columns]
        before = len(df)
        if critical_present:
            df = df.dropna(subset=critical_present)
        dropped = before - len(df)
        actions["drop_rows_missing_critical_features"] = dropped
        if dropped:
            print(f"  Dropped {dropped} rows missing critical features")

        # --- 12. Winsorise ---
        if winsorize:
            winsor_present = [c for c in WINSORIZE_COLS if c in df.columns]
            clip_stats: dict = {}
            total_clipped = 0
            for col in winsor_present:
                lo = df[col].quantile(0.01)
                hi = df[col].quantile(0.99)
                if pd.notna(lo) and pd.notna(hi):
                    clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
                    df[col] = df[col].clip(lo, hi)
                    clip_stats[col] = {
                        "lo": round(float(lo), 6),
                        "hi": round(float(hi), 6),
                        "rows_clipped": clipped,
                    }
                    total_clipped += clipped
            actions["winsorize_1_99"] = clip_stats
            print(f"  Winsorized {len(winsor_present)} columns (total {total_clipped} values clipped)")

        # --- 13. Final sort ---
        if {"date", "ticker"}.issubset(df.columns):
            df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.df = df
        self.report["actions"]   = actions
        self.report["rows_after"] = len(df)
        self.report["summary"] = {
            "date_min": None if df.empty else str(df["date"].min().date()),
            "date_max": None if df.empty else str(df["date"].max().date()),
            "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
            "n_columns": int(df.shape[1]),
            "remaining_missing_total": int(df.isna().sum().sum()),
        }

        print(f"\n  ✅ Clean complete")
        print(f"     Rows  : {self.report['rows_before']:,}  ->  {self.report['rows_after']:,}")
        print(f"     Cols  : {self.report['summary']['n_columns']}")
        print(f"     Remaining NaN: {self.report['summary']['remaining_missing_total']}")
        return df

    # ------------------------------------------------------------------
    # 4. SAVE
    # ------------------------------------------------------------------

    def save_outputs(
        self,
        cleaned_csv_path: Optional[Union[str, Path]] = None,
        report_json_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        if self.df is None:
            raise ValueError("Run load()/clean() before save_outputs().")

        if cleaned_csv_path is None:
            cleaned_csv_path = self.csv_path.with_name(
                self.csv_path.stem + "_cleaned.csv"
            )
        if report_json_path is None:
            report_json_path = self.csv_path.with_name(
                self.csv_path.stem + "_validation_report.json"
            )

        cleaned_csv_path = Path(cleaned_csv_path)
        report_json_path = Path(report_json_path)

        self.df.to_csv(cleaned_csv_path, index=False)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("SAVE -- Outputs written")
        print(f"{'='*60}")
        print(f"  Cleaned CSV -> {cleaned_csv_path}")
        print(f"  Report JSON -> {report_json_path}")
        return cleaned_csv_path, report_json_path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def validate_and_clean_dataset(
    input_csv: Union[str, Path],
    cleaned_csv: Optional[Union[str, Path]] = None,
    report_json: Optional[Union[str, Path]] = None,
    winsorize: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """One-shot: load -> validate -> clean -> save.  Returns (df, report)."""
    runner = DatasetValidatorCleaner(input_csv)
    runner.load()
    runner.validate()
    cleaned = runner.clean(winsorize=winsorize)
    runner.save_outputs(cleaned_csv, report_json)
    return cleaned, runner.report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_csv    = project_root / "data" / "processed" / "dataset_final.csv"
    cleaned_csv  = project_root / "data" / "processed" / "dataset_final_cleaned.csv"
    report_json  = project_root / "data" / "processed" / "dataset_final_validation_report.json"

    cleaned, report = validate_and_clean_dataset(
        input_csv=input_csv,
        cleaned_csv=cleaned_csv,
        report_json=report_json,
        winsorize=True,
    )

    # ── Pretty-print top-level issue summary ───────────────────────────
    print("\n" + "=" * 60)
    print("ISSUE SUMMARY")
    print("=" * 60)
    issues = report["issues"]
    for key in [
        "missing_columns", "extra_columns",
        "bad_date_rows", "missing_ticker_rows",
        "duplicate_date_ticker_rows",
        "bad_ohlc_rows",
        "tickers_not_sorted_by_date", "tickers_with_duplicate_dates",
        "target_mismatch_rows_abs_gt_1e_8",
    ]:
        val = issues.get(key, "n/a")
        print(f"  {key:<45}: {val}")

    print("\n  Feature issues:")
    for k, v in issues.get("feature_issues", {}).items():
        print(f"    {k:<43}: {v}")

    print("\n  Missing values by column (if any):")
    for col, cnt in issues.get("missing_by_column", {}).items():
        print(f"    {col:<43}: {cnt}")

    print(f"\n  Rows before : {report['rows_before']:,}")
    print(f"  Rows after  : {report['rows_after']:,}")
    print(f"  Date range  : {report['summary']['date_min']}  ->  {report['summary']['date_max']}")
    print(f"  Tickers     : {report['summary']['n_tickers']}")
    print(f"  Remaining NaN total: {report['summary']['remaining_missing_total']}")
