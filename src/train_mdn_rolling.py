"""
train_mdn_rolling.py
====================

Rolling-window **Mixture Density Network (MDN)** for per-ticker equity
range prediction.  Models the conditional return distribution as a
mixture of Gaussians and trains with negative log-likelihood (NLL).

Architecture
------------
    Input (d features)
      -> BatchNorm -> Linear(d, 256) -> ReLU -> Dropout(0.3)
      -> Linear(256, 128) -> ReLU -> Dropout(0.3)
      -> Linear(128, 64)  -> ReLU -> Dropout(0.2)
      -> MDN head:
           Linear(64, K)    -> softmax   -> pi    (mixture weights)
           Linear(64, K)                 -> mu    (component means)
           Linear(64, K)    -> exp+eps   -> sigma (component std devs)

    Loss = -log sum_k pi_k * N(y | mu_k, sigma_k)

    Point prediction = sum_k pi_k * mu_k   (mixture mean)

The MDN additionally outputs **predictive uncertainty** (mixture std)
and can be evaluated via NLL — a metric that captures calibration.

Targets / evaluation / rolling windows are **identical** to the XGBoost
baseline so the comparison is fair.

Usage
-----
    cd <project_root>
    python src/train_mdn_rolling.py
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Shared modules ──────────────────────────────────────────────────
from common.constants import (
    PROJECT_ROOT,
    DATA_PATH,
    DATE_COL,
    TICKER_COL,
    TARGET_UP,
    TARGET_DN,
    RANDOM_SEED,
    TRAIN_MONTHS,
    TEST_MONTHS,
    STEP_MONTHS,
    PURGE_GAP_DAYS,
    HIDDEN_DIMS,
    DROPOUT,
    LR,
    WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    VAL_FRAC,
    PRED_FEAT_COLS,
    FORECAST_TICKERS,
    get_device,
)
from common.data import (
    check_required_columns,
    prepare_data_nn,
    infer_feature_columns,
    encode_and_scale,
    robust_standardize,
    inverse_standardize,
    build_rolling_windows,
    update_dataset,
    build_stock_features,
    build_macro_features,
)
from common.metrics import (
    rmse_mae,
    improvement_pct,
    compute_ticker_metrics as _compute_ticker_metrics,
    summarize_all_predictions as _summarize_all_predictions,
)
from common.evaluation import (
    split_fold,
    save_results,
    print_overall,
    print_ticker_table,
    print_saved_paths,
)
from common.plotting import (
    plot_fold_rmse,
    plot_fold_improvement,
    plot_ticker_scatter,
    plot_overall_bar,
    plot_fold_nll,
    plot_ticker_accuracy,
)
from common.live import (
    load_and_prepare_live_data,
    download_latest_features,
    print_live_results,
    save_live_forecast,
)

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIG
# ===================================================================
MODEL_NAME  = "mdn"
MODEL_LABEL = "MDN"
OUTPUT_DIR  = PROJECT_ROOT / "results" / "mdn_rolling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MDN-specific hyper-parameters
N_COMPONENTS = 5            # number of Gaussian mixture components
SIGMA_MIN    = 1e-4         # floor on predicted sigma to avoid log(0)

DEVICE = get_device()


# ===================================================================
# MDN MODEL
# ===================================================================
class MDN(nn.Module):
    """
    Mixture Density Network.

    Backbone is a shared feedforward trunk.  Three parallel heads
    produce (pi, mu, sigma) for K Gaussian components.
    """

    def __init__(self, input_dim: int, n_components: int = N_COMPONENTS):
        super().__init__()
        self.K = n_components

        # Shared backbone
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h, d in zip(HIDDEN_DIMS, DROPOUT):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        self.backbone = nn.Sequential(*layers)

        # MDN heads
        self.pi_head    = nn.Linear(prev, n_components)   # mixture weights
        self.mu_head    = nn.Linear(prev, n_components)   # means
        self.sigma_head = nn.Linear(prev, n_components)   # log-std devs

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        pi    = torch.softmax(self.pi_head(h), dim=-1)
        mu    = self.mu_head(h)
        sigma = torch.exp(self.sigma_head(h)) + SIGMA_MIN
        return pi, mu, sigma


# ===================================================================
# MDN LOSS & UTILITY FUNCTIONS
# ===================================================================
def mdn_nll_loss(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for a Gaussian mixture."""
    y = y.unsqueeze(-1)                              # (B, 1)
    log_pi    = torch.log(pi + 1e-10)                # (B, K)
    log_norm  = -0.5 * math.log(2 * math.pi) \
                - torch.log(sigma + 1e-10)           # (B, K)
    log_exp   = -0.5 * ((y - mu) / (sigma + 1e-10)) ** 2
    log_probs = log_pi + log_norm + log_exp          # (B, K)
    log_sum   = torch.logsumexp(log_probs, dim=-1)   # (B,)
    return -log_sum.mean()


def mdn_mean(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Mixture mean = sum_k pi_k * mu_k — point prediction."""
    return (pi * mu).sum(dim=-1)


def mdn_std(pi: torch.Tensor, mu: torch.Tensor,
            sigma: torch.Tensor) -> torch.Tensor:
    """Mixture std = sqrt(sum pi*(sigma^2 + mu^2) - (sum pi*mu)^2)."""
    mean = mdn_mean(pi, mu)
    var = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1) - mean ** 2
    return torch.sqrt(var.clamp(min=1e-10))


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
    up_mdn_rmse: float
    up_mdn_mae: float
    up_mdn_nll: float
    up_naive_rmse: float
    up_naive_mae: float
    dn_mdn_rmse: float
    dn_mdn_mae: float
    dn_mdn_nll: float
    dn_naive_rmse: float
    dn_naive_mae: float


# ===================================================================
# TRAINING LOOP FOR ONE TARGET
# ===================================================================
def _train_mdn_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Train MDN with temporal val split + target preprocessing.
    Returns (point_predictions, uncertainties, test_NLL).
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_proc, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_train)

    # Temporal validation split
    n = len(X_train)
    n_val = max(1, int(n * VAL_FRAC))
    n_tr  = n - n_val
    tr_idx = np.arange(n_tr)
    val_idx = np.arange(n_tr, n)

    Xtr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y_proc[tr_idx],  dtype=torch.float32)
    Xva = torch.tensor(X_train[val_idx], dtype=torch.float32)
    yva = torch.tensor(y_proc[val_idx],  dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    # Standardize test targets for NLL computation
    y_test_z = (y_test - y_center) / y_scale
    yte = torch.tensor(y_test_z, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, ytr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False)

    model = MDN(X_train.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_nll_loss(pi, mu, sigma, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            pi, mu, sigma = model(Xva.to(DEVICE))
            val_loss = mdn_nll_loss(pi, mu, sigma, yva.to(DEVICE)).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(Xte.to(DEVICE))
        pred_mean_z = mdn_mean(pi, mu).cpu().numpy()
        pred_std_z  = mdn_std(pi, mu, sigma).cpu().numpy()
        test_nll    = mdn_nll_loss(pi, mu, sigma, yte.to(DEVICE)).item()

    pred_mean = inverse_standardize(pred_mean_z, y_center, y_scale, clip_lo, clip_hi)
    pred_std  = pred_std_z * y_scale

    return pred_mean, pred_std, float(test_nll)


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

    train_df, test_df, X_train_raw, X_test_raw = split_fold(
        df, feature_cols, fold_id, train_start, train_end, test_start, test_end,
    )

    Xtr, Xte, _ = encode_and_scale(X_train_raw, X_test_raw)

    y_up_train = train_df[TARGET_UP].values.astype(np.float32)
    y_dn_train = train_df[TARGET_DN].values.astype(np.float32)
    y_up_test  = test_df[TARGET_UP].values.astype(np.float32)
    y_dn_test  = test_df[TARGET_DN].values.astype(np.float32)

    # Train separate MDN for y_up and y_down
    pred_up, std_up, nll_up = _train_mdn_full(Xtr, y_up_train, Xte, y_up_test)
    pred_dn, std_dn, nll_dn = _train_mdn_full(Xtr, y_dn_train, Xte, y_dn_test)

    # Assemble prediction dataframe
    pred_df = test_df[[DATE_COL, TICKER_COL]].copy()
    pred_df["y_up_true"]  = y_up_test
    pred_df["y_up_pred"]  = pred_up
    pred_df["y_up_std"]   = std_up
    pred_df["y_up_naive"] = 0.0
    pred_df["y_dn_true"]  = y_dn_test
    pred_df["y_dn_pred"]  = pred_dn
    pred_df["y_dn_std"]   = std_dn
    pred_df["y_dn_naive"] = 0.0
    pred_df["fold_id"]    = fold_id

    # Fold-level metrics
    up_mdn_rmse, up_mdn_mae   = rmse_mae(pred_df["y_up_true"], pred_df["y_up_pred"])
    up_naive_rmse, up_naive_mae = rmse_mae(pred_df["y_up_true"], pred_df["y_up_naive"])
    dn_mdn_rmse, dn_mdn_mae   = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_pred"])
    dn_naive_rmse, dn_naive_mae = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_naive"])

    return FoldResult(
        fold_id=fold_id,
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        n_train=len(train_df),
        n_test=len(test_df),
        up_mdn_rmse=up_mdn_rmse, up_mdn_mae=up_mdn_mae,
        up_mdn_nll=nll_up,
        up_naive_rmse=up_naive_rmse, up_naive_mae=up_naive_mae,
        dn_mdn_rmse=dn_mdn_rmse, dn_mdn_mae=dn_mdn_mae,
        dn_mdn_nll=nll_dn,
        dn_naive_rmse=dn_naive_rmse, dn_naive_mae=dn_naive_mae,
    ), pred_df


# ===================================================================
# PLOTTING
# ===================================================================
def generate_plots(
    fold_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
    overall: Dict[str, float],
) -> None:
    plot_fold_rmse(fold_df, MODEL_LABEL, MODEL_NAME, OUTPUT_DIR,
                   model_color="tab:purple")
    plot_fold_improvement(fold_df, MODEL_LABEL, MODEL_NAME, overall, OUTPUT_DIR)
    plot_ticker_scatter(ticker_df, MODEL_LABEL, OUTPUT_DIR,
                        scatter_color="tab:purple")
    plot_overall_bar(overall, MODEL_LABEL, OUTPUT_DIR)
    plot_fold_nll(fold_df, MODEL_LABEL, MODEL_NAME, OUTPUT_DIR)
    print(f"\n  All 5 figures saved to {OUTPUT_DIR}/")


# ===================================================================
# MAIN
# ===================================================================
def main() -> None:
    update_dataset()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"\n{'='*60}")
    print("  MDN ROLLING-WINDOW TRAINER  (y_up / y_down)")
    print(f"{'='*60}")
    print(f"  Model       : MDN  (K={N_COMPONENTS} Gaussians, "
          f"hidden={HIDDEN_DIMS})")
    print(f"  Device      : {DEVICE}")
    print(f"  Data        : {DATA_PATH}")
    print(f"  Targets     : y_up  = max(P_{{t+1:t+5}})/P_t - 1")
    print(f"              : y_down= min(P_{{t+1:t+5}})/P_t - 1")
    print(f"  Naive       : predict 0")
    print(f"  Train window: {TRAIN_MONTHS} months")
    print(f"  Test  window: {TEST_MONTHS} months")
    print(f"  Step        : {STEP_MONTHS} months")
    print(f"  Purge gap   : {PURGE_GAP_DAYS} days")
    print(f"  LR={LR}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}, "
          f"patience={PATIENCE}, val_frac={VAL_FRAC}")

    df = pd.read_csv(DATA_PATH)
    check_required_columns(df)
    df = prepare_data_nn(df)

    feature_cols = infer_feature_columns(df)
    print(f"\n  Rows         : {len(df):,}")
    print(f"  Date range   : {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"  Tickers      : {df[TICKER_COL].nunique()}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:8]} ...")

    windows = build_rolling_windows(
        all_dates=df[DATE_COL],
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        purge_gap_days=PURGE_GAP_DAYS,
    )
    if not windows:
        raise ValueError("No rolling windows created.")

    print(f"  Rolling folds: {len(windows)}\n")

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

            up_imp = improvement_pct(fold_res.up_mdn_rmse, fold_res.up_naive_rmse)
            dn_imp = improvement_pct(fold_res.dn_mdn_rmse, fold_res.dn_naive_rmse)
            print(f"\n        y_up:  MDN RMSE={fold_res.up_mdn_rmse:.6f}  "
                  f"Naive={fold_res.up_naive_rmse:.6f}  Improv={up_imp:+.1f}%  "
                  f"NLL={fold_res.up_mdn_nll:.3f}"
                  f"\n        y_dn:  MDN RMSE={fold_res.dn_mdn_rmse:.6f}  "
                  f"Naive={fold_res.dn_naive_rmse:.6f}  Improv={dn_imp:+.1f}%  "
                  f"NLL={fold_res.dn_mdn_nll:.3f}")
        except Exception as e:
            print(f"  ⚠️  SKIPPED: {e}")

    if not all_preds_list:
        raise RuntimeError("No folds completed.")

    all_preds       = pd.concat(all_preds_list, ignore_index=True)
    fold_results_df = pd.DataFrame([asdict(fr) for fr in all_fold_results])

    # Compute summaries using shared functions with NLL values
    fold_nll_values = {
        "up": [fr.up_mdn_nll for fr in all_fold_results],
        "dn": [fr.dn_mdn_nll for fr in all_fold_results],
    }
    overall = _summarize_all_predictions(
        all_preds, MODEL_NAME, fold_nll_values=fold_nll_values, has_std=True,
    )
    ticker_metrics = _compute_ticker_metrics(all_preds, MODEL_NAME, has_std=True)

    # ── Save, Print, Plot ──────────────────────────────────────────
    save_results(OUTPUT_DIR, MODEL_NAME, all_preds, fold_results_df,
                 ticker_metrics, overall)
    print_overall(overall, MODEL_LABEL)
    print_ticker_table(ticker_metrics, MODEL_NAME, MODEL_LABEL, has_std=True)
    print_saved_paths(OUTPUT_DIR, MODEL_NAME)
    generate_plots(fold_results_df, ticker_metrics, overall)


# ===================================================================
# FIG 6 — PER-TICKER TRUE vs PREDICTED
# ===================================================================
def generate_ticker_accuracy_plot(
    tickers=("AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"),
):
    plot_ticker_accuracy(
        OUTPUT_DIR, MODEL_NAME, MODEL_LABEL, fig_num=6,
        tickers=tickers, up_color="#7B1FA2", dn_color="#E65100",
        has_std=True,
    )


# ===================================================================
# LIVE FORECAST — train on ALL data, predict next 5 days
# ===================================================================
def live_forecast(
    tickers=("AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"),
) -> None:
    """
    Train MDN on the full (updated) dataset using PRED_FEAT_COLS,
    then predict y_up / y_down (with uncertainty) for the given tickers
    for the next 5 trading days.
    """
    from datetime import date

    today = date.today()
    print(f"\n{'='*70}")
    print(f"  MDN LIVE FORECAST  —  {today}  (next 5 trading days)")
    print(f"{'='*70}")

    # ── 1. Load updated dataset ──────────────────────────────
    print("  [1/3] Loading updated training data ...")
    train_df, X_all_sc, y_up_all, y_dn_all, scaler, all_tickers, data_end = (
        load_and_prepare_live_data()
    )

    # ── 2. Train MDN for y_up and y_down ───────────────────────
    print("  [2/3] Training MDN models (y_up & y_down) ...")

    def _train_full_mdn(X_sc, y_raw):
        """Train MDN on full data, return (model, y_center, y_scale, clip_lo, clip_hi)."""
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        y_proc, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_raw)

        n = len(X_sc)
        n_val = max(1, int(n * VAL_FRAC))
        n_tr = n - n_val
        tr_idx = np.arange(n_tr)
        val_idx = np.arange(n_tr, n)

        Xtr = torch.tensor(X_sc[tr_idx], dtype=torch.float32)
        ytr = torch.tensor(y_proc[tr_idx], dtype=torch.float32)
        Xva = torch.tensor(X_sc[val_idx], dtype=torch.float32)
        yva = torch.tensor(y_proc[val_idx], dtype=torch.float32)

        train_ds = TensorDataset(Xtr, ytr)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False)

        model = MDN(X_sc.shape[1]).to(DEVICE)
        optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimiser.zero_grad()
                pi, mu, sigma = model(xb)
                loss = mdn_nll_loss(pi, mu, sigma, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            model.eval()
            with torch.no_grad():
                pi, mu, sigma = model(Xva.to(DEVICE))
                val_loss = mdn_nll_loss(pi, mu, sigma,
                                         yva.to(DEVICE)).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model, y_center, y_scale, clip_lo, clip_hi

    model_up, c_up, s_up, lo_up, hi_up = _train_full_mdn(X_all_sc, y_up_all)
    model_dn, c_dn, s_dn, lo_dn, hi_dn = _train_full_mdn(X_all_sc, y_dn_all)

    # ── 3. Download latest features & predict ──────────────────
    print(f"  [3/3] Predicting {len(tickers)} tickers ...\n")
    dl_start = (pd.Timestamp(today) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    dl_end = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    macro = build_macro_features(dl_start, dl_end)

    live_results = []
    for ticker in tickers:
        result = download_latest_features(ticker, macro, scaler, dl_start, dl_end)
        if result is None:
            continue
        X_sc, price, latest_date = result

        Xt = torch.tensor(X_sc, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pi_up, mu_up, sigma_up = model_up(Xt)
            pred_up_z = mdn_mean(pi_up, mu_up).cpu().numpy()
            std_up_z  = mdn_std(pi_up, mu_up, sigma_up).cpu().numpy()

            pi_dn, mu_dn, sigma_dn = model_dn(Xt)
            pred_dn_z = mdn_mean(pi_dn, mu_dn).cpu().numpy()
            std_dn_z  = mdn_std(pi_dn, mu_dn, sigma_dn).cpu().numpy()

        pred_up = float(inverse_standardize(
            pred_up_z, c_up, s_up, lo_up, hi_up).item())
        pred_dn = float(inverse_standardize(
            pred_dn_z, c_dn, s_dn, lo_dn, hi_dn).item())
        std_up = float(std_up_z.item()) * s_up
        std_dn = float(std_dn_z.item()) * s_dn

        live_results.append({
            "ticker": ticker,
            "latest_date": str(latest_date.date()),
            "price": float(price),
            "y_up": pred_up,
            "y_down": pred_dn,
            "y_up_std": std_up,
            "y_down_std": std_dn,
            "price_high": float(price) * (1 + pred_up),
            "price_low": float(price) * (1 + pred_dn),
        })

    # ── Print & save ──────────────────────────────────────────
    print_live_results(live_results, MODEL_LABEL, all_tickers, data_end,
                       has_std=True)
    save_live_forecast(OUTPUT_DIR, MODEL_LABEL, live_results)


if __name__ == "__main__":
    main()
    generate_ticker_accuracy_plot()
    live_forecast()
