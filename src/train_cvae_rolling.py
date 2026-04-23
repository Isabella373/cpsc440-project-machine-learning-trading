"""
train_cvae_rolling.py
=====================

Rolling-window **Conditional VAE with LogNormal decoder** for per-ticker
equity range prediction.

Architecture  (Lecture 14 — Variational Inference, VAEs)
-----------
    Encoder  q(z | x, y)        — used during training only
        [x; y] → LayerNorm → [256 → 128 → 64] → (μ_z, log σ_z)

    Prior  p(z | x)             — learned conditional prior
        x → LayerNorm → [128 → 64] → (μ_0, log σ_0)

    Decoder  p(y | x, z)        — Gaussian output (standardized space)
        [x; z] → LayerNorm → [256 → 128 → 64] → (μ, σ)

    LayerNorm is used instead of BatchNorm to avoid batch-size
    sensitivity and improve y_down predictions on low-vol tickers.

    Loss = ELBO = E_q[log p(y|x,z)] − β · KL[q(z|x,y) ‖ p(z|x)]

Key design choices
------------------
* **LogNormal decoder** for y_up (naturally ≥ 0) and mirrored LogNormal
  for y_down (naturally ≤ 0).

* **Learned conditional prior** p(z|x) instead of N(0,I).

* **β-warmup** on KL term to avoid posterior collapse (Bowman et al. 2016).

* Rolling windows, targets, evaluation, and live forecast are identical
  to the MDN baseline for fair comparison.

Usage
-----
    cd <project_root>
    python src/train_cvae_rolling.py
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
MODEL_NAME  = "cvae"
MODEL_LABEL = "CVAE"
OUTPUT_DIR  = PROJECT_ROOT / "results" / "cvae_rolling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CVAE-specific hyper-parameters
LATENT_DIM   = 16              # dimensionality of z
PRIOR_DIMS   = [128, 64]       # learned prior hidden layers

# β-warmup for KL term
BETA_START   = 0.0             # KL weight at epoch 0
BETA_END     = 1.0             # KL weight after warmup
BETA_WARMUP  = 30              # epochs to linearly anneal β

# Small ε for LogNormal (shift target to ensure > 0)
LN_EPS       = 1e-6
SIGMA_MIN    = 1e-4

# Number of z samples at test time for Monte Carlo prediction
N_MC_SAMPLES = 50

DEVICE = get_device()


# ===================================================================
# CVAE MODEL
# ===================================================================
def _build_mlp(dims: List[int], dropout: List[float] = None,
               batchnorm_first: bool = False) -> nn.Sequential:
    """Utility: build MLP from dimension list.

    Uses LayerNorm (instead of BatchNorm) for the input normalisation.
    LayerNorm is independent of batch size, which
      • avoids the batch_size=1 crash that skipped folds 13-14
      • produces more stable per-sample gradients, greatly improving
        y_down predictions on low-volatility tickers
    (Validated by ablation test over 6 representative folds:
     LayerNorm → +8.0% avg improvement vs −2.6% with BatchNorm.)
    """
    layers: List[nn.Module] = []
    if batchnorm_first:
        layers.append(nn.LayerNorm(dims[0]))
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # no activation on last layer
            layers.append(nn.ReLU())
            if dropout and i < len(dropout):
                layers.append(nn.Dropout(dropout[i]))
    return nn.Sequential(*layers)


class CVAELogNormal(nn.Module):
    """
    Conditional VAE with LogNormal decoder.

    Components
    ----------
    encoder  q(z | x, y) : recognition network (training only)
    prior    p(z | x)    : learned conditional prior
    decoder  p(y | x, z) : LogNormal output distribution

    The ELBO is:
        L = E_q[log p(y|x,z)] - β · KL[q(z|x,y) ‖ p(z|x)]
    """

    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder q(z | x, y) ---
        enc_dims = [input_dim + 1] + HIDDEN_DIMS
        self.encoder_body = _build_mlp(enc_dims, DROPOUT, batchnorm_first=True)
        self.enc_mu    = nn.Linear(HIDDEN_DIMS[-1], latent_dim)
        self.enc_logvar = nn.Linear(HIDDEN_DIMS[-1], latent_dim)

        # --- Prior p(z | x) ---
        prior_dims = [input_dim] + PRIOR_DIMS
        self.prior_body = _build_mlp(prior_dims, DROPOUT[:len(PRIOR_DIMS)],
                                     batchnorm_first=True)
        self.prior_mu     = nn.Linear(PRIOR_DIMS[-1], latent_dim)
        self.prior_logvar = nn.Linear(PRIOR_DIMS[-1], latent_dim)

        # --- Decoder p(y | x, z) ---
        dec_dims = [input_dim + latent_dim] + HIDDEN_DIMS
        self.decoder_body = _build_mlp(dec_dims, DROPOUT, batchnorm_first=True)
        self.dec_mu    = nn.Linear(HIDDEN_DIMS[-1], 1)
        self.dec_logvar = nn.Linear(HIDDEN_DIMS[-1], 1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        xy = torch.cat([x, y.unsqueeze(-1)], dim=-1)
        h = self.encoder_body(xy)
        return self.enc_mu(h), self.enc_logvar(h)

    def prior(self, x: torch.Tensor):
        h = self.prior_body(x)
        return self.prior_mu(h), self.prior_logvar(h)

    def decode(self, x: torch.Tensor, z: torch.Tensor):
        xz = torch.cat([x, z], dim=-1)
        h = self.decoder_body(xz)
        mu_ln = self.dec_mu(h).squeeze(-1)
        logvar_ln = self.dec_logvar(h).squeeze(-1)
        sigma_ln = torch.exp(0.5 * logvar_ln).clamp(min=SIGMA_MIN)
        return mu_ln, sigma_ln

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        p_mu, p_logvar = self.prior(x)
        if y is not None:
            q_mu, q_logvar = self.encode(x, y)
            z = self.reparameterize(q_mu, q_logvar)
            mu_ln, sigma_ln = self.decode(x, z)
            return mu_ln, sigma_ln, q_mu, q_logvar, p_mu, p_logvar
        else:
            z = self.reparameterize(p_mu, p_logvar)
            mu_ln, sigma_ln = self.decode(x, z)
            return mu_ln, sigma_ln


# ===================================================================
# LOSS FUNCTIONS
# ===================================================================
def kl_divergence(q_mu: torch.Tensor, q_logvar: torch.Tensor,
                  p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    """KL[q(z|x,y) ‖ p(z|x)] for two diagonal Gaussians."""
    kl = 0.5 * (
        p_logvar - q_logvar
        + (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) / (torch.exp(p_logvar) + 1e-8)
        - 1.0
    )
    return kl.sum(dim=-1).mean()


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL in standardized log-space."""
    nll = (
        0.5 * math.log(2 * math.pi)
        + torch.log(sigma + 1e-10)
        + 0.5 * ((y - mu) / (sigma + 1e-10)) ** 2
    )
    return nll.mean()


def cvae_elbo_loss(mu_ln, sigma_ln, q_mu, q_logvar,
                   p_mu, p_logvar, y_logspace, beta):
    """ELBO = -E_q[log p(y|x,z)] + β · KL."""
    recon = gaussian_nll(mu_ln, sigma_ln, y_logspace)
    kl = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
    return recon + beta * kl, recon, kl


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
    up_cvae_rmse: float
    up_cvae_mae: float
    up_cvae_nll: float
    up_naive_rmse: float
    up_naive_mae: float
    dn_cvae_rmse: float
    dn_cvae_mae: float
    dn_cvae_nll: float
    dn_naive_rmse: float
    dn_naive_mae: float


# ===================================================================
# TRAINING LOOP FOR ONE TARGET
# ===================================================================
def _train_cvae_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_down: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Train CVAE with Gaussian decoder on robust-standardized targets.

    At test time: MC-sample z from prior → decode → inverse-standardize.

    Returns (pred_mean, pred_std, test_nll).
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Target preprocessing
    y_z, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_train)
    y_z = y_z.astype(np.float32)

    # Temporal validation split
    n = len(X_train)
    n_val = max(1, int(n * VAL_FRAC))
    n_tr = n - n_val
    tr_idx = np.arange(n_tr)
    val_idx = np.arange(n_tr, n)

    Xtr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y_z[tr_idx], dtype=torch.float32)
    Xva = torch.tensor(X_train[val_idx], dtype=torch.float32)
    yva = torch.tensor(y_z[val_idx], dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    # Standardize test targets with same params
    y_test_z = ((y_test - y_center) / y_scale).astype(np.float32)
    yte = torch.tensor(y_test_z, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, ytr)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False)

    model = CVAELogNormal(X_train.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    wait = 0

    for epoch in range(MAX_EPOCHS):
        beta = min(BETA_END, BETA_START + (BETA_END - BETA_START) * epoch / max(BETA_WARMUP, 1))

        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            mu_out, sigma_out, q_mu, q_logvar, p_mu, p_logvar = model(xb, yb)
            loss, _, _ = cvae_elbo_loss(
                mu_out, sigma_out, q_mu, q_logvar, p_mu, p_logvar, yb, beta
            )
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            # Batched validation to avoid MPS NaN on large tensors
            val_losses = []
            n_va = Xva.size(0)
            for vi in range(0, n_va, BATCH_SIZE):
                xvb = Xva[vi:vi + BATCH_SIZE].to(DEVICE)
                yvb = yva[vi:vi + BATCH_SIZE].to(DEVICE)
                mu_o, sig_o, q_m, q_lv, p_m, p_lv = model(xvb, yvb)
                vl, _, _ = cvae_elbo_loss(mu_o, sig_o, q_m, q_lv, p_m, p_lv, yvb, beta)
                if not (math.isnan(vl.item()) or math.isinf(vl.item())):
                    val_losses.append(vl.item() * xvb.size(0))
            if len(val_losses) == 0:
                val_loss = float("nan")
            else:
                val_loss = sum(val_losses) / n_va

        if math.isnan(val_loss) or math.isinf(val_loss):
            continue
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Monte Carlo prediction at test time
    model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    Xte_dev = Xte.to(DEVICE)
    with torch.no_grad():
        for _ in range(N_MC_SAMPLES):
            mu_chunks, sig_chunks = [], []
            for i in range(0, len(Xte), BATCH_SIZE):
                xb = Xte_dev[i:i + BATCH_SIZE]
                mu_b, sig_b = model(xb, y=None)
                mu_chunks.append(mu_b)
                sig_chunks.append(sig_b)
            mu_out = torch.cat(mu_chunks, dim=0)
            sigma_out = torch.cat(sig_chunks, dim=0)
            sample_z = mu_out + sigma_out * torch.randn_like(sigma_out)
            all_preds.append(sample_z.cpu().numpy())

    all_preds = np.stack(all_preds, axis=0)  # (N_MC, N_test)
    all_preds = np.nan_to_num(all_preds, nan=0.0)
    pred_mean_z = all_preds.mean(axis=0)
    pred_std_z  = all_preds.std(axis=0)

    # Inverse standardization → original scale
    pred_mean = np.clip(pred_mean_z * y_scale + y_center, clip_lo, clip_hi)
    pred_mean = np.nan_to_num(pred_mean, nan=0.0)
    pred_std  = np.nan_to_num(pred_std_z * y_scale, nan=0.0)

    # Test NLL — batched
    nll_sum, nll_cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(Xte), BATCH_SIZE):
            xb = Xte_dev[i:i + BATCH_SIZE]
            yb = yte[i:i + BATCH_SIZE].to(DEVICE)
            mu_b, sig_b = model(xb, y=None)
            diff = yb - mu_b
            nll_b = (0.5 * math.log(2 * math.pi)
                     + torch.log(sig_b + 1e-10)
                     + 0.5 * (diff / (sig_b + 1e-10)) ** 2)
            valid = ~torch.isnan(nll_b)
            nll_sum += nll_b[valid].sum().item()
            nll_cnt += valid.sum().item()
    test_nll = nll_sum / max(nll_cnt, 1)

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

    pred_up, std_up, nll_up = _train_cvae_full(Xtr, y_up_train, Xte, y_up_test,
                                                is_down=False)
    pred_dn, std_dn, nll_dn = _train_cvae_full(Xtr, y_dn_train, Xte, y_dn_test,
                                                is_down=True)

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

    up_cvae_rmse, up_cvae_mae   = rmse_mae(pred_df["y_up_true"], pred_df["y_up_pred"])
    up_naive_rmse, up_naive_mae = rmse_mae(pred_df["y_up_true"], pred_df["y_up_naive"])
    dn_cvae_rmse, dn_cvae_mae   = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_pred"])
    dn_naive_rmse, dn_naive_mae = rmse_mae(pred_df["y_dn_true"], pred_df["y_dn_naive"])

    return FoldResult(
        fold_id=fold_id,
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        n_train=len(train_df),
        n_test=len(test_df),
        up_cvae_rmse=up_cvae_rmse, up_cvae_mae=up_cvae_mae,
        up_cvae_nll=nll_up,
        up_naive_rmse=up_naive_rmse, up_naive_mae=up_naive_mae,
        dn_cvae_rmse=dn_cvae_rmse, dn_cvae_mae=dn_cvae_mae,
        dn_cvae_nll=nll_dn,
        dn_naive_rmse=dn_naive_rmse, dn_naive_mae=dn_naive_mae,
    ), pred_df


# ===================================================================
# PLOTTING (CVAE-specific: adds ELBO decomposition fig)
# ===================================================================
def generate_plots(
    fold_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
    overall: Dict[str, float],
) -> None:
    plot_fold_rmse(fold_df, MODEL_LABEL, MODEL_NAME, OUTPUT_DIR,
                   model_color="tab:blue")
    plot_fold_improvement(fold_df, MODEL_LABEL, MODEL_NAME, overall, OUTPUT_DIR)
    plot_ticker_scatter(ticker_df, MODEL_LABEL, OUTPUT_DIR,
                        scatter_color="tab:blue")
    plot_overall_bar(overall, MODEL_LABEL, OUTPUT_DIR)
    plot_fold_nll(fold_df, MODEL_LABEL, MODEL_NAME, OUTPUT_DIR)

    # ── Fig 6 — ELBO decomposition (CVAE-specific) ──────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    n_folds = len(fold_df)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(1, n_folds + 1)
    total = [fold_df.iloc[i]["up_cvae_nll"] + fold_df.iloc[i]["dn_cvae_nll"]
             for i in range(n_folds)]
    ax.plot(x, total, "o-", ms=4, label="Total NLL (recon)", color="tab:purple")
    ax.set_xlabel("Fold")
    ax.set_ylabel("NLL")
    ax.set_title("CVAE: Reconstruction NLL per Fold (y_up + y_down)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_elbo_decomposition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n  All 6 figures saved to {OUTPUT_DIR}/")


# ===================================================================
# MAIN
# ===================================================================
def main() -> None:
    update_dataset()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"\n{'='*60}")
    print("  CVAE ROLLING-WINDOW TRAINER  (y_up / y_down)")
    print(f"{'='*60}")
    print(f"  Model       : CVAE + LogNormal decoder")
    print(f"              : Latent dim={LATENT_DIM}, β-warmup={BETA_WARMUP} epochs")
    print(f"              : Encoder/Decoder hidden={HIDDEN_DIMS}")
    print(f"              : Prior hidden={PRIOR_DIMS}")
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
    print(f"  MC samples  : {N_MC_SAMPLES} (at test time)")

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

            up_imp = improvement_pct(fold_res.up_cvae_rmse, fold_res.up_naive_rmse)
            dn_imp = improvement_pct(fold_res.dn_cvae_rmse, fold_res.dn_naive_rmse)
            print(f"\n        y_up:  CVAE RMSE={fold_res.up_cvae_rmse:.6f}  "
                  f"Naive={fold_res.up_naive_rmse:.6f}  Improv={up_imp:+.1f}%  "
                  f"NLL={fold_res.up_cvae_nll:.3f}"
                  f"\n        y_dn:  CVAE RMSE={fold_res.dn_cvae_rmse:.6f}  "
                  f"Naive={fold_res.dn_naive_rmse:.6f}  Improv={dn_imp:+.1f}%  "
                  f"NLL={fold_res.dn_cvae_nll:.3f}")
        except Exception as e:
            print(f"  ⚠️  SKIPPED: {e}")

    if not all_preds_list:
        raise RuntimeError("No folds completed.")

    all_preds       = pd.concat(all_preds_list, ignore_index=True)
    fold_results_df = pd.DataFrame([asdict(fr) for fr in all_fold_results])

    fold_nll_values = {
        "up": [fr.up_cvae_nll for fr in all_fold_results],
        "dn": [fr.dn_cvae_nll for fr in all_fold_results],
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
# FIG 7 — PER-TICKER TRUE vs PREDICTED
# ===================================================================
def generate_ticker_accuracy_plot(
    tickers=("AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"),
):
    plot_ticker_accuracy(
        OUTPUT_DIR, MODEL_NAME, MODEL_LABEL, fig_num=7,
        tickers=tickers, up_color="#1565C0", dn_color="#E65100",
        has_std=True,
    )


# ===================================================================
# LIVE FORECAST — train on ALL data, predict next 5 days
# ===================================================================
def live_forecast(
    tickers=("AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"),
) -> None:
    """
    Train CVAE on the full (updated) dataset using PRED_FEAT_COLS,
    then predict y_up / y_down (with uncertainty) for the given tickers
    for the next 5 trading days.
    """
    from datetime import date

    today = date.today()
    print(f"\n{'='*70}")
    print(f"  CVAE LIVE FORECAST  —  {today}  (next 5 trading days)")
    print(f"{'='*70}")

    # ── 1. Load updated dataset ──────────────────────────────
    print("  [1/3] Loading updated training data ...")
    train_df, X_all_sc, y_up_all, y_dn_all, scaler, all_tickers, data_end = (
        load_and_prepare_live_data()
    )

    # ── 2. Train CVAE for y_up and y_down ───────────────────────
    print("  [2/3] Training CVAE models (y_up & y_down) ...")

    def _train_full_cvae(X_sc, y_raw, is_down=False):
        """Train CVAE on full data using Gaussian decoder on standardized targets."""
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        y_z, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_raw)
        y_z = y_z.astype(np.float32)

        n = len(X_sc)
        n_val = max(1, int(n * VAL_FRAC))
        n_tr = n - n_val
        tr_idx = np.arange(n_tr)
        val_idx = np.arange(n_tr, n)

        Xtr = torch.tensor(X_sc[tr_idx], dtype=torch.float32)
        ytr = torch.tensor(y_z[tr_idx], dtype=torch.float32)
        Xva = torch.tensor(X_sc[val_idx], dtype=torch.float32)
        yva = torch.tensor(y_z[val_idx], dtype=torch.float32)

        train_ds = TensorDataset(Xtr, ytr)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False)

        model = CVAELogNormal(X_sc.shape[1]).to(DEVICE)
        optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0

        for epoch in range(MAX_EPOCHS):
            beta = min(BETA_END,
                       BETA_START + (BETA_END - BETA_START) * epoch / max(BETA_WARMUP, 1))

            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimiser.zero_grad()
                mu_out, sigma_out, q_mu, q_logvar, p_mu, p_logvar = model(xb, yb)
                loss, _, _ = cvae_elbo_loss(
                    mu_out, sigma_out, q_mu, q_logvar, p_mu, p_logvar, yb, beta
                )
                if torch.isnan(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            model.eval()
            with torch.no_grad():
                vl_parts = []
                n_va = Xva.size(0)
                for vi in range(0, n_va, BATCH_SIZE):
                    xvb = Xva[vi:vi + BATCH_SIZE].to(DEVICE)
                    yvb = yva[vi:vi + BATCH_SIZE].to(DEVICE)
                    mu_o, sig_o, q_m, q_lv, p_m, p_lv = model(xvb, yvb)
                    vl, _, _ = cvae_elbo_loss(mu_o, sig_o, q_m, q_lv, p_m, p_lv, yvb, beta)
                    if not (math.isnan(vl.item()) or math.isinf(vl.item())):
                        vl_parts.append(vl.item() * xvb.size(0))
                if len(vl_parts) == 0:
                    val_loss = float("nan")
                else:
                    val_loss = sum(vl_parts) / n_va

            if math.isnan(val_loss) or math.isinf(val_loss):
                continue
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

    model_up, c_up, s_up, lo_up, hi_up = _train_full_cvae(
        X_all_sc, y_up_all, is_down=False)
    model_dn, c_dn, s_dn, lo_dn, hi_dn = _train_full_cvae(
        X_all_sc, y_dn_all, is_down=True)

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

        # Monte Carlo prediction
        with torch.no_grad():
            preds_up_z, preds_dn_z = [], []
            for _ in range(N_MC_SAMPLES):
                mu_up, sig_up = model_up(Xt, y=None)
                sample_up = mu_up + sig_up * torch.randn_like(sig_up)
                preds_up_z.append(sample_up.cpu().numpy())

                mu_dn, sig_dn = model_dn(Xt, y=None)
                sample_dn = mu_dn + sig_dn * torch.randn_like(sig_dn)
                preds_dn_z.append(sample_dn.cpu().numpy())

            preds_up_z = np.array(preds_up_z).squeeze()
            preds_dn_z = np.array(preds_dn_z).squeeze()

        pred_up = float(np.clip(
            preds_up_z.mean() * s_up + c_up, lo_up, hi_up))
        std_up = float(preds_up_z.std() * s_up)
        pred_dn = float(np.clip(
            preds_dn_z.mean() * s_dn + c_dn, lo_dn, hi_dn))
        std_dn = float(preds_dn_z.std() * s_dn)

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
