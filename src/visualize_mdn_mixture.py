"""
visualize_mdn_mixture.py
========================

Generate a Gaussian mixture density visualization that shows what
the MDN actually learns.  Trains on the last rolling fold, extracts
(pi, mu, sigma) for selected predictions, and plots individual
Gaussian components + the combined mixture density.

Outputs
-------
    results/mdn_rolling/fig7_gaussian_mixture.png
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Shared modules ──────────────────────────────────────────────────
from common.constants import (
    PROJECT_ROOT, DATA_PATH,
    DATE_COL, TICKER_COL, TARGET_UP, TARGET_DN,
    RANDOM_SEED, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS, PURGE_GAP_DAYS,
    HIDDEN_DIMS, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    PATIENCE, VAL_FRAC,
    get_device,
)
from common.data import (
    check_required_columns, prepare_data_nn, infer_feature_columns,
    encode_and_scale, robust_standardize, inverse_standardize,
    build_rolling_windows, update_dataset,
)
from common.evaluation import split_fold

# ── MDN model & loss (re-used from train_mdn_rolling) ──────────────
N_COMPONENTS = 5
SIGMA_MIN    = 1e-4
DEVICE       = get_device()
OUTPUT_DIR   = PROJECT_ROOT / "results" / "mdn_rolling"


class MDN(nn.Module):
    def __init__(self, input_dim: int, n_components: int = N_COMPONENTS):
        super().__init__()
        self.K = n_components
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h, d in zip(HIDDEN_DIMS, DROPOUT):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.pi_head    = nn.Linear(prev, n_components)
        self.mu_head    = nn.Linear(prev, n_components)
        self.sigma_head = nn.Linear(prev, n_components)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        pi    = torch.softmax(self.pi_head(h), dim=-1)
        mu    = self.mu_head(h)
        sigma = torch.exp(self.sigma_head(h)) + SIGMA_MIN
        return pi, mu, sigma


def mdn_nll_loss(pi, mu, sigma, y):
    y = y.unsqueeze(-1)
    log_pi   = torch.log(pi + 1e-10)
    log_norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma + 1e-10)
    log_exp  = -0.5 * ((y - mu) / (sigma + 1e-10)) ** 2
    log_probs = log_pi + log_norm + log_exp
    return -torch.logsumexp(log_probs, dim=-1).mean()


def mdn_mean(pi, mu):
    return (pi * mu).sum(dim=-1)


def mdn_std(pi, mu, sigma):
    mean = mdn_mean(pi, mu)
    var = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1) - mean ** 2
    return torch.sqrt(var.clamp(min=1e-10))


# ───────────────────────────────────────────────────────────────────
# TRAIN ONE MDN MODEL
# ───────────────────────────────────────────────────────────────────
def train_mdn(X_train, y_train, X_test, y_test):
    """
    Train an MDN and return (model, y_center, y_scale, clip_lo, clip_hi,
    pi_test, mu_test, sigma_test) with the MDN head outputs on the
    test set in **standardized** space.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    y_proc, clip_lo, clip_hi, y_center, y_scale = robust_standardize(y_train)

    n = len(X_train)
    n_val = max(1, int(n * VAL_FRAC))
    n_tr  = n - n_val

    Xtr = torch.tensor(X_train[:n_tr],    dtype=torch.float32)
    ytr = torch.tensor(y_proc[:n_tr],      dtype=torch.float32)
    Xva = torch.tensor(X_train[n_tr:],     dtype=torch.float32)
    yva = torch.tensor(y_proc[n_tr:],      dtype=torch.float32)
    Xte = torch.tensor(X_test,             dtype=torch.float32)

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=False)

    model = MDN(X_train.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_nll_loss(pi, mu, sigma, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pi, mu, sigma = model(Xva.to(DEVICE))
            vl = mdn_nll_loss(pi, mu, sigma, yva.to(DEVICE)).item()
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pi_t, mu_t, sigma_t = model(Xte.to(DEVICE))
        pi_t  = pi_t.cpu().numpy()
        mu_t  = mu_t.cpu().numpy()
        sigma_t = sigma_t.cpu().numpy()

    # Convert mu and sigma back to original scale
    mu_orig    = mu_t * y_scale + y_center
    sigma_orig = sigma_t * y_scale

    return pi_t, mu_orig, sigma_orig, y_center, y_scale


# ───────────────────────────────────────────────────────────────────
# GAUSSIAN PDF HELPER
# ───────────────────────────────────────────────────────────────────
def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def mixture_pdf(x, pi, mu, sigma):
    pdf = np.zeros_like(x)
    for k in range(len(pi)):
        pdf += pi[k] * gaussian_pdf(x, mu[k], sigma[k])
    return pdf


# ───────────────────────────────────────────────────────────────────
# MAIN VISUALIZATION
# ───────────────────────────────────────────────────────────────────
def main():
    update_dataset()
    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    check_required_columns(df)
    df = prepare_data_nn(df)
    feature_cols = infer_feature_columns(df)

    windows = build_rolling_windows(
        df[DATE_COL], TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS, PURGE_GAP_DAYS)

    # Use the second-to-last fold (not the last, which may be short)
    fold_idx = max(0, len(windows) - 2)
    tr_s, tr_e, te_s, te_e = windows[fold_idx]
    fold_id = fold_idx + 1
    print(f"Using fold {fold_id}: train [{tr_s.date()} -> {tr_e.date()}]  "
          f"test [{te_s.date()} -> {te_e.date()}]")

    train_df, test_df, X_train_raw, X_test_raw = split_fold(
        df, feature_cols, fold_id, tr_s, tr_e, te_s, te_e)
    Xtr, Xte, _ = encode_and_scale(X_train_raw, X_test_raw)

    y_up_train = train_df[TARGET_UP].values.astype(np.float32)
    y_dn_train = train_df[TARGET_DN].values.astype(np.float32)
    y_up_test  = test_df[TARGET_UP].values.astype(np.float32)
    y_dn_test  = test_df[TARGET_DN].values.astype(np.float32)

    tickers_test = test_df[TICKER_COL].values

    print("Training y_up MDN ...")
    pi_up, mu_up, sig_up, _, _ = train_mdn(Xtr, y_up_train, Xte, y_up_test)
    print("Training y_dn MDN ...")
    pi_dn, mu_dn, sig_dn, _, _ = train_mdn(Xtr, y_dn_train, Xte, y_dn_test)

    # ── Select representative predictions ─────────────────────
    # Pick examples from different tickers & regimes
    target_tickers = ["NVDA", "AAPL", "TSLA", "JPM", "XOM", "KO"]
    sample_indices = []
    sample_labels  = []

    for tk in target_tickers:
        mask = tickers_test == tk
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        # Pick one from middle of test set
        mid = idxs[len(idxs) // 2]
        sample_indices.append(mid)
        date_str = str(test_df.iloc[mid][DATE_COL])[:10]
        sample_labels.append(f"{tk} ({date_str})")

    n_samples = len(sample_indices)
    if n_samples == 0:
        print("No target tickers found in test fold.")
        return

    # ── Create figure ─────────────────────────────────────────
    fig = plt.figure(figsize=(20, 5 * n_samples))
    gs = gridspec.GridSpec(n_samples, 2, hspace=0.35, wspace=0.25)

    component_colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for row, (idx, label) in enumerate(zip(sample_indices, sample_labels)):
        for col, (pi, mu, sig, y_true, target_name) in enumerate([
            (pi_up[idx], mu_up[idx], sig_up[idx], y_up_test[idx], "y_up"),
            (pi_dn[idx], mu_dn[idx], sig_dn[idx], y_dn_test[idx], "y_dn"),
        ]):
            ax = fig.add_subplot(gs[row, col])

            # Determine x range
            mix_mean = np.sum(pi * mu)
            mix_std  = np.sqrt(np.sum(pi * (sig**2 + mu**2)) - mix_mean**2)
            x_lo = min(mix_mean - 4 * mix_std, y_true - 2 * mix_std,
                       np.min(mu - 3 * sig))
            x_hi = max(mix_mean + 4 * mix_std, y_true + 2 * mix_std,
                       np.max(mu + 3 * sig))
            x = np.linspace(x_lo, x_hi, 500)

            # Plot individual components
            for k in range(len(pi)):
                comp_pdf = pi[k] * gaussian_pdf(x, mu[k], sig[k])
                ax.fill_between(x, comp_pdf, alpha=0.15,
                                color=component_colors[k % len(component_colors)])
                ax.plot(x, comp_pdf, linewidth=1.2, alpha=0.7,
                        color=component_colors[k % len(component_colors)],
                        label=f"k={k+1}: π={pi[k]:.2f}, μ={mu[k]:.4f}, σ={sig[k]:.4f}")

            # Plot mixture density
            mix = mixture_pdf(x, pi, mu, sig)
            ax.plot(x, mix, "k-", linewidth=2.5, label="Mixture density", zorder=5)

            # True value
            ax.axvline(y_true, color="red", linestyle="--", linewidth=2,
                       label=f"True {target_name} = {y_true:.4f}", zorder=6)

            # Mixture mean
            ax.axvline(mix_mean, color="green", linestyle="--", linewidth=1.5,
                       label=f"Pred mean = {mix_mean:.4f}", zorder=6)

            # ±1 std shading
            ax.axvspan(mix_mean - mix_std, mix_mean + mix_std,
                       alpha=0.08, color="green",
                       label=f"±1σ = [{mix_mean-mix_std:.4f}, {mix_mean+mix_std:.4f}]")

            ax.set_xlabel("Return", fontsize=11)
            ax.set_ylabel("Probability Density", fontsize=11)
            ax.set_title(f"{label}  —  {target_name.replace('_', ' ').upper()} prediction",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"MDN Gaussian Mixture Decomposition  (K={N_COMPONENTS} components)\n"
        f"Fold {fold_id}: test {te_s.date()} → {te_e.date()}",
        fontsize=16, fontweight="bold", y=1.01
    )

    out_path = OUTPUT_DIR / "fig7_gaussian_mixture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✅  Saved: {out_path}")

    # ── Also create a summary distribution comparison ─────────
    _plot_distribution_comparison(
        pi_up, mu_up, sig_up, y_up_test, tickers_test,
        pi_dn, mu_dn, sig_dn, y_dn_test,
    )


def _plot_distribution_comparison(pi_up, mu_up, sig_up, y_up_test, tickers,
                                  pi_dn, mu_dn, sig_dn, y_dn_test):
    """
    Create a figure showing how mixture parameters (pi, mu, sigma) are
    distributed across all test predictions — i.e., what the MDN learns
    at a population level.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (pi_all, mu_all, sig_all, y_true, target) in enumerate([
        (pi_up, mu_up, sig_up, y_up_test, "y_up"),
        (pi_dn, mu_dn, sig_dn, y_dn_test, "y_dn"),
    ]):
        # (a) Distribution of mixture weights pi
        ax = axes[row, 0]
        for k in range(pi_all.shape[1]):
            ax.hist(pi_all[:, k], bins=50, alpha=0.5,
                    label=f"k={k+1}", density=True)
        ax.set_xlabel("π (mixture weight)")
        ax.set_ylabel("Density")
        ax.set_title(f"{target}: Distribution of π across all test predictions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (b) Distribution of component means mu
        ax = axes[row, 1]
        for k in range(mu_all.shape[1]):
            ax.hist(mu_all[:, k], bins=50, alpha=0.5,
                    label=f"k={k+1}", density=True)
        ax.axvline(np.mean(y_true), color="red", linestyle="--", linewidth=2,
                   label=f"True mean={np.mean(y_true):.4f}")
        ax.set_xlabel("μ (component mean)")
        ax.set_ylabel("Density")
        ax.set_title(f"{target}: Distribution of μ across all test predictions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (c) Distribution of component std devs sigma
        ax = axes[row, 2]
        for k in range(sig_all.shape[1]):
            ax.hist(sig_all[:, k], bins=50, alpha=0.5,
                    label=f"k={k+1}", density=True)
        ax.set_xlabel("σ (component std dev)")
        ax.set_ylabel("Density")
        ax.set_title(f"{target}: Distribution of σ across all test predictions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "MDN Learned Parameter Distributions  (π, μ, σ)  —  all test predictions",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUT_DIR / "fig8_mixture_parameter_distributions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Saved: {out_path}")


if __name__ == "__main__":
    main()
