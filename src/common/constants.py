"""
common.constants
================
Shared constants, paths, and configuration used by all model trainers.
"""

from __future__ import annotations

from pathlib import Path

import torch

# ===================================================================
# PATHS
# ===================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "dataset_final_cleaned.csv"

# ===================================================================
# COLUMN NAMES
# ===================================================================
DATE_COL     = "date"
TICKER_COL   = "ticker"
PRICE_COL    = "adj_close"
TARGET_UP    = "y_up"              # max(P_{t+1:t+5}) / P_t - 1
TARGET_DN    = "y_down"            # min(P_{t+1:t+5}) / P_t - 1

# ===================================================================
# ROLLING-WINDOW SETUP
# ===================================================================
TRAIN_MONTHS   = 12
TEST_MONTHS    = 3
STEP_MONTHS    = 3
PURGE_GAP_DAYS = 5                 # avoid label overlap with 5-day forward window

RANDOM_SEED = 42

# ===================================================================
# COLUMNS TO EXCLUDE FROM MODEL FEATURES
# ===================================================================
# Base exclusion set used by NN models (FNN, MDN, CVAE).
# XGBoost baseline uses a slightly different set (no TICKER_COL).
EXCLUDE_COLS_NN = {
    DATE_COL,
    TICKER_COL,
    TARGET_UP,
    TARGET_DN,
    "target_ret_5d",
}

# ===================================================================
# SHARED NN HYPER-PARAMETERS
# ===================================================================
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = [0.3, 0.3, 0.2]
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 512
MAX_EPOCHS   = 200
PATIENCE     = 15
VAL_FRAC     = 0.15        # temporal split: last 15% of training window
WINSORIZE_PCT = (1.0, 99.0)

# ===================================================================
# LIVE FORECAST
# ===================================================================
FORECAST_TICKERS = ["AMD", "TSLA", "MU", "GOOGL", "MSFT", "AMZN", "NVDA", "SNDK"]

# Features used for live prediction (subset buildable from Yahoo data)
PRED_FEAT_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "vol_20d", "skew_20d", "kurt_20d",
    "momentum_20d", "momentum_60d",
    "volume_zscore", "hl_spread", "oc_return",
    "vix", "vix_change_9d", "vvix", "vix_ma20", "vix_slope",
    "bond_yield_3m", "bond_yield_5y", "gld_ret", "wti_ret", "tlt_ret", "ief_ret",
    "dxy", "dxy_ret_1d", "dxy_ma20",
]

# ===================================================================
# DEVICE DETECTION
# ===================================================================
def get_device() -> torch.device:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
