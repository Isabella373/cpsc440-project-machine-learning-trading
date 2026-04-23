"""
Configuration file for the equity return prediction project.
"""

from pathlib import Path
from datetime import datetime

import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "start_date": "2018-01-02",
    "end_date": datetime.today().strftime("%Y-%m-%d"),  # always up-to-date
    "universe_size": 100,
    "universe_selection": "fixed_top100_by_dollar_volume",
    "universe_reference_date": "2025-12-31",
    "target_horizon": 5,  # 5-day forward return
}

# Feature configuration
FEATURE_CONFIG = {
    "price_volume_features": [
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "skew_20d", "kurt_20d",
        "momentum_20d", "momentum_60d", "rsi_14", "macd",
        "ma_5", "ma_20", "ma_ratio_5", "ma_ratio_20",
        "volume_zscore", "dollar_volume", "hl_spread", "oc_return",
    ],
    "market_regime_features": [
        "vix", "vix_9d_change", "vvix", "vix_ma_slope",
        "bond_yield_1m", "bond_yield_1y",
        "nfp", "unemployment_rate", "jobless_claims",
        "usd_index", "sp500_forward_pe", "etf_flows",
        "put_call_ratio", "iv_skew",
        "gold_return", "oil_return", "bond_return",
        "is_holiday", "is_friday", "is_month_end", "is_quarter_end",
    ],
}

# Model configuration
MODEL_CONFIG = {
    "feedforward_nn": {
        "hidden_dims": [256, 128, 64],
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
    },
    "mdn": {
        "hidden_dims": [256, 128, 64],
        "num_mixtures": 3,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}

# Training configuration
TRAINING_CONFIG = {
    "random_seed": 42,
    "use_explicit_validation_window": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Automatically use GPU if available
    "early_stopping_patience": 20,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["mse", "mae", "r2", "directional_accuracy"],
    "probabilistic_metrics": ["nll", "calibration", "coverage"],
    "percentile_range": [0.1, 0.25, 0.5, 0.75, 0.9],
}

# Time-series rolling window configuration
ROLLING_WINDOW_CONFIG = {
    "train_months": 12,
    "val_months": 1,
    "test_months": 1,
    "step_months": 1,
}
