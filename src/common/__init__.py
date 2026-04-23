"""
Common utilities shared across all model training scripts.

This package extracts duplicated functions and patterns from
train_baseline_rolling.py, train_fnn_rolling.py, train_mdn_rolling.py,
and train_cvae_rolling.py into reusable modules:

  common.constants   — shared column names, paths, rolling-window config
  common.metrics     — RMSE/MAE helpers, ticker metrics, overall summary
  common.data        — data prep, feature scaling, target preprocessing
  common.training    — PyTorch training loop with early stopping
  common.evaluation  — main() orchestrator, fold runner, result saving
  common.live        — live forecast pipeline
  common.plotting    — shared plotting functions
"""
