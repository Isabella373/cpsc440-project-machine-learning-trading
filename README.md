# Equity Return Prediction: Probabilistic vs Deterministic Models

## Overview

Comparing three approaches for cross-sectional equity return prediction:
1. **Feedforward Neural Network** — point estimates
2. **Mixture Density Network (MDN)** — conditional return distributions
3. **XGBoost** — baseline tabular model

**Target**: 5-day forward log return $y_{i,t} = \log(P_{i,t+5}) - \log(P_{i,t})$

**Universe**: S&P 500 top 100 by dollar volume · Daily data Jan 2018 – Mar 2026

## Repository Structure

```
src/
├── data_processing/          # Feature engineering & data cleaning
│   ├── build_dataset.py      # Download & build features (yfinance + FRED)
│   ├── validate_dataset.py   # Anomaly detection & cleaning
│   └── ticker_candidates.py  # S&P 500 candidate universe
├── train_baseline_rolling.py # Rolling-window XGBoost baseline
├── config.py
data/
├── raw/                      # Cached downloads (parquet)
├── processed/                # Cleaned dataset CSVs
models/                       # Saved checkpoints
results/                      # Predictions, metrics, reports
notebooks/                    # Exploratory analysis
docs/                         # Proposal & documentation
```

## Features (49 columns)

| Category | Features |
|---|---|
| **Returns** | `ret_1d`, `ret_5d`, `ret_20d` |
| **Volatility** | `vol_20d`, `skew_20d`, `kurt_20d` |
| **Momentum** | `momentum_20d`, `momentum_60d`, `rsi_14`, `macd` |
| **Moving Avg Ratios** | `ma_ratio_5`, `ma_ratio_20` |
| **Volume** | `volume_zscore`, `dollar_volume`, `hl_spread`, `oc_return` |
| **VIX / Regime** | VIX, VIX Δ9d, VVIX, VIX MA |
| **Macro** | Bond yields (1M/1Y), NFP, unemployment, initial claims |
| **Cross-asset** | GLD, WTI, TLT, IEF, USD index |
| **Calendar** | Holiday, Friday, month/quarter-end flags |

## Validation Strategy

**Rolling Time-Series Split** (no data leakage):
- **Train**: 12-month window
- **Purge gap**: 5 trading days (avoids label overlap)
- **Test**: next 3 months
- **Step**: roll forward 3 months → ~28 folds over 2018–2026

## Evaluation Metrics

| Aspect | Metrics |
|---|---|
| **Point prediction** | RMSE, MAE, Spearman rank corr, daily IC |
| **Ranking quality** | Top-10% minus bottom-10% return spread |
| **Probabilistic (MDN)** | NLL, calibration curves, 90% coverage |

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. XGBoost on macOS (Apple Silicon)

XGBoost and LightGBM require OpenMP (`libomp`). On macOS without Homebrew:

```bash
# Copy libomp from scikit-learn (already bundled)
sudo mkdir -p /opt/homebrew/opt/libomp/lib
sudo cp ~/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs/libomp.dylib \
  /opt/homebrew/opt/libomp/lib/

# Add to shell config
echo 'export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_FALLBACK_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc

# Verify
python3 -c "import xgboost; print('XGBoost works')"
```

> If you have Homebrew, simply run `brew install libomp`.

### 3. Build dataset

```bash
python3 src/data_processing/build_dataset.py
python3 src/data_processing/validate_dataset.py
```

### 4. Train baseline model

```bash
python3 src/train_baseline_rolling.py
```

Results are saved to `results/baseline_rolling/`.

## References
- [Project Proposal](./docs/PROPOSAL.md)
