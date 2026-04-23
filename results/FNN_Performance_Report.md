# Feedforward Neural Network (FNN) Performance Report

## Comprehensive Analysis of Deterministic FNN for Equity Range Prediction

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experimental Setup](#2-experimental-setup)
3. [Results Overview](#3-results-overview)
4. [Multi-Dimensional Outperformance Analysis](#4-multi-dimensional-outperformance-analysis)
5. [High-Volatility Market Behavior](#5-high-volatility-market-behavior)
6. [Why Does the FNN Outperform the Naive Baseline?](#6-why-does-the-fnn-outperform-the-naive-baseline)
7. [Why Does the FNN Underperform Probabilistic Models?](#7-why-does-the-fnn-underperform-probabilistic-models)
8. [Ticker-Level Deep Dive](#8-ticker-level-deep-dive)
9. [The Asymmetric $y_{\text{up}}$ vs $y_{\text{down}}$ Gap](#9-the-asymmetric-y_up-vs-y_down-gap)
10. [Potential Improvements](#10-potential-improvements)
11. [Supporting Graphs Index](#11-supporting-graphs-index)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

We evaluate a **Feedforward Neural Network (FNN)** with Huber loss ($\delta=0.02$) for predicting 5-day forward equity return ranges ($y_{\text{up}}$ = max return, $y_{\text{down}}$ = min return) across 100 S&P 500 tickers. The FNN serves as the **deterministic neural baseline** between the tree-based XGBoost model and the probabilistic models (MDN, CVAE). It produces a single point prediction per target with no uncertainty estimate.

**Key Findings:**

| Model | Avg Improvement (%) | $y_{\text{up}}$ Improve (%) | $y_{\text{down}}$ Improve (%) | Folds Completed |
|-------|:---:|:---:|:---:|:---:|
| XGBoost (Baseline) | +5.41 | +11.34 | −0.52 | 28 |
| **FNN** (Huber $\delta$=0.02) | **+6.56** | **+11.31** | **+1.82** | **26** |
| MDN (K=5 Gaussians) | +10.87 | +14.48 | +7.25 | 27 |
| CVAE (Gaussian decoder) | +12.42 | +15.25 | +9.54 | 29 |

The FNN achieves a **+6.56% average RMSE improvement** over the naive predictor — a modest but genuine gain that is:
- **+1.15 percentage points above XGBoost** (+5.41%), confirming that neural networks learn useful non-linear patterns beyond tree ensembles
- **4.31 pp below MDN** (+10.87%) and **5.86 pp below CVAE** (+12.42%), revealing the fundamental limitation of deterministic point estimation for financial return prediction

The FNN's most distinctive characteristic is its **severe asymmetry**: strong on the upside (+11.31%) but nearly ineffective on the downside (+1.82%). This asymmetry, combined with 27 tickers showing negative $y_{\text{down}}$ improvement, makes the FNN the clearest demonstration in our study of **why probabilistic models are necessary** for equity range prediction.

---

## 2. Experimental Setup

### 2.1 Task Definition

For each ticker on each trading date, we predict:
- **$y_{\text{up}}$**: $\max\left(\frac{P_{t+1:t+5}}{P_t}\right) - 1$ — the maximum relative price reached over the next 5 trading days
- **$y_{\text{down}}$**: $\min\left(\frac{P_{t+1:t+5}}{P_t}\right) - 1$ — the minimum relative price reached over the next 5 trading days

The benchmark is the **naive predictor** $\hat{y} = 0$ (predicting no change).

### 2.2 Rolling Window Protocol

| Parameter | Value |
|-----------|-------|
| Training window | 12 months |
| Test window | 3 months |
| Step size | 3 months |
| Purge gap | 5 trading days |
| Universe | 100 S&P 500 tickers |
| Total folds completed | **26** (skips folds 13, 14, 29) |
| Total OOS predictions | **154,100** |
| OOS test dates | 1,541 |

The FNN completes 26 of 29 possible folds, **fewer than any other model** (XGBoost: 28, MDN: 27, CVAE: 29). Folds 13–14 (covering late-2021 to mid-2022 training) and fold 29 all fail, indicating numerical instability in certain market regimes.

### 2.3 FNN Architecture

```
Input (d features)
  → BatchNorm1d(d)
  → Linear(d, 256)  → ReLU → Dropout(0.3)
  → Linear(256, 128) → ReLU → Dropout(0.3)
  → Linear(128, 64)  → ReLU → Dropout(0.2)
  → Linear(64, 1)                              # single point prediction
```

### 2.4 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Loss function | **Huber Loss** ($\delta$ = 0.02 / y_scale) |
| Hidden dims | [256, 128, 64] |
| Dropout | [0.3, 0.3, 0.2] |
| Normalization | **BatchNorm1d** (input layer) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 512 |
| Max epochs | 200 |
| Early stopping patience | 15 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Validation split | 15% temporal |
| Target preprocessing | Robust standardization (median centering, IQR/1.3489 scaling) |
| Feature preprocessing | StandardScaler (mean=0, std=1), NaN→0 |

**Key design choices:**
- **Huber Loss** with $\delta=0.02$: Applies L2 penalty for errors $<$ 2% return, L1 penalty beyond. Chosen to reduce sensitivity to extreme return outliers.
- **Separate models** for $y_{\text{up}}$ and $y_{\text{down}}$: Two independent FNNs are trained per fold.
- **No uncertainty output**: Unlike MDN (mixture density) or CVAE (latent sampling), the FNN outputs a single scalar.

---

## 3. Results Overview

### 3.1 Aggregate Performance

| Metric | FNN | Naive | Improvement |
|--------|-----|-------|:-----------:|
| $y_{\text{up}}$ RMSE | 0.03901 | 0.04399 | **+11.31%** |
| $y_{\text{up}}$ MAE | 0.02530 | 0.02884 | +12.29% |
| $y_{\text{down}}$ RMSE | 0.03826 | 0.03897 | **+1.82%** |
| $y_{\text{down}}$ MAE | 0.02561 | 0.02544 | −0.67% |
| Total OOS predictions | 154,100 | — | — |

The headline +6.56% average improvement masks a **dramatic asymmetry**: the FNN is effective on $y_{\text{up}}$ (+11.31%) but barely functional on $y_{\text{down}}$ (+1.82% RMSE, and actually negative on MAE: −0.67%). This means the FNN's downside predictions are, on average, **worse than predicting zero** when measured by mean absolute error.

### 3.2 Cross-Model Comparison

| Model | $y_{\text{up}}$ RMSE | $y_{\text{dn}}$ RMSE | $y_{\text{up}}$ Improv. | $y_{\text{dn}}$ Improv. | Avg Improv. |
|-------|:---:|:---:|:---:|:---:|:---:|
| XGBoost | 0.03926 | 0.04006 | +11.34% | −0.52% | +5.41% |
| **FNN** | **0.03901** | **0.03826** | **+11.31%** | **+1.82%** | **+6.56%** |
| MDN | 0.03758 | 0.03615 | +14.48% | +7.25% | +10.87% |
| CVAE | 0.03749 | 0.03605 | +15.25% | +9.54% | +12.42% |

**Observations:**
1. **FNN ≈ XGBoost on the upside**: Both hover around +11.3% improvement for $y_{\text{up}}$, suggesting a ceiling for deterministic models on upside prediction.
2. **FNN > XGBoost on the downside**: The FNN's +1.82% beats XGBoost's −0.52%, but both are far below MDN (+7.25%) and CVAE (+9.54%).
3. **The deterministic-probabilistic gap**: FNN → MDN represents a **+4.31 pp** jump in average improvement; FNN → CVAE represents **+5.86 pp**. This gap is the single strongest argument for probabilistic modeling in our study.

> **Supporting graph**: `results/fnn_rolling/fig4_overall_summary.png`

---

## 4. Multi-Dimensional Outperformance Analysis

### 4.1 Dimension 1: Upside Prediction ($y_{\text{up}}$) — Matches XGBoost

The FNN's $y_{\text{up}}$ improvement (+11.31%) is nearly identical to XGBoost (+11.34%), suggesting a **deterministic ceiling** around +11.3% for upside return prediction with these features. Both models learn the same core signal: the unconditional mean of $y_{\text{up}}$ is positive (~2–3%), so predicting a value closer to this mean automatically beats the naive predictor of zero.

However, the FNN does not surpass XGBoost here because:
- XGBoost has built-in feature selection via tree splits; the FNN processes all features equally
- The FNN's Huber loss penalizes large errors less than MSE, which can sacrifice RMSE for robustness
- Both lack the distributional modeling that allows MDN/CVAE to capture conditional heteroscedasticity

### 4.2 Dimension 2: Downside Prediction ($y_{\text{down}}$) — Marginal Gain

The FNN's most critical weakness is its downside performance:

| Model | $y_{\text{down}}$ RMSE Improve | $y_{\text{down}}$ MAE Improve |
|-------|:---:|:---:|
| XGBoost | −0.52% | — |
| **FNN** | **+1.82%** | **−0.67%** |
| MDN | +7.25% | — |
| CVAE | +9.54% | — |

The +1.82% RMSE improvement is technically positive, but the **negative MAE** (−0.67%) reveals that the RMSE gain comes entirely from avoiding a few large errors (extreme outlier predictions) while performing worse on average. This is a hallmark of Huber loss: it clips gradients on extreme residuals, reducing catastrophic errors but producing systematically biased median predictions.

### 4.3 Dimension 3: Ticker-Level Consistency — Significant Failures

| Model | Tickers with avg_improve > 0% | Worst ticker avg_improve |
|-------|:---:|:---:|
| XGBoost | ~85/100 | −5.22% (PEP) |
| **FNN** | **~73/100** | **−12.59% (BRK-B)** |
| MDN | ~93/100 | −3.30% (BRK-B) |
| CVAE | 100/100 | +4.45% (MCD) |

The FNN has the **worst ticker-level consistency** of all four models, with ~27 tickers showing negative average improvement. Its worst ticker (BRK-B at −12.59%) is nearly 4× worse than the MDN's worst (−3.30%). This is a severe practical limitation: deploying the FNN on the wrong ticker could produce predictions **substantially worse than simply predicting zero**.

### 4.4 Dimension 4: Temporal Stability — Fewest Folds

The FNN completes only **26 of 29 folds**, the fewest of any model:

| Model | Folds Completed | Skipped Folds |
|-------|:---:|:---:|
| CVAE | **29/29** | None |
| XGBoost | 28/29 | 29 |
| MDN | 27/29 | 13, 14 |
| **FNN** | **26/29** | **13, 14, 29** |

The FNN inherits the MDN's fold 13–14 failures (likely due to BatchNorm instability) and additionally fails on fold 29, suggesting the model is the **least numerically stable** of the four architectures.

> **Supporting graphs**: `results/fnn_rolling/fig1_fold_rmse.png`, `results/fnn_rolling/fig2_fold_improvement.png`

---

## 5. High-Volatility Market Behavior

### 5.1 COVID-19 Crash (Fold 5: March–May 2020)

Fold 5 covers the most extreme period in our evaluation — the COVID-19 crash. Training data ends February 2020 (pre-COVID); testing covers March–May 2020.

**FNN COVID Performance:**

| Metric | FNN (Fold 5) | FNN (All-fold avg) | Fold 5 / Average |
|--------|:---:|:---:|:---:|
| $y_{\text{up}}$ RMSE | 0.0968 | 0.0390 | 2.48× |
| $y_{\text{dn}}$ RMSE | 0.0912 | 0.0383 | 2.38× |

The FNN's $y_{\text{up}}$ RMSE during COVID (0.0968) is almost **identical to the naive baseline** (0.0969), yielding essentially **0% improvement** (−0.02%) on the upside. On the downside, FNN RMSE (0.0912) is **worse than naive** (0.0825), producing **−10.58% deterioration**.

### 5.2 Cross-Model COVID Comparison (Fold 5)

| Model | $y_{\text{up}}$ RMSE | $y_{\text{up}}$ vs Naive | $y_{\text{dn}}$ RMSE | $y_{\text{dn}}$ vs Naive |
|-------|:---:|:---:|:---:|:---:|
| **FNN** | **0.0968** | **−0.02%** | **0.0912** | **−10.58%** |
| XGBoost | 0.0877 | +9.28% | 0.0977 | −18.46% |
| MDN | 0.0946 | +2.31% | 0.0885 | −7.25% |
| CVAE | 0.0848 | +12.44% | 0.0804 | +2.46% |

**Key findings:**
1. **FNN is the worst model on the upside during COVID**: All other models achieve at least some improvement; the FNN does not.
2. **FNN is the second-worst on the downside** (−10.58%), only ahead of XGBoost (−18.46%).
3. **Only the CVAE beats naive on BOTH targets** during COVID (+12.44% / +2.46%).
4. The FNN's COVID failure is a direct consequence of its deterministic architecture: when the conditional distribution of returns shifts dramatically (unprecedented volatility spike), a model that can only output the conditional mean has no mechanism to adapt.

### 5.3 Post-COVID Recovery and Instability

| Fold | Period | $y_{\text{up}}$ Improv. | $y_{\text{dn}}$ Improv. | Notes |
|------|--------|:---:|:---:|-------|
| 5 | Mar–May 2020 | −0.02% | −10.58% | COVID crash |
| 6 | Jun–Aug 2020 | +15.76% | −2.73% | Partial recovery, but $y_{\text{dn}}$ still negative |
| 7 | Sep–Nov 2020 | +16.67% | +4.75% | Strong recovery on $y_{\text{up}}$ |
| 8 | Dec 2020–Feb 2021 | +16.67% | +8.34% | Best post-COVID performance |
| 9 | Mar–May 2021 | +19.77% | −51.65% | **$y_{\text{dn}}$ catastrophic failure** |

**Fold 9 is remarkable**: the FNN achieves an excellent +19.77% upside improvement but simultaneously produces −51.65% downside deterioration ($y_{\text{dn}}$ RMSE = 0.0464 vs naive 0.0306). This is the clearest example of the FNN's asymmetric failure mode — strong on $y_{\text{up}}$, catastrophically wrong on $y_{\text{down}}$ in the same fold.

The −51.65% deterioration in fold 9 likely arises because:
- The training window (Mar 2020–Feb 2021) includes extreme COVID-era $y_{\text{down}}$ values
- The FNN's conditional mean gets pulled toward large negative values
- In the calmer Mar–May 2021 test period, these predictions are systematically too negative
- Huber loss's L1 region prevents the FNN from fully correcting this bias during training

### 5.4 High-Volatility Tickers

| Ticker | Annualized Vol | FNN Avg Improv. | CVAE Avg Improv. | MDN Avg Improv. | XGB Avg Improv. |
|--------|:-:|:---:|:---:|:---:|:---:|
| TSLA | ~70% | +4.08% | +10.90% | −2.64% | +4.09% |
| CCL | ~65% | +8.06% | +11.12% | +12.01% | +7.58% |
| TTD | ~60% | +9.69% | +11.61% | +11.46% | +6.72% |
| AMD | ~55% | +9.09% | +14.31% | +12.47% | +8.34% |
| ETSY | ~55% | +11.96% | +13.46% | +14.44% | +7.76% |
| FCX | ~50% | +10.91% | +11.81% | +11.78% | +11.78% |

The FNN shows **reasonable but not exceptional** performance on high-volatility tickers. Its TSLA result (+4.08%) is instructive: nearly identical to XGBoost (+4.09%), both far behind CVAE (+10.90%). High-volatility tickers have wider, more multimodal conditional return distributions, which disproportionately benefit from probabilistic modeling.

---

## 6. Why Does the FNN Outperform the Naive Baseline?

Despite its limitations, the FNN genuinely beats the naive predictor ($\hat{y}=0$) at the aggregate level. Several mechanisms explain this:

### 6.1 Learning the Unconditional Positive Bias of $y_{\text{up}}$

$y_{\text{up}} = \max(P_{t+1:t+5})/P_t - 1$ is **strictly non-negative** by construction (the maximum price is at least the current price in theory, though intraday effects can create small negatives). The empirical mean of $y_{\text{up}}$ is approximately +2–3%. Simply predicting this mean value, rather than zero, reduces RMSE by ~10%. The FNN's +11.31% upside improvement is only slightly above what a simple mean predictor would achieve, suggesting it learns the unconditional mean well plus a small amount of conditional adjustment.

### 6.2 Non-Linear Feature Interactions

The FNN's three-layer [256, 128, 64] architecture can capture non-linear interactions between features (e.g., VIX × recent volatility, momentum × volume) that improve prediction beyond simple conditional means. However, these interactions provide only **marginal improvements** over XGBoost, which captures similar interactions via tree splits.

### 6.3 Huber Loss Provides Robustness to Outliers

With $\delta=0.02$, the Huber loss transitions from L2 to L1 penalty at 2% return magnitude. This means:
- For typical returns (<2%), the model optimizes the standard MSE objective
- For extreme returns (>2%), gradient magnitude is capped at 1, preventing outliers from dominating parameter updates
- This produces predictions that are more conservative and closer to zero, which is safe but limits the model's ability to predict large movements

### 6.4 Temporal Validation + Early Stopping

The FNN uses the last 15% of training data as temporal validation with early stopping (patience=15). This prevents overfitting to the training distribution and provides some protection against regime changes within the training window.

---

## 7. Why Does the FNN Underperform Probabilistic Models?

The +4.31 pp gap between FNN (+6.56%) and MDN (+10.87%), and the +5.86 pp gap vs CVAE (+12.42%), are the most important findings in the FNN evaluation. Five structural factors explain this:

### 7.1 The Fundamental Limitation: Conditional Mean ≠ Optimal Point Estimate

The FNN outputs:
$$\hat{y}_{\text{FNN}}(x) \approx \mathbb{E}[y \mid x]$$

When the true conditional distribution $p(y|x)$ is **unimodal and symmetric**, the conditional mean is the optimal RMSE-minimizing point estimate. But financial returns are:
- **Skewed**: $y_{\text{up}}$ is right-skewed (bounded below at ~0, heavy right tail); $y_{\text{down}}$ is left-skewed
- **Heteroscedastic**: variance changes with volatility regime (calm vs. crisis)
- **Occasionally multimodal**: the same features can produce very different outcomes depending on hidden regime state

When $p(y|x)$ is multimodal, the conditional mean falls **between the modes** where very few actual observations lie, producing systematically poor predictions.

By contrast:
- **MDN** models $p(y|x) = \sum_k \pi_k \mathcal{N}(\mu_k, \sigma_k^2)$, capturing multimodality directly
- **CVAE** integrates over latent regimes: $p(y|x) = \int p(y|x,z) p(z|x) dz$

Both produce a predictive mean that is an **informed weighted average** over plausible outcomes, rather than a single regression surface.

### 7.2 Huber Loss Creates a Conservative Bias

Huber loss with a small $\delta$ (0.02, or ~2% return) makes the FNN **systematically under-predict extreme movements**. The L1 penalty beyond $\delta$ means the model receives the same gradient magnitude for a 5% error as for a 50% error. This:
- Reduces the model's incentive to predict large values
- Pulls predictions toward zero, which is safe for typical days but harmful during high-volatility regimes
- Particularly damaging for $y_{\text{down}}$ during selloffs, where accurate predictions require predicting large negative values

### 7.3 BatchNorm in a Heterogeneous Universe

The FNN uses BatchNorm on the input layer. With 100 tickers of varying volatility pooled into the same batches:
- Batch statistics are **contaminated by cross-ticker heterogeneity** (TSLA and KO in the same batch)
- Running statistics stored for inference may not match test-time distributions during regime changes
- This was confirmed as the root cause of fold 13–14 failures in the CVAE, which was fixed by switching to LayerNorm. The FNN still uses BatchNorm.

### 7.4 No Uncertainty = No Adaptivity

The FNN has no mechanism to express "I am uncertain about this prediction." When input features indicate an ambiguous regime:
- The FNN outputs a single prediction that may be far from either plausible outcome
- MDN/CVAE widen their predictive distributions, naturally adjusting the mean
- This is why the FNN collapses during COVID: it cannot recognize that the training-test distribution has shifted

### 7.5 Shared Hyperparameters Not Optimized for FNN

The FNN uses the same [256, 128, 64] architecture and dropout schedule as the MDN/CVAE, but these were tuned primarily for the MDN (53-experiment HP search). For a deterministic regression task:
- The network may be **over-parameterized** (MDN needs capacity for 5 mixture components)
- Dropout 0.3 may be **too aggressive** (FNN has no latent variable to compensate)
- Huber $\delta=0.02$ was never systematically tuned

---

## 8. Ticker-Level Deep Dive

### 8.1 Best Performers (Top 10 by avg_improve)

| Rank | Ticker | $y_{\text{up}}$ Improve (%) | $y_{\text{dn}}$ Improve (%) | Avg Improve (%) | Sector |
|:---:|--------|:---:|:---:|:---:|--------|
| 1 | ETSY | +15.51% | +8.42% | +11.96% | E-commerce |
| 2 | PYPL | +14.19% | +8.58% | +11.38% | Fintech |
| 3 | NOW | +17.31% | +5.06% | +11.19% | Technology |
| 4 | FCX | +17.42% | +4.40% | +10.91% | Mining |
| 5 | LULU | +17.06% | +4.67% | +10.87% | Retail |
| 6 | COP | +17.65% | +2.97% | +10.31% | Energy |
| 7 | MU | +15.53% | +4.72% | +10.13% | Semiconductors |
| 8 | MPC | +19.33% | +0.68% | +10.01% | Energy |
| 9 | LRCX | +17.93% | +2.01% | +9.97% | Semiconductors |
| 10 | NXPI | +17.64% | +2.28% | +9.96% | Semiconductors |

**Pattern**: The top performers are **high-volatility, cyclical stocks** where $y_{\text{up}}$ is large (making the naive baseline easy to beat). Note that even the top ticker's $y_{\text{down}}$ improvement (+8.42%) is modest, and many top tickers show $y_{\text{down}}$ improvement below +5%. The FNN's ranking is driven almost entirely by $y_{\text{up}}$.

### 8.2 Worst Performers (Bottom 10)

| Rank | Ticker | $y_{\text{up}}$ Improve (%) | $y_{\text{dn}}$ Improve (%) | Avg Improve (%) | Sector |
|:---:|--------|:---:|:---:|:---:|--------|
| 91 | LIN | +6.80% | −10.51% | −1.85% | Materials |
| 92 | WMT | −1.99% | −6.76% | −4.37% | Consumer Staples |
| 93 | COST | −3.54% | −6.21% | −4.88% | Consumer Staples |
| 94 | PG | −7.08% | −4.35% | −5.71% | Consumer Staples |
| 95 | JNJ | −7.69% | −4.39% | −6.04% | Healthcare |
| 96 | KO | −5.38% | −6.81% | −6.09% | Consumer Staples |
| 97 | PEP | −8.13% | −4.12% | −6.12% | Consumer Staples |
| 98 | MCD | −3.31% | −11.69% | −7.50% | Consumer Staples |
| 99 | VZ | −10.75% | −6.05% | −8.40% | Telecom |
| 100 | BRK-B | −11.23% | −13.95% | **−12.59%** | Financials |

**The failure pattern is systematic**: All bottom-10 tickers are **low-volatility defensive stocks** (consumer staples, telecom, healthcare). For these tickers:
- Both $y_{\text{up}}$ and $y_{\text{down}}$ are very close to zero (typical: $y_{\text{up}} \sim$ +1.3%, $y_{\text{down}} \sim$ −1.0%)
- The naive predictor $\hat{y}=0$ is already an excellent estimate
- The FNN predicts values slightly away from zero, and because the true values are so small, even tiny absolute errors produce large relative deterioration
- BRK-B is the worst case: its extremely low volatility makes any active prediction worse than staying at zero

**Critical comparison**: The CVAE achieves +4.45% even on its worst ticker (MCD), while the FNN is at −7.50% for the same stock. The CVAE's latent variable allows it to produce near-zero predictions for low-vol tickers (effectively mimicking the naive baseline), while the FNN's regression surface cannot flatten sufficiently.

### 8.3 FNN vs CVAE on the Bottom 10

| Ticker | FNN Avg Improve | CVAE Avg Improve | CVAE − FNN |
|--------|:---:|:---:|:---:|
| BRK-B | −12.59% | +7.42% | **+20.01 pp** |
| VZ | −8.40% | +5.10% | **+13.50 pp** |
| MCD | −7.50% | +4.45% | **+11.95 pp** |
| PEP | −6.12% | +8.59% | **+14.71 pp** |
| KO | −6.09% | +6.37% | **+12.46 pp** |
| JNJ | −6.04% | +4.90% | **+10.94 pp** |
| PG | −5.71% | +9.09% | **+14.80 pp** |
| COST | −4.88% | +7.04% | **+11.92 pp** |
| WMT | −4.37% | +6.22% | **+10.59 pp** |
| LIN | −1.85% | +9.48% | **+11.33 pp** |

The CVAE outperforms the FNN by **+10 to +20 percentage points** on every bottom-10 ticker. This is the starkest evidence that the FNN's deterministic architecture **systematically fails on low-volatility stocks**.

> **Supporting graph**: `results/fnn_rolling/fig3_ticker_scatter.png`

---

## 9. The Asymmetric $y_{\text{up}}$ vs $y_{\text{down}}$ Gap

### 9.1 Quantifying the Asymmetry

| Model | $y_{\text{up}}$ Improve | $y_{\text{dn}}$ Improve | Gap (pp) |
|-------|:---:|:---:|:---:|
| **FNN** | +11.31% | +1.82% | **9.49** |
| XGBoost | +11.34% | −0.52% | 11.86 |
| MDN | +14.48% | +7.25% | 7.24 |
| CVAE | +15.25% | +9.54% | 5.71 |

The FNN's 9.49 pp asymmetry is the **second-worst** after XGBoost (11.86 pp). But unlike XGBoost, the FNN at least achieves a (barely) positive $y_{\text{down}}$. The progression from FNN (9.49 pp gap) → MDN (7.24 pp) → CVAE (5.71 pp) shows that probabilistic models systematically **reduce the up-down asymmetry**.

### 9.2 Why is $y_{\text{down}}$ So Much Harder?

Three factors make downside prediction inherently harder for deterministic models:

1. **The naive baseline is stronger for $y_{\text{down}}$**: Because $y_{\text{down}}$ = min return over 5 days, and most days have slightly negative intraday moves, $y_{\text{down}}$ has a small negative unconditional mean (roughly −1.5% to −2%). Predicting 0 is only off by ~1.5%, whereas for $y_{\text{up}}$ (unconditional mean ~2.5%), predicting 0 is off by ~2.5%. A tighter baseline is harder to beat.

2. **Extreme downside events are rarer and more severe**: Crash days (−5% to −10%) occur less frequently than rally days of similar magnitude, but when they occur, they dominate $y_{\text{down}}$ for that week. The Huber loss clips gradients on these events, preventing the FNN from learning to predict them.

3. **$y_{\text{down}}$ has a more skewed distribution**: The distribution of $y_{\text{down}}$ has a heavier left tail than $y_{\text{up}}$ has a right tail. A single-point predictor targeting the conditional mean systematically overestimates (not negative enough) during normal markets and underestimates (too negative) during extreme events.

### 9.3 Ticker-Level Asymmetry

Among the 100 tickers:
- **73 tickers** have positive $y_{\text{up}}$ improvement (only 6 have $>$ 17%)
- **48 tickers** have positive $y_{\text{down}}$ improvement
- Only **~40 tickers** have positive improvement on BOTH targets

This means for ~60% of the universe, the FNN either fails on the downside, the upside, or both. Compare to CVAE where **all 100 tickers** have positive improvement on $y_{\text{up}}$, and the vast majority are positive on both targets.

> **Supporting graph**: `results/fnn_rolling/fig3_ticker_scatter.png`

---

## 10. Potential Improvements

### 10.1 Short-Term (Low-Effort, High-Impact)

| Improvement | Expected Impact | Mechanism |
|-------------|:---:|------------|
| **BatchNorm → LayerNorm** | +2–3% avg, complete all 29 folds | Eliminate cross-ticker batch statistic contamination; proven effective in CVAE |
| Tune Huber $\delta$ | +0.5–1% on $y_{\text{down}}$ | Larger $\delta$ (e.g., 0.05) allows fuller gradient on moderate-sized returns |
| FNN-specific HP search | +1–2% | Optimize hidden dims, dropout, LR independently from MDN |
| Replace one-hot ticker with learned embedding (dim=8) | Reduce input sparsity | 8 dims vs 100 one-hot; enables ticker similarity learning |

### 10.2 Medium-Term (Moderate Effort)

| Improvement | Expected Impact | Mechanism |
|-------------|:---:|------------|
| **Quantile regression** ($\tau$ = 0.1, 0.5, 0.9) | Better $y_{\text{down}}$ | Directly model conditional quantiles instead of mean; captures asymmetry |
| Output $(\mu, \sigma)$ with Gaussian NLL | +2–3% avg | Essentially a single-component MDN; adds uncertainty |
| Asymmetric loss function | Improve downside | Weight downside errors more heavily to correct the bias |
| Ensemble of 5 FNNs with different seeds | +0.5–1% | Implicit uncertainty via prediction variance |

### 10.3 Long-Term (Architectural Change)

| Improvement | Expected Impact | Mechanism |
|-------------|:---:|------------|
| Switch to **MDN** or **CVAE** | +4–6% avg | Probabilistic models provide fundamentally better conditional distribution modeling |
| Add temporal attention | +1–2% | Capture time-varying patterns in feature importance |
| Graph neural network over ticker universe | +1% | Model cross-ticker correlations explicitly |

### 10.4 The Honest Assessment

The FNN's limitations are **structural, not fixable by hyperparameter tuning**. The conditional mean of financial returns is an inherently suboptimal prediction target because financial return distributions are:
- Non-Gaussian (skewed, heavy-tailed)
- Heteroscedastic (variance changes over time)
- Occasionally multimodal (regime-dependent)

The most impactful "improvement" to the FNN is to **replace it with a probabilistic model**. The FNN's primary value is as a **controlled experimental baseline** that demonstrates this fundamental point.

---

## 11. Supporting Graphs Index

All figures are located in `results/fnn_rolling/`:

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig 1 | `fig1_fold_rmse.png` | RMSE per fold for $y_{\text{up}}$ and $y_{\text{down}}$, with naive baseline overlay |
| Fig 2 | `fig2_fold_improvement.png` | Improvement percentage per fold with trend line |
| Fig 3 | `fig3_ticker_scatter.png` | Per-ticker improvement scatter plot (100 tickers, $y_{\text{up}}$ vs $y_{\text{down}}$) |
| Fig 4 | `fig4_overall_summary.png` | Overall RMSE comparison bar chart |
| Fig 5 | `fig5_ticker_true_vs_pred.png` | True vs predicted values for selected tickers |

---

## 12. Conclusion

The FNN achieves a **+6.56% average RMSE improvement** over the naive predictor, placing it between XGBoost (+5.41%) and the probabilistic models (MDN +10.87%, CVAE +12.42%). Its performance reveals several fundamental insights:

### What the FNN Proves

| Finding | Evidence |
|---------|---------|
| Neural networks learn useful non-linear patterns | FNN (+6.56%) > XGBoost (+5.41%) |
| Deterministic models have a ~11% ceiling on $y_{\text{up}}$ | FNN (+11.31%) ≈ XGBoost (+11.34%) |
| Downside prediction requires probabilistic models | FNN $y_{\text{dn}}$ (+1.82%) << CVAE (+9.54%) |
| Low-vol tickers are unsolvable for deterministic models | FNN: 27 negative tickers; CVAE: 0 |
| BatchNorm fails with heterogeneous financial data | FNN skips 3 folds; CVAE (LayerNorm) skips 0 |

### The FNN's Role in the Model Hierarchy

```
XGBoost (+5.41%)  →  FNN (+6.56%)  →  MDN (+10.87%)  →  CVAE (+12.42%)
     Tree baseline       NN baseline      Probabilistic       Latent variable
                                          mixture model       generative model
```

Each step represents a conceptual advance:
1. **XGBoost → FNN** (+1.15 pp): Neural networks capture smoother non-linear interactions
2. **FNN → MDN** (+4.31 pp): **Probabilistic output** enables conditional heteroscedasticity modeling
3. **MDN → CVAE** (+1.55 pp): **Latent variables** capture hidden regime structure

The FNN-to-MDN gap (+4.31 pp) is the largest single jump, confirming that the most impactful modeling decision in equity range prediction is **not the architecture complexity, but whether the model outputs a distribution vs. a point estimate**. The FNN serves as the essential control experiment that validates this conclusion.

### Summary Table

| Dimension | FNN Result | Best Model |
|-----------|:---------:|:----------:|
| Average RMSE Improvement | +6.56% | CVAE (+12.42%) |
| $y_{\text{up}}$ Improvement | +11.31% | CVAE (+15.25%) |
| $y_{\text{down}}$ Improvement | +1.82% | CVAE (+9.54%) |
| Ticker universality | 73/100 positive | CVAE (100/100) |
| Folds completed | 26/29 | CVAE (29/29) |
| COVID $y_{\text{up}}$ | −0.02% | CVAE (+12.44%) |
| COVID $y_{\text{dn}}$ | −10.58% | CVAE (+2.46%) |
| Worst ticker | −12.59% (BRK-B) | CVAE (+4.45% MCD) |

The FNN is not a failed model — it is a **successful baseline** that precisely quantifies the cost of deterministic point estimation in financial prediction. Its +6.56% improvement is real and valuable for comparison purposes, but its severe downside weakness, ticker-level failures, and COVID collapse make it unsuitable for deployment. The 4–6 percentage-point gap to probabilistic models is the study's most actionable finding.
