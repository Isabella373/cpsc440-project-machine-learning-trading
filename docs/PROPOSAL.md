# Project Proposal: Equity Return Prediction

## 1. Title
Deterministic vs Probabilistic Models for Cross-Sectional Equity Return Prediction

## 2. Problem Statement

Compare two model classes for future equity return prediction:

1. **Deterministic Feedforward Neural Network** - trained to predict a point estimate of future returns
2. **Probabilistic Mixture Density Network** - trained to model the full conditional distribution of returns as a mixture of Gaussians

Baseline: **XGBoost / LightGBM** (strongest tabular models)

## 3. Research Gap & Contribution

### Key Questions
1. **Do probabilistic models improve predictive performance** compared with deterministic NNs and strong baselines?
2. **Where do improvements arise** (point prediction, uncertainty calibration, ranking quality)?
3. **When is probabilistic modeling useful** (e.g., high volatility regimes)?
4. **Is the added complexity justified** in noisy financial prediction tasks?

### Contribution
Empirical evidence on whether explicitly modeling the conditional distribution of equity returns provides practical advantages over point prediction in real-world financial settings characterized by:
- Non-stationarity
- Heavy-tailed returns
- Heteroskedasticity
- Noisy signals

## 4. Data

### Features (Input)

#### Price-Volume Features
- Returns: `ret_1d`, `ret_5d`, `ret_20d`
- Volatility: `vol_20d`, `skew_20d`, `kurt_20d`
- Momentum: `momentum_20d`, `momentum_60d`, `rsi_14`, `macd`
- Moving Averages: `ma_5`, `ma_20`, `ma_ratio_5`, `ma_ratio_20`
- Volume: `volume_zscore`, `dollar_volume`, `hl_spread`, `oc_return`

#### Regime Features
- **VIX** (Cboe): daily level, 9-day change, VVIX, moving average/slope
- **Bond Yields** (FRED): 1M and 1Y Treasury yields
- **Employment** (BLS): NFP, unemployment rate, initial jobless claims
- **USD Index (DXY)**: risk-off indicator
- **S&P500 Forward P/E**: regime feature
- **ETF Flows**: SPY/VOO/IVV capital flows
- **Options**: Put/Call ratio, IV skew
- **Cross-Asset**: GLD (gold), WTI (oil), TLT/IEF (bonds)
- **Calendar**: holidays, Fridays, month-end, quarter-end, dividend dates

#### Stock Universe
- S&P 500 constituents (top 100 by 5-year trading volume)
- **Ticker embedding**: stock ID as feature to capture cross-stock patterns
- Daily OHLCV data: yfinance (Jan 2020 - Mar 2026)

### Target (Output)
**Future 5-day log return**: 
$$y_{i,t} = \log(P_{i,t+5}) - \log(P_{i,t})$$

Computed for each stock $i$ on each date $t$.

## 5. Model Architectures

### 1. Feedforward Neural Network
- Point prediction: $\hat{y} = f(x)$
- Standard regression loss (MSE)
- Baseline deterministic approach

### 2. Mixture Density Network (MDN)
- Output: mixture parameters $\pi_k(x), \mu_k(x), \sigma_k(x)$ for $k=1,\ldots,K$
- Models conditional distribution: $p(y|x) = \sum_{k=1}^K \pi_k(x) \mathcal{N}(y; \mu_k(x), \sigma_k^2(x))$
- Expected return: $E[y|x] = \sum_k \pi_k(x) \mu_k(x)$
- Training: Negative Log-Likelihood (NLL) loss
- Advantages: captures multimodality, provides uncertainty estimates

### 3. XGBoost / LightGBM
- Strong tabular baseline
- Point prediction of future returns
- Fast training, competitive performance

## 6. Methodology

### 6.1 Data Processing
- Download historical OHLCV from yfinance
- Compute technical features (returns, volatility, momentum, MAs)
- Fetch macro/regime data (FRED, BLS, Cboe)
- Handle missing data and outliers
- Standardize/normalize features

### 6.2 Time-Series Validation (No Leakage!)
**Rolling Window Strategy** (quarterly splits):
- Train: Jan-Mar (Q1) → Val: Mar-Apr → Test: Apr-May
- Train: Apr-Jun (Q2) → Val: Jun-Jul → Test: Jul-Aug
- Repeat for 2020-2026
- Each fold uses only past data (no future peeking)

### 6.3 Model Training
- **FNN & MDN**: PyTorch, batch training with early stopping
- **XGBoost/LightGBM**: scikit-learn interfaces
- All models receive same features and validation protocol
- Hyperparameter tuning on validation set

## 7. Evaluation Framework

### Point Prediction Quality
- **MSE, MAE**: prediction error
- **R²**: explained variance
- **Directional Accuracy**: % correct sign prediction
- **Spread Analysis**: top-minus-bottom quintile return spread

### Probabilistic Quality (MDN focus)
- **Negative Log-Likelihood (NLL)**: distributional fit quality
- **Calibration**: empirical vs predicted quantiles
- **Coverage**: % of actual values within $\alpha$-level prediction intervals
- **P(|error| > 10%)**: probability of large errors

### Computational Efficiency
- Training time, inference speed, model size

### Regime-Based Analysis
- Split results by:
  - **High VIX** (volatile) vs low VIX periods
  - **Stock-level volatility** quintiles
  - **Return uncertainty** (fitted $\sigma$ levels)
- Question: Does MDN show greater advantage during uncertainty periods?

## 8. Expected Insights

### Deliverables
1. Model implementations with proper rolling time-series validation
2. Comparative results across FNN, MDN, XGBoost, LightGBM
3. Calibration analysis showing uncertainty quality
4. Multi-dimensional evaluation (accuracy, calibration, efficiency, ranking)
5. Regime-based breakdown (when does each model excel?)
6. **Actionable insights**:
   - Is probabilistic structure valuable here? Why or why not?
   - When/where does MDN outperform simpler approaches?
   - What explains the differences (better point forecast vs better uncertainty)?

### Key Questions to Answer
- Does modeling the full distribution help, or is point prediction sufficient?
- Is added complexity justified given financial data's heavy tails and non-stationarity?
- Are there specific market conditions (high vol, stressed periods) where probabilistic methods shine?
- How much of MDN's value comes from reduced point error vs improved uncertainty?

## 9. Scope & Feasibility

### Limitations
- Uses daily price-volume + macro features only (no news/fundamentals)
- Assumes Gaussian mixtures (may underestimate tail risk)
- Results specific to chosen stock universe and time period
- Non-stationarity may limit generalization beyond 2026

### Feasibility
- **Data**: Freely available (yfinance, FRED, BLS, Cboe)
- **Models**: Standard, well-understood architectures
- **Computing**: CPU sufficient; GPU optional
- **Timeline**: Realistic for one semester project

### Risk Mitigation
- Start with subset of data/stocks if needed
- Fallback: compare only deterministic models if probabilistic training is problematic
- Focus on single time period if rolling window is complex

## 10. Timeline
- **Week 1**: Data download & feature engineering
- **Week 2**: Model implementation
- **Week 3**: Training & validation
- **Week 4**: Evaluation & regime analysis
- **Week 5**: Report writing & insights

---

**Status**: Ready for implementation  
**Last Updated**: March 10, 2026
