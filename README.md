# VaR Risk Analytics Library

A quantitative portfolio risk management library implementing five VaR estimation methods, coherent Expected Shortfall, multi-framework backtesting, GMM market regime detection, component VaR attribution, and a six-scenario stress testing engine — with a Streamlit visualisation layer on top.

Built by a Senior Software Engineer with hands-on derivatives lifecycle and risk controls experience, and FRM Level 1 (GARP).

---

## Repository Structure

```
VaR-Dashboard/
├── var_streamlit_dashboard.py   # Streamlit UI — imports from src/
├── requirements.txt
├── src/
│   ├── returns.py               # Log returns, EWMA covariance, rolling cov
│   ├── var_models.py            # Historical, Parametric, FHS, Monte Carlo, Cornish-Fisher
│   ├── expected_shortfall.py    # Historical ES, Parametric ES, Stressed ES
│   ├── backtesting.py           # Kupiec POF + Christoffersen CC + Basel traffic light
│   ├── regime_detection.py      # GMM regime detection (4 features, 3 regimes)
│   ├── stress_testing.py        # 6-scenario engine + reverse stress test
│   └── risk_decomposition.py    # Component VaR, Marginal VaR, attribution report
├── tests/
│   ├── test_var_models.py       # 48 tests, all passing
│   ├── test_backtesting.py
│   └── test_risk_decomposition.py
├── notebooks/
│   ├── 01_var_methodology.ipynb
│   ├── 02_backtesting_analysis.ipynb
│   ├── 03_regime_analysis.ipynb
│   └── 04_stress_testing.ipynb
└── results/figures/
```

---

## Methodology

### VaR Estimation Methods

| Method | Summary |
|--------|---------|
| **Historical Simulation** | Non-parametric empirical quantile of the return distribution. No distributional assumption. Sensitive to the composition of the lookback window. |
| **Parametric (Normal)** | `VaR = mu - z_alpha * sigma`. Fast and analytical but underestimates tail risk when returns have excess kurtosis — the core failure of pre-2008 risk models. |
| **Parametric (Student-t)** | MLE-fit degrees of freedom. Accounts for fat tails; consistently gives higher VaR than Normal at 99% confidence. |
| **Filtered Historical Simulation** | Hull-White (1998). Standardise returns by EWMA conditional vol, run historical simulation on standardised residuals, scale back by current vol. Basel III internal models standard at most major banks. |
| **Cornish-Fisher Expansion** | First-order Edgeworth correction for skewness S and excess kurtosis K: `z_CF = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`. More extreme than Normal at 99% when K > 0. |
| **Monte Carlo (EWMA vol)** | 100K paths using current EWMA volatility. Returns point estimate and standard error. 10K paths is insufficient for stable 99% VaR estimation (~100 tail observations). |

### Expected Shortfall

`ES = -E[r | r < VaR_alpha]`

ES is a coherent risk measure satisfying subadditivity. VaR is not: two portfolios with identical VaR can have very different tail severity. Basel III / FRTB replaced VaR with ES (at 97.5% confidence) as the primary internal models metric.

### Backtesting Frameworks

**Kupiec (1995) POF Test** — tests whether the observed violation frequency equals the nominal confidence level. `LR_uc ~ chi-squared(1)`.

**Christoffersen (1998) Conditional Coverage** — jointly tests correct frequency AND violation independence. Extends Kupiec with a transition matrix over violation sequences. `LR_cc = LR_uc + LR_ind ~ chi-squared(2)`. Violation clustering (n11 > 0) is the most common failure mode during market stress — VaR models that pass Kupiec can still fail CC if violations bunch in crisis periods.

**Basel III Traffic Light** — 250-day assessment: 0-4 violations = Green (no add-on), 5-9 = Yellow (multiplier 1.13-1.28), 10+ = Red (multiplier 1.85, model failure presumption).

### Risk Decomposition

For portfolio weights **w** and covariance matrix **Sigma**:

- Portfolio VaR: `VaR_p = z_alpha * sigma_p` where `sigma_p = sqrt(w'*Sigma*w)`
- Marginal VaR: `MVaR_i = z_alpha * (Sigma*w)_i / sigma_p`
- Component VaR: `CVaR_i = w_i * MVaR_i`
- **Key identity: sum(CVaR_i) = VaR_p** (Euler decomposition, exact)

Sign convention: VaR is a negative return (loss). Component VaR < 0 means the position adds to portfolio loss; Component VaR > 0 means it reduces total risk (hedge/diversifier).

### Stress Testing Scenarios

| Scenario | Equity Shock | Vol Multiplier | Description |
|----------|-------------|----------------|-------------|
| 2008 Financial Crisis | -42% | 2.5x | Lehman collapse, global credit freeze |
| COVID-19 Crash | -34% | 3.0x | Fastest 30% S&P 500 decline in history |
| Dot-com Bust | -49% | 1.2x | 2.5-year tech bear market |
| 1987 Black Monday | -22.6% | 4.0x | Single-day DJIA crash |
| Taper Tantrum 2013 | -6% | 0.5x | Fed QE tapering signal |
| 2022 Rate Shock | -25% | 0.8x | Fastest Fed hiking cycle since 1980s |

Reverse stress testing: given a target loss (e.g., 20% of portfolio), solve for the equity shock that causes it. Increasingly required under ICAAP post-2008.

---

## Benchmark Results — SPY / QQQ / IWM (Equal Weight, 2018-2024)

*All values are daily losses expressed as percentage of portfolio value (positive = loss).*

### VaR and ES Estimates

| Method | 95% VaR | 99% VaR | 95% ES |
|--------|---------|---------|--------|
| Historical Simulation | 2.28% | 3.79% | 3.41% |
| Parametric (Normal) | 2.27% | 3.24% | — |
| FHS (EWMA) | 1.57% | 2.81% | — |
| Monte Carlo (EWMA) | 1.44% | 2.06% | — |
| Cornish-Fisher | 2.26% | **6.55%** | — |

*Cornish-Fisher 99% VaR is notably higher than Historical due to the large negative skewness and excess kurtosis in the 2018-2024 sample (includes COVID and 2022 rate shock). The `(z^3-3z)` correction term is negative at 99% confidence, amplifying the kurtosis effect.*

### Annual Backtesting — 95% VaR (prior-year calibration, fixed threshold)

| Year | Violations | Expected | Kupiec p-val | Christoffersen p-val | n11 | Zone |
|------|-----------|---------|--------------|----------------------|-----|------|
| 2019 | 6 | 12.6 | 0.034 | 0.092 | 0 | YELLOW |
| **2020** | **40** | 12.7 | **<0.001** | **<0.001** | **9** | **RED** |
| 2021 | 0 | 12.6 | 1.000 | 1.000 | 0 | GREEN |
| **2022** | **37** | 12.6 | **<0.001** | **<0.001** | **5** | **RED** |
| 2023 | 0 | 12.5 | 1.000 | 1.000 | 0 | GREEN |

**Key finding:** 2020 and 2022 both fail Kupiec (wrong frequency) and Christoffersen (clustered violations: n11=9 and n11=5). A VaR calibrated on a calm prior year is dramatically wrong when the regime shifts. This is the practical case for EWMA-based FHS — which adapts the VaR threshold within days of a vol spike, rather than waiting for the lookback window to roll.

---

## GMM Regime Detection

Three-regime GMM on four features (daily returns, 21-day rolling vol, 10-day momentum, 30-day mean-reversion z-score). Per-regime VaR in the stressed regime is typically 2-4x the full-sample estimate, demonstrating why single-regime VaR systematically understates tail risk during crises.

---

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run var_streamlit_dashboard.py
```

## Running Tests

```bash
pytest tests/ -v
# 48 tests — all passing
```

---

## Key Design Notes

- **FHS is the Basel III internal models standard.** It decouples the empirical residual distribution from current volatility, addressing the stale-volatility failure of plain historical simulation. Implemented with RiskMetrics EWMA (lambda=0.94).
- **Christoffersen is required alongside Kupiec.** Kupiec alone misses the clustering failure mode — the most dangerous failure pattern in 2008 and 2020. Both tests are run and reported in every backtest.
- **Component VaR for attribution** uses the Euler decomposition identity. This is the standard risk budget report format at sell-side risk desks.
- **100K MC paths minimum for 99% VaR.** 10K paths leaves only ~100 observations in the 1% tail, giving unstable quantile estimates.
