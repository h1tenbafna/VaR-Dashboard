"""
Return computation and data utilities.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns: r_t = ln(P_t / P_{t-1})

    Log returns are additive across time: r(0,T) = sum(r(t-1,t)), which makes
    multi-period aggregation trivial. Simple returns are additive across assets
    (portfolio aggregation), but multiplicative across time. For single-period
    daily VaR the difference is negligible, but for horizon scaling and
    longer-horizon risk calculations log returns are more appropriate.
    Use simple returns when aggregating across assets in a portfolio;
    use log returns for time-series analysis and horizon scaling.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_ewma_covariance(returns: pd.DataFrame, lambda_: float = 0.94) -> np.ndarray:
    """
    RiskMetrics EWMA covariance matrix.

    Sigma_t^2 = lambda * Sigma_{t-1}^2 + (1 - lambda) * r_{t-1} * r_{t-1}'

    lambda = 0.94 is the RiskMetrics standard for daily data.
    lambda = 0.97 is standard for monthly data.

    EWMA puts exponentially more weight on recent observations than a simple
    historical covariance — critical for capturing volatility clustering in
    crisis periods. When volatility spikes (as in March 2020), the EWMA
    covariance responds within days, while a 252-day rolling window takes
    weeks to reflect the new regime.

    Returns the most recent EWMA covariance matrix (shape: n_assets x n_assets).
    """
    n = len(returns)
    n_assets = returns.shape[1]
    r = returns.values

    # Initialise with sample covariance of first 30 observations
    init_window = min(30, n)
    sigma = np.cov(r[:init_window].T, ddof=1)
    if n_assets == 1:
        sigma = sigma.reshape(1, 1)

    for t in range(init_window, n):
        r_t = r[t - 1].reshape(-1, 1)
        sigma = lambda_ * sigma + (1 - lambda_) * (r_t @ r_t.T)

    return sigma


def compute_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    RiskMetrics EWMA volatility (scalar series).

    Returns a Series of conditional volatility estimates aligned with the
    input returns index. Used internally by FHS VaR and Monte Carlo.
    """
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.iloc[0] ** 2

    for t in range(1, len(returns)):
        variance.iloc[t] = lambda_ * variance.iloc[t - 1] + (1 - lambda_) * returns.iloc[t - 1] ** 2

    return np.sqrt(variance)


def compute_rolling_covariance(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Rolling covariance matrix over a specified window.

    Returns a DataFrame where each row contains the flattened covariance
    matrix for that date. Useful for analysing how correlations evolve
    across market regimes — a key driver of portfolio VaR.
    """
    n_assets = returns.shape[1]
    cols = [f"cov_{i}_{j}" for i in range(n_assets) for j in range(n_assets)]
    result = pd.DataFrame(index=returns.index, columns=cols, dtype=float)

    for end in range(window, len(returns) + 1):
        window_data = returns.iloc[end - window:end]
        cov = window_data.cov().values.flatten()
        result.iloc[end - 1] = cov

    return result.dropna()
