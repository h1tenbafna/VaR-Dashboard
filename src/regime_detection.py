"""
GMM-based market regime detection.

Extracted from the original StreamlitVaRModel.detect_market_regimes.
The 4-feature GMM approach (returns, vol, momentum, mean-reversion)
is preserved exactly — it is correct and non-trivial.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def detect_market_regimes(
    returns: pd.DataFrame,
    n_regimes: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    GMM-based market regime detection using 4 rolling features.

    Features:
      - returns:        daily portfolio return (level signal)
      - volatility:     21-day rolling std (magnitude of risk)
      - momentum:       10-day rolling mean return (trend signal)
      - mean_reversion: z-score vs 30-day rolling mean (reversion signal)

    These four features together capture the most diagnostically useful
    dimensions of market state: is the market trending, mean-reverting,
    calm, or stressed? A 3-regime GMM typically identifies a low-vol
    bull regime, a high-vol stress regime, and an intermediate recovery
    or sideways regime — which maps closely to practitioner intuition
    about market cycle phases.

    Returns:
        regime_df        — DataFrame with 'regime' label and per-regime probabilities
        portfolio_returns — equal-weighted portfolio return series (Series)
    """
    if isinstance(returns, pd.DataFrame):
        portfolio_returns = returns.mean(axis=1)
    else:
        portfolio_returns = returns.copy()

    if portfolio_returns.empty:
        raise ValueError("Portfolio returns are empty after processing")

    feature_data = pd.DataFrame(index=portfolio_returns.index)
    feature_data["returns"] = portfolio_returns
    feature_data["volatility"] = portfolio_returns.rolling(21).std()
    feature_data["momentum"] = portfolio_returns.rolling(10).mean()
    feature_data["mean_reversion"] = (
        portfolio_returns - portfolio_returns.rolling(30).mean()
    ) / portfolio_returns.rolling(30).std()

    feature_data = feature_data.dropna()

    if feature_data.empty:
        raise ValueError("Feature data empty after rolling calculations (too few observations?)")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_data)

    gmm = GaussianMixture(
        n_components=n_regimes,
        random_state=random_state,
        covariance_type="full",
    )
    gmm.fit(features_scaled)

    regime_labels = gmm.predict(features_scaled)
    regime_probabilities = gmm.predict_proba(features_scaled)

    regime_df = pd.DataFrame(index=feature_data.index)
    regime_df["regime"] = regime_labels
    for i in range(n_regimes):
        regime_df[f"prob_regime_{i}"] = regime_probabilities[:, i]

    return regime_df, portfolio_returns.loc[feature_data.index]


def regime_var_summary(
    returns: pd.Series,
    regime_df: pd.DataFrame,
    n_regimes: int,
    confidence_level: float = 0.05,
) -> pd.DataFrame:
    """
    Per-regime VaR summary: mean, volatility, and historical VaR for each regime.

    Demonstrates the key insight: a single-regime VaR estimate ignores the
    bimodal (or trimodal) nature of the return distribution. The VaR in the
    stressed regime can easily be 3-5x the VaR in the calm regime, yet a
    full-sample VaR averages across both and will consistently understate risk
    when the market enters the stressed regime.
    """
    aligned = returns.loc[regime_df.index]
    rows = []
    for r in range(n_regimes):
        mask = regime_df["regime"] == r
        regime_rets = aligned[mask]
        if len(regime_rets) < 10:
            continue
        rows.append(
            {
                "Regime": f"Regime {r}",
                "Observations": len(regime_rets),
                "Ann Return": regime_rets.mean() * 252,
                "Ann Volatility": regime_rets.std() * np.sqrt(252),
                "VaR (historical)": float(
                    np.percentile(regime_rets, confidence_level * 100)
                ),
                "Freq (%)": 100 * len(regime_rets) / len(aligned),
            }
        )
    return pd.DataFrame(rows)
