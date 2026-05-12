"""
Expected Shortfall (CVaR) and tail risk methods.

ES is a coherent risk measure satisfying subadditivity; VaR is not.
Basel III / FRTB moved from VaR to ES as the primary internal-models metric.
"""

import numpy as np
import pandas as pd
from scipy import stats


def historical_es(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Expected Shortfall (CVaR): mean of losses beyond the VaR threshold.
        ES = -E[r | r < VaR_alpha]

    ES is a coherent risk measure: it satisfies subadditivity, meaning
    ES(A+B) <= ES(A) + ES(B). VaR is NOT coherent — two portfolios with
    identical VaR can have very different tail risk profiles. ES captures
    the full shape of the tail beyond the VaR threshold; VaR only tells you
    the threshold itself.

    Basel III / FRTB replaced VaR with ES (at 97.5% confidence level) as
    the primary risk metric for the internal models approach. This reflects
    regulatory consensus that ES is a more complete description of tail risk
    and better captures the severity of extreme losses, not just their frequency.
    """
    returns_clean = returns.dropna()
    var_threshold = np.percentile(returns_clean, confidence_level * 100)
    tail_returns = returns_clean[returns_clean <= var_threshold]

    if len(tail_returns) == 0:
        return float(var_threshold)

    return float(tail_returns.mean())


def parametric_es(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Analytical ES under the normality assumption.

        ES = mu - sigma * phi(z_alpha) / alpha

    where phi is the standard normal PDF and z_alpha = norm.ppf(confidence_level).

    This is the closed-form solution for the conditional expectation of a
    Normal distribution below its alpha-th quantile. It will underestimate
    actual ES whenever returns have fat tails (excess kurtosis > 0), which
    is almost always the case for daily equity returns.
    """
    returns_clean = returns.dropna()
    mu = returns_clean.mean()
    sigma = returns_clean.std()

    z_alpha = stats.norm.ppf(confidence_level)
    phi_z = stats.norm.pdf(z_alpha)

    es = mu - sigma * phi_z / confidence_level
    return float(es)


def stressed_es(
    returns: pd.Series,
    stress_window: tuple,
    confidence_level: float = 0.05,
) -> float:
    """
    Stressed ES: compute ES over a specific historical stress period.

    Under Basel 2.5 and FRTB, banks must compute a Stressed VaR (and under
    FRTB a Stressed ES) using a continuous 12-month window of significant
    financial stress relevant to their portfolio. Canonical windows:
        2008-09-01 to 2009-09-01  — Global Financial Crisis
        2020-02-01 to 2020-06-01  — COVID-19 market shock

    stress_window: tuple of (start_date_str, end_date_str), e.g.
        ('2008-09-01', '2009-09-01')

    Returns the ES computed on the stressed sub-period. If the window
    falls outside the available data range, returns NaN with a warning.
    """
    start_str, end_str = stress_window
    returns_clean = returns.dropna()

    try:
        stressed = returns_clean.loc[start_str:end_str]
    except Exception:
        stressed = pd.Series(dtype=float)

    if len(stressed) < 10:
        import warnings
        warnings.warn(
            f"Stress window {stress_window} contains fewer than 10 observations. "
            "Returning NaN. Check that the returns index covers this period."
        )
        return float("nan")

    return historical_es(stressed, confidence_level)


def es_ratio(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    ES / VaR ratio: measures how heavy the tail is beyond the VaR threshold.

    A ratio close to 1.0 implies thin tails (Normal-like); ratios above 1.3
    indicate meaningful tail risk beyond VaR and suggest ES-based limits.
    """
    from src.var_models import historical_var
    var = historical_var(returns, confidence_level)
    es = historical_es(returns, confidence_level)

    if abs(var) < 1e-10:
        return float("nan")

    return float(es / var)
