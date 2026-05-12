"""
Component VaR, Marginal VaR, and portfolio risk attribution.

These are the standard decomposition tools used in risk attribution
at banks and asset managers. The key identity: sum(Component VaR) = Portfolio VaR.
"""

import numpy as np
import pandas as pd
from scipy import stats
from src.var_models import historical_var


def component_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.05,
) -> pd.Series:
    """
    Component VaR: contribution of each position to portfolio VaR.

    For a portfolio with weights w and covariance matrix Sigma:
        Portfolio variance:  sigma_p^2 = w' * Sigma * w
        Portfolio VaR:       VaR_p = z_alpha * sigma_p  (Normal approx)
        Marginal VaR:        MVaR_i = z_alpha * (Sigma @ w)_i / sigma_p
        Component VaR:       CVaR_i = w_i * MVaR_i

    Key identity: sum(CVaR_i) = Portfolio VaR (exact by Euler's homogeneous
    function theorem, since VaR is degree-1 homogeneous in weights).

    Sign convention: VaR and component VaR are expressed as negative returns
    (losses are negative). Component VaR_i < 0 means position i is ADDING to
    portfolio loss; component VaR_i > 0 means it is REDUCING total portfolio
    loss (acting as a hedge). Sum of component VaRs = portfolio VaR (negative).

    The risk-reduction question is: which positions have the most negative
    component VaR? Those are driving the loss. Positions with positive
    component VaR are providing diversification — removing them would worsen VaR.
    """
    weights = np.array(weights, dtype=float)
    cov_matrix = returns.cov().values

    sigma_p_sq = float(weights @ cov_matrix @ weights)
    sigma_p = np.sqrt(sigma_p_sq) if sigma_p_sq > 0 else 1e-10

    z_alpha = stats.norm.ppf(confidence_level)

    # Marginal contribution of each position to portfolio variance
    marginal_contrib = cov_matrix @ weights

    # Portfolio VaR (Normal parametric)
    portfolio_var_norm = z_alpha * sigma_p

    # Component VaR = weight_i * (partial VaR_p / partial w_i)
    component_vars = weights * (z_alpha * marginal_contrib / sigma_p)

    return pd.Series(component_vars, index=returns.columns, name="component_var")


def marginal_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.05,
) -> pd.Series:
    """
    Marginal VaR: rate of change of portfolio VaR per unit increase in position.

        MVaR_i = correlation(r_i, r_p) * VaR_portfolio / w_i_dollar

    Expressed here relative to a $1 increase in notional exposure to asset i,
    with portfolio VaR computed under the Normal parametric approach.

    Marginal VaR answers the portfolio construction question: "if I add $1
    to position i and offset it by reducing the residual cash balance, how
    much does portfolio VaR change?" This drives position sizing and risk
    budget allocation decisions. Positions with the highest marginal VaR
    consume the most risk budget per dollar of exposure.
    """
    weights = np.array(weights, dtype=float)
    cov_matrix = returns.cov().values

    sigma_p_sq = float(weights @ cov_matrix @ weights)
    sigma_p = np.sqrt(sigma_p_sq) if sigma_p_sq > 0 else 1e-10

    z_alpha = stats.norm.ppf(confidence_level)
    portfolio_var_norm = z_alpha * sigma_p

    marginal_contrib = cov_matrix @ weights
    mvars = z_alpha * marginal_contrib / sigma_p

    return pd.Series(mvars, index=returns.columns, name="marginal_var")


def standalone_var(
    returns: pd.DataFrame,
    confidence_level: float = 0.05,
) -> pd.Series:
    """Historical VaR for each individual position (ignoring correlations)."""
    result = {}
    for col in returns.columns:
        result[col] = historical_var(returns[col].dropna(), confidence_level)
    return pd.Series(result, name="standalone_var")


def var_attribution_report(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.05,
) -> pd.DataFrame:
    """
    Full VaR attribution table with component and marginal VaR per position.

    Returns a DataFrame with columns:
        Position | Weight | Standalone VaR | Component VaR | % of Total VaR | Marginal VaR

    This is the standard format for risk attribution reports at sell-side
    risk management desks. The "% of Total VaR" column is the key output:
    it shows which positions are responsible for portfolio risk, accounting
    for diversification. A position might represent 25% of portfolio weight
    but only 10% of VaR if it is lowly correlated with the rest of the book,
    or 35% of VaR if it is highly correlated.

    Diversification benefit = sum(standalone VaRs) - portfolio VaR.
    """
    weights = np.array(weights, dtype=float)
    n_assets = len(weights)

    if n_assets != returns.shape[1]:
        raise ValueError(
            f"weights length {n_assets} != number of assets {returns.shape[1]}"
        )

    cvars = component_var(returns, weights, confidence_level)
    mvars = marginal_var(returns, weights, confidence_level)
    svars = standalone_var(returns, confidence_level)

    portfolio_var_total = float(cvars.sum())

    pct_of_total = (cvars / portfolio_var_total * 100) if abs(portfolio_var_total) > 1e-10 else cvars * 0

    report = pd.DataFrame(
        {
            "Position": returns.columns,
            "Weight": weights,
            "Standalone VaR": svars.values,
            "Component VaR": cvars.values,
            "% of Total VaR": pct_of_total.values,
            "Marginal VaR": mvars.values,
        }
    )

    # Totals row
    sum_standalone = svars.sum()
    diversification_benefit = sum_standalone - portfolio_var_total

    totals = pd.DataFrame(
        [
            {
                "Position": "TOTAL / Diversification",
                "Weight": weights.sum(),
                "Standalone VaR": sum_standalone,
                "Component VaR": portfolio_var_total,
                "% of Total VaR": 100.0,
                "Marginal VaR": float("nan"),
            }
        ]
    )

    report = pd.concat([report, totals], ignore_index=True)
    report.attrs["diversification_benefit"] = diversification_benefit

    return report
