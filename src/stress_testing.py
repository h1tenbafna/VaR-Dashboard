"""
Scenario-based stress testing engine.

Replaces the four hardcoded shocks in the original dashboard with a proper
scenario library covering six historical crises plus user-defined hypotheticals
and reverse stress testing.
"""

import numpy as np
import pandas as pd
from src.var_models import historical_var
from src.expected_shortfall import historical_es


PREDEFINED_SCENARIOS = {
    "2008 Financial Crisis (Sep-Dec 2008)": {
        "equity_shock": -0.42,
        "vol_shock": 2.5,
        "credit_shock": 0.03,
        "description": "Lehman Brothers collapse, global credit freeze",
    },
    "COVID-19 Crash (Feb-Mar 2020)": {
        "equity_shock": -0.34,
        "vol_shock": 3.0,
        "credit_shock": 0.02,
        "description": "Fastest 30% decline in S&P 500 history",
    },
    "Dot-com Bust (2000-2002)": {
        "equity_shock": -0.49,
        "vol_shock": 1.2,
        "credit_shock": 0.015,
        "description": "Tech sector collapse, 2.5-year bear market",
    },
    "1987 Black Monday": {
        "equity_shock": -0.226,
        "vol_shock": 4.0,
        "credit_shock": 0.01,
        "description": "Single-day 22.6% decline in DJIA",
    },
    "Taper Tantrum (2013)": {
        "equity_shock": -0.06,
        "vol_shock": 0.5,
        "credit_shock": 0.005,
        "description": "Fed signals QE tapering, emerging market selloff",
    },
    "2022 Rate Shock": {
        "equity_shock": -0.25,
        "vol_shock": 0.8,
        "credit_shock": 0.015,
        "description": "Fastest Fed rate hiking cycle since 1980s",
    },
}


def historical_scenario_var(
    returns: pd.Series,
    scenario_name: str,
    portfolio_value: float,
) -> dict:
    """
    Apply a predefined historical scenario shock to the current portfolio.

    The equity_shock is applied directly to portfolio_value to estimate
    the dollar loss. We also compute a stressed VaR by scaling the
    historical return distribution by the vol_shock multiplier — this
    reflects how tail quantiles expand when volatility is elevated.

    In practice at trading desks, scenario P&L is computed by repricing
    all positions under the shocked risk factors rather than scaling
    historical returns. This simplified approach is appropriate for a
    portfolio of equity ETFs where vol scaling is a reasonable proxy.
    """
    if scenario_name not in PREDEFINED_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(PREDEFINED_SCENARIOS.keys())}"
        )

    scenario = PREDEFINED_SCENARIOS[scenario_name]
    equity_shock = scenario["equity_shock"]
    vol_multiplier = scenario["vol_shock"]

    portfolio_loss = portfolio_value * equity_shock

    # Stressed VaR: scale returns by vol_shock to simulate elevated volatility
    stressed_returns = returns * vol_multiplier
    stressed_var = historical_var(stressed_returns, confidence_level=0.05)
    stressed_es = historical_es(stressed_returns, confidence_level=0.05)

    normal_var = historical_var(returns, confidence_level=0.05)
    var_ratio = (stressed_var / normal_var) if abs(normal_var) > 1e-10 else float("nan")

    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "equity_shock": equity_shock,
        "vol_multiplier": vol_multiplier,
        "credit_shock_bps": scenario["credit_shock"] * 10000,
        "portfolio_loss": portfolio_loss,
        "portfolio_loss_pct": equity_shock,
        "normal_var": normal_var,
        "stressed_var": stressed_var,
        "stressed_es": stressed_es,
        "var_ratio": var_ratio,
    }


def hypothetical_scenario_var(
    returns: pd.Series,
    equity_shock: float,
    vol_multiplier: float,
    portfolio_value: float,
) -> dict:
    """
    User-defined hypothetical scenario.

    equity_shock:    e.g., -0.20 for a 20% equity market decline
    vol_multiplier:  e.g., 2.0 to double current historical volatility
    """
    portfolio_loss = portfolio_value * equity_shock

    stressed_returns = returns * vol_multiplier
    stressed_var = historical_var(stressed_returns, confidence_level=0.05)
    stressed_es = historical_es(stressed_returns, confidence_level=0.05)
    normal_var = historical_var(returns, confidence_level=0.05)

    return {
        "scenario": "User-Defined Hypothetical",
        "equity_shock": equity_shock,
        "vol_multiplier": vol_multiplier,
        "portfolio_loss": portfolio_loss,
        "portfolio_loss_pct": equity_shock,
        "normal_var": normal_var,
        "stressed_var": stressed_var,
        "stressed_es": stressed_es,
    }


def reverse_stress_test(
    returns: pd.Series,
    portfolio_value: float,
    target_loss: float,
) -> dict:
    """
    Reverse stress test: find the market move that causes a specific portfolio loss.

    Given a target loss (e.g., 20% of portfolio value), solve for the equity
    shock that produces that loss. This is the direct inverse of a standard
    scenario test — rather than asking "how much do we lose in scenario X?",
    we ask "what scenario causes us to lose target_loss?".

    Reverse stress testing became a regulatory requirement under the Internal
    Capital Adequacy Assessment Process (ICAAP) post-2008. Regulators require
    firms to demonstrate they understand what it would take to break their
    business model — not just to quantify losses under plausible scenarios,
    but to identify the implausible-but-possible scenarios that could be
    catastrophic. This forces honest engagement with tail scenarios that
    standard scenario analysis might overlook.

    The implementation solves for the equity_shock analytically (loss = value * shock)
    and also computes the vol_multiplier that would be implied by that loss level
    based on the relationship between loss quantiles and the return distribution.
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")

    target_loss_pct = target_loss / portfolio_value
    implied_equity_shock = -abs(target_loss_pct)

    # What vol multiplier makes the 5th percentile of returns equal to implied_shock?
    current_var_pct = historical_var(returns, confidence_level=0.05)
    if abs(current_var_pct) > 1e-10:
        implied_vol_multiplier = abs(implied_equity_shock / current_var_pct)
    else:
        implied_vol_multiplier = float("nan")

    # Find the historical period closest to this shock level
    cumulative = (1 + returns).cumprod()
    drawdowns = cumulative / cumulative.expanding().max() - 1
    worst_drawdown = float(drawdowns.min())
    worst_date = drawdowns.idxmin()

    return {
        "target_loss": target_loss,
        "target_loss_pct": target_loss_pct,
        "implied_equity_shock": implied_equity_shock,
        "implied_vol_multiplier": implied_vol_multiplier,
        "worst_historical_drawdown": worst_drawdown,
        "worst_historical_drawdown_date": worst_date,
        "exceeds_historical_worst": abs(implied_equity_shock) > abs(worst_drawdown),
        "comparable_scenario": _find_comparable_scenario(implied_equity_shock),
    }


def _find_comparable_scenario(equity_shock: float) -> str:
    """Return the predefined scenario whose equity_shock is closest to the given value."""
    best_name = None
    best_diff = float("inf")
    for name, params in PREDEFINED_SCENARIOS.items():
        diff = abs(params["equity_shock"] - equity_shock)
        if diff < best_diff:
            best_diff = diff
            best_name = name
    return best_name or "None"


def run_all_scenarios(returns: pd.Series, portfolio_value: float) -> pd.DataFrame:
    """
    Run all predefined scenarios and return a summary DataFrame.
    """
    rows = []
    for name in PREDEFINED_SCENARIOS:
        result = historical_scenario_var(returns, name, portfolio_value)
        rows.append(
            {
                "Scenario": name,
                "Equity Shock": f"{result['equity_shock']:.1%}",
                "Portfolio Loss ($)": f"${abs(result['portfolio_loss']):,.0f}",
                "Normal VaR": f"{result['normal_var']:.2%}",
                "Stressed VaR": f"{result['stressed_var']:.2%}",
                "VaR Ratio": f"{result['var_ratio']:.2f}x",
                "Description": result["description"],
            }
        )
    return pd.DataFrame(rows)
