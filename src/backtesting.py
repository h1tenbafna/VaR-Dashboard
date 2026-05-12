"""
VaR backtesting: Kupiec POF test, Christoffersen conditional coverage test,
and Basel III traffic light assessment.
"""

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.05,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Null hypothesis: true violation rate = confidence_level (model is correctly
    calibrated on average).

    Test statistic:
        LR_uc = -2 * ln(L(p0) / L(p_hat))
               = -2 * [x*ln(p0) + (T-x)*ln(1-p0) - x*ln(p_hat) - (T-x)*ln(1-p_hat)]

    Under H0, LR_uc ~ chi-squared(1).

    var_series should be aligned with returns (same index). Both series should
    be expressed as return-level values (negative numbers for losses).
    """
    aligned = pd.DataFrame({"ret": returns, "var": var_series}).dropna()
    T = len(aligned)

    if T == 0:
        raise ValueError("No overlapping observations between returns and var_series")

    violations = (aligned["ret"] < aligned["var"]).sum()
    x = int(violations)
    p0 = confidence_level
    p_hat = x / T if T > 0 else 0.0

    if x == 0:
        lr_stat = 0.0
        p_value = 1.0
    elif x == T:
        lr_stat = float("inf")
        p_value = 0.0
    else:
        lr_stat = -2 * (
            x * np.log(p0 / p_hat) + (T - x) * np.log((1 - p0) / (1 - p_hat))
        )
        p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))

    return {
        "n_observations": T,
        "n_violations": x,
        "violation_rate": p_hat,
        "expected_violations": T * p0,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "passes": p_value > 0.05,
    }


def christoffersen_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.05,
) -> dict:
    """
    Christoffersen (1998) Conditional Coverage test.

    Extends Kupiec by jointly testing:
      1. Unconditional coverage (correct frequency of violations)
      2. Independence (violations should not cluster in time)

    In practice, violation clustering is the most common failure mode for
    VaR models during market stress — the model appears well-calibrated
    during normal markets but systematically understates risk when correlations
    spike and volatility regimes shift simultaneously. This is exactly what
    happened in September-December 2008 and February-March 2020: each
    individual day's loss exceeded VaR, but the model kept using quiet-period
    volatility to set the next day's limit.

    Transition matrix:
        n00: days with no violation following no violation
        n01: days with violation following no violation
        n10: days with no violation following violation
        n11: days with violation following violation

    LR_ind = -2 * ln(L(pi) / L(pi_00, pi_01, pi_10, pi_11))
    LR_cc  = LR_uc + LR_ind ~ chi-squared(2) under H0
    """
    aligned = pd.DataFrame({"ret": returns, "var": var_series}).dropna()
    T = len(aligned)

    if T < 2:
        raise ValueError("Need at least 2 observations for Christoffersen test")

    hits = (aligned["ret"] < aligned["var"]).astype(int).values

    # Count transitions
    n00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
    n01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
    n10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
    n11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11)  # pooled violation rate

    # Kupiec (unconditional coverage)
    kupiec = kupiec_test(returns, var_series, confidence_level)
    lr_uc = kupiec["lr_stat"]

    # Independence LR
    def safe_log_ratio(a, b):
        if a <= 0 or b <= 0:
            return 0.0
        return a * np.log(b)

    # Log-likelihood under restricted model (same pi for all transitions)
    ll_restricted = (
        safe_log_ratio(n00 + n10, 1 - pi_hat)
        + safe_log_ratio(n01 + n11, pi_hat)
    )
    # Log-likelihood under unrestricted model
    ll_unrestricted = (
        safe_log_ratio(n00, 1 - pi_01)
        + safe_log_ratio(n01, pi_01)
        + safe_log_ratio(n10, 1 - pi_11)
        + safe_log_ratio(n11, pi_11)
    )

    lr_ind = -2 * (ll_restricted - ll_unrestricted)
    lr_ind = max(0.0, lr_ind)  # numerical safety

    lr_cc = lr_uc + lr_ind

    p_value_uc = float(1 - stats.chi2.cdf(lr_uc, df=1))
    p_value_ind = float(1 - stats.chi2.cdf(lr_ind, df=1))
    p_value_cc = float(1 - stats.chi2.cdf(lr_cc, df=2))

    return {
        "n_observations": T,
        "n_violations": kupiec["n_violations"],
        "violation_rate": kupiec["violation_rate"],
        "expected_violations": kupiec["expected_violations"],
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi_01": pi_01,
        "pi_11": pi_11,
        "lr_uc": lr_uc,
        "lr_ind": lr_ind,
        "lr_cc": lr_cc,
        "p_value_uc": p_value_uc,
        "p_value_ind": p_value_ind,
        "p_value_cc": p_value_cc,
        "passes_uc": p_value_uc > 0.05,
        "passes_ind": p_value_ind > 0.05,
        "passes_cc": p_value_cc > 0.05,
    }


def basel_traffic_light(n_violations: int, n_observations: int = 250) -> dict:
    """
    Basel III traffic light assessment over 250 trading days at 99% VaR.

    Green zone:  0-4 violations  — model acceptable, no capital add-on
    Yellow zone: 5-9 violations  — increasing capital multiplier (1.13 to 1.85)
    Red zone:    10+ violations  — presumption of model failure, multiplier 1.85

    The Basel traffic light was calibrated for 99% VaR over 250 days.
    It implicitly embeds a binomial test: the zone boundaries correspond to
    regions where the observed violation count is statistically consistent
    (or inconsistent) with a true 1% violation probability. At 2.5 expected
    violations per year, seeing 10+ is extremely unlikely under a correctly
    specified model (~0.01% probability), hence the red zone presumption.

    Yellow zone capital multipliers (Basel III, Table 1):
        5 → 1.13, 6 → 1.17, 7 → 1.22, 8 → 1.25, 9 → 1.28 (→ 1.33 in some jurisdictions)
    """
    yellow_multipliers = {5: 1.13, 6: 1.17, 7: 1.22, 8: 1.25, 9: 1.28}

    if n_violations <= 4:
        zone = "green"
        multiplier = 1.0
        interpretation = (
            f"{n_violations} violations in {n_observations} days — model passes "
            "regulatory backtesting. No capital add-on required."
        )
    elif n_violations <= 9:
        zone = "yellow"
        multiplier = yellow_multipliers.get(n_violations, 1.28)
        interpretation = (
            f"{n_violations} violations in {n_observations} days — model in yellow zone. "
            f"Capital multiplier {multiplier:.2f}x applied. Model should be reviewed."
        )
    else:
        zone = "red"
        multiplier = 1.85
        interpretation = (
            f"{n_violations} violations in {n_observations} days — model in red zone. "
            "Presumption of model failure. Multiplier 1.85x. Immediate recalibration required."
        )

    return {
        "zone": zone,
        "capital_multiplier": multiplier,
        "n_violations": n_violations,
        "n_observations": n_observations,
        "interpretation": interpretation,
    }


def full_backtest_report(
    returns: pd.Series,
    var_series: pd.Series,
    confidence_level: float = 0.05,
) -> dict:
    """
    Run complete backtest: Kupiec + Christoffersen + Basel traffic light.

    var_series must be aligned with returns (same index). For rolling VaR
    backtests, compute var_series as a rolling quantile of the trailing window
    before passing it here.

    Returns a combined results dict suitable for display or logging.
    """
    kupiec = kupiec_test(returns, var_series, confidence_level)
    christoffersen = christoffersen_test(returns, var_series, confidence_level)

    n_violations = kupiec["n_violations"]
    n_obs = kupiec["n_observations"]
    traffic = basel_traffic_light(n_violations, n_obs)

    return {
        "kupiec": kupiec,
        "christoffersen": christoffersen,
        "basel_traffic_light": traffic,
        "summary": {
            "n_observations": n_obs,
            "n_violations": n_violations,
            "violation_rate": kupiec["violation_rate"],
            "expected_rate": confidence_level,
            "passes_kupiec": kupiec["passes"],
            "passes_christoffersen_uc": christoffersen["passes_uc"],
            "passes_christoffersen_ind": christoffersen["passes_ind"],
            "passes_christoffersen_cc": christoffersen["passes_cc"],
            "zone": traffic["zone"],
            "capital_multiplier": traffic["capital_multiplier"],
        },
    }
