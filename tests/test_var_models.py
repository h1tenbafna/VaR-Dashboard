"""
Tests for src/var_models.py
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.var_models import (
    historical_var,
    parametric_var,
    filtered_historical_simulation_var,
    monte_carlo_var,
    cornish_fisher_var,
    portfolio_var,
)


@pytest.fixture
def normal_returns():
    """Synthetic normally-distributed returns (1000 days)."""
    rng = np.random.default_rng(0)
    r = rng.normal(loc=0.0005, scale=0.01, size=1000)
    return pd.Series(r)


@pytest.fixture
def fat_tail_returns():
    """Student-t returns with 4 dof — fat tails, excess kurtosis = 6."""
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=4, size=1000) * 0.01
    return pd.Series(r)


@pytest.fixture
def volatile_returns():
    """Returns with a mid-period volatility spike (mimics a crisis)."""
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 500)
    crisis = rng.normal(0, 0.03, 500)  # 6x volatility
    return pd.Series(np.concatenate([calm, crisis]))


@pytest.fixture
def multi_asset_returns():
    """Four correlated assets, 500 days."""
    rng = np.random.default_rng(3)
    cov = np.array([
        [0.0001, 0.00006, 0.00004, 0.00002],
        [0.00006, 0.0001, 0.00005, 0.00003],
        [0.00004, 0.00005, 0.0001, 0.00004],
        [0.00002, 0.00003, 0.00004, 0.0001],
    ])
    r = rng.multivariate_normal([0.0005] * 4, cov, 500)
    return pd.DataFrame(r, columns=['A', 'B', 'C', 'D'])


# ── Historical VaR ─────────────────────────────────────────────────────────

class TestHistoricalVar:
    def test_violation_rate_close_to_confidence(self, normal_returns):
        var = historical_var(normal_returns, confidence_level=0.05)
        violations = (normal_returns < var).mean()
        assert abs(violations - 0.05) < 0.02, (
            f"Violation rate {violations:.3f} too far from 5%"
        )

    def test_returns_negative_value(self, normal_returns):
        var = historical_var(normal_returns, confidence_level=0.05)
        assert var < 0, "VaR should be a negative return (loss)"

    def test_99_var_more_extreme_than_95(self, normal_returns):
        var_95 = historical_var(normal_returns, confidence_level=0.05)
        var_99 = historical_var(normal_returns, confidence_level=0.01)
        assert var_99 < var_95, "99% VaR must be more extreme than 95% VaR"

    def test_horizon_scaling(self, normal_returns):
        var_1d = historical_var(normal_returns, confidence_level=0.05, horizon=1)
        var_5d = historical_var(normal_returns, confidence_level=0.05, horizon=5)
        expected = var_1d * np.sqrt(5)
        assert abs(var_5d - expected) < 1e-10, "sqrt(h) scaling not applied correctly"

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="No valid"):
            historical_var(pd.Series([], dtype=float))


# ── Parametric VaR ─────────────────────────────────────────────────────────

class TestParametricVar:
    def test_normal_close_to_theoretical(self):
        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0, 0.01, 5000))
        var = parametric_var(r, confidence_level=0.05)
        expected = stats.norm.ppf(0.05, 0, 0.01)
        assert abs(var - expected) < 0.001, (
            f"Parametric Normal VaR {var:.4f} too far from theoretical {expected:.4f}"
        )

    def test_student_t_larger_than_normal_at_99(self, fat_tail_returns):
        var_normal = parametric_var(fat_tail_returns, confidence_level=0.01, distribution='normal')
        var_t = parametric_var(fat_tail_returns, confidence_level=0.01, distribution='student_t')
        assert var_t < var_normal, (
            "Student-t VaR should be more extreme than Normal VaR on fat-tail returns at 99%"
        )

    def test_invalid_distribution_raises(self, normal_returns):
        with pytest.raises(ValueError, match="distribution must be"):
            parametric_var(normal_returns, distribution='laplace')


# ── FHS VaR ────────────────────────────────────────────────────────────────

class TestFHSVar:
    def test_fhs_adapts_to_volatility_spike(self, volatile_returns):
        """FHS on only the last 200 (high-vol) observations should give more
        extreme VaR than FHS on only the first 200 (low-vol) observations,
        because EWMA volatility is high at the end of the series."""
        calm_half = volatile_returns.iloc[:200].reset_index(drop=True)
        crisis_half = volatile_returns.iloc[500:700].reset_index(drop=True)

        var_calm = filtered_historical_simulation_var(calm_half, confidence_level=0.05)
        var_crisis = filtered_historical_simulation_var(crisis_half, confidence_level=0.05)

        assert var_crisis < var_calm, (
            f"FHS VaR should be more extreme in high-vol period. "
            f"calm={var_calm:.4f}, crisis={var_crisis:.4f}"
        )

    def test_fhs_returns_negative_value(self, normal_returns):
        var = filtered_historical_simulation_var(normal_returns)
        assert var < 0


# ── Monte Carlo VaR ────────────────────────────────────────────────────────

class TestMonteCarloVar:
    def test_close_to_parametric_normal(self, normal_returns):
        var_mc, se = monte_carlo_var(normal_returns, confidence_level=0.05, n_simulations=200_000, use_ewma=False)
        var_param = parametric_var(normal_returns, confidence_level=0.05)
        # Within 3 std errors
        assert abs(var_mc - var_param) < 3 * se + 0.002, (
            f"MC VaR {var_mc:.4f} too far from parametric {var_param:.4f}"
        )

    def test_se_decreases_with_more_simulations(self, normal_returns):
        _, se_low = monte_carlo_var(normal_returns, n_simulations=1_000, use_ewma=False)
        _, se_high = monte_carlo_var(normal_returns, n_simulations=100_000, use_ewma=False)
        assert se_high < se_low, "SE should decrease with more simulations"

    def test_returns_tuple(self, normal_returns):
        result = monte_carlo_var(normal_returns)
        assert isinstance(result, tuple) and len(result) == 2


# ── Cornish-Fisher VaR ─────────────────────────────────────────────────────

class TestCornishFisherVar:
    def test_cf_more_extreme_than_normal_at_99_with_excess_kurtosis(self, fat_tail_returns):
        """
        At 99% VaR (1% quantile), positive excess kurtosis pushes z_CF below z_Normal
        (z^3-3z < 0 at z=-2.33), making CF VaR more extreme than Normal.
        At 95% VaR (5% quantile) the correction goes the other direction — tested separately.
        """
        cf_var = cornish_fisher_var(fat_tail_returns, confidence_level=0.01)
        normal_var = parametric_var(fat_tail_returns, confidence_level=0.01, distribution='normal')
        assert cf_var < normal_var, (
            "At 99% confidence, CF VaR should be more extreme than Normal VaR when excess kurtosis > 0"
        )

    def test_negative_skew_makes_var_worse(self):
        """Negative skewness should push CF VaR more extreme."""
        rng = np.random.default_rng(99)
        # Negatively skewed: mix of lognormal
        r_sym = pd.Series(rng.normal(0, 0.01, 2000))
        # Introduce negative skew by subtracting occasional large draws
        r_skewed = r_sym.copy()
        idx = rng.choice(2000, 50, replace=False)
        r_skewed.iloc[idx] -= 0.05

        cf_sym = cornish_fisher_var(r_sym)
        cf_skewed = cornish_fisher_var(r_skewed)
        assert cf_skewed < cf_sym, "Negative skewness should worsen CF VaR"


# ── Portfolio VaR & Diversification ───────────────────────────────────────

class TestPortfolioDiversification:
    def test_portfolio_var_le_sum_of_standalone(self, multi_asset_returns):
        """Diversification benefit: portfolio VaR ≤ sum of individual VaRs."""
        n = multi_asset_returns.shape[1]
        weights = np.array([1 / n] * n)

        port_var = portfolio_var(multi_asset_returns, weights, method='historical')
        standalone_sum = sum(
            historical_var(multi_asset_returns[col], confidence_level=0.05)
            for col in multi_asset_returns.columns
        )

        assert port_var >= standalone_sum, (
            f"Portfolio VaR {port_var:.4f} should be >= (less negative than) "
            f"sum of standalones {standalone_sum:.4f} (diversification benefit)"
        )
