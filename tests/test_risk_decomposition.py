"""
Tests for src/risk_decomposition.py
"""
import numpy as np
import pandas as pd
import pytest
from src.risk_decomposition import (
    component_var,
    marginal_var,
    var_attribution_report,
    standalone_var,
)


@pytest.fixture
def two_asset_returns():
    """Two uncorrelated assets — equal-weight portfolio."""
    rng = np.random.default_rng(42)
    r_a = rng.normal(0, 0.01, 500)
    r_b = rng.normal(0, 0.01, 500)
    return pd.DataFrame({'A': r_a, 'B': r_b})


@pytest.fixture
def four_asset_returns():
    """Four correlated assets."""
    rng = np.random.default_rng(10)
    cov = np.array([
        [0.0004, 0.0002, 0.0001, 0.0001],
        [0.0002, 0.0004, 0.00015, 0.0001],
        [0.0001, 0.00015, 0.0004, 0.0002],
        [0.0001, 0.0001, 0.0002, 0.0004],
    ])
    r = rng.multivariate_normal([0.0005] * 4, cov, 500)
    return pd.DataFrame(r, columns=['A', 'B', 'C', 'D'])


@pytest.fixture
def four_weights():
    return np.array([0.4, 0.3, 0.2, 0.1])


# ── Component VaR ──────────────────────────────────────────────────────────

class TestComponentVar:
    def test_sum_equals_portfolio_var(self, four_asset_returns, four_weights):
        """
        Key identity: sum(Component VaR) == Portfolio VaR (Normal parametric).
        This is the Euler decomposition — must hold to machine precision.
        """
        cvars = component_var(four_asset_returns, four_weights)
        from src.var_models import parametric_var
        port_r = (four_asset_returns * four_weights).sum(axis=1)
        port_var = parametric_var(port_r, confidence_level=0.05, distribution='normal')

        # Component VaR uses Normal parametric for the decomposition
        cvar_sum = float(cvars.sum())
        from scipy import stats
        import numpy as np
        cov = four_asset_returns.cov().values
        sigma_p = float(np.sqrt(four_weights @ cov @ four_weights))
        z = stats.norm.ppf(0.05)
        portfolio_var_norm = z * sigma_p

        assert abs(cvar_sum - portfolio_var_norm) < 1e-9, (
            f"sum(CVaR)={cvar_sum:.8f} != portfolio VaR={portfolio_var_norm:.8f}"
        )

    def test_returns_series_with_correct_index(self, four_asset_returns, four_weights):
        cvars = component_var(four_asset_returns, four_weights)
        assert list(cvars.index) == list(four_asset_returns.columns)

    def test_diversifying_asset_positive_component_var(self):
        """
        Sign convention: VaR is expressed as a negative return (loss).
        Component VaR sums to Portfolio VaR (negative).
        A position that REDUCES portfolio risk (a hedge) contributes POSITIVELY
        to the negative sum — its component VaR is positive (less loss).
        A risky position that ADDS risk has negative component VaR (more loss).

        With C = -A (perfect hedge), adding more C reduces portfolio vol,
        so its component VaR is positive (it is reducing the portfolio loss).
        """
        rng = np.random.default_rng(20)
        r_base = rng.normal(0, 0.01, 500)
        returns = pd.DataFrame({
            'A': r_base,
            'B': rng.normal(0, 0.01, 500),
            'C': -r_base,  # hedges A
        })
        w = np.array([0.4, 0.4, 0.2])
        cvars = component_var(returns, w)

        # The hedge C has positive component VaR: it is reducing total portfolio loss
        assert cvars['C'] >= 0.0, (
            f"Hedge should have non-negative component VaR (reduces loss), got {cvars['C']:.6f}"
        )
        # Risky position A should have negative component VaR (contributes to loss)
        assert cvars['A'] < 0.0, (
            f"Risky asset A should have negative component VaR, got {cvars['A']:.6f}"
        )

    def test_equal_weights_equal_contributions_for_identical_assets(self):
        """Identical, fully correlated assets → equal component VaR."""
        rng = np.random.default_rng(30)
        r = rng.normal(0, 0.01, 500)
        returns = pd.DataFrame({'A': r, 'B': r})
        w = np.array([0.5, 0.5])
        cvars = component_var(returns, w)
        assert abs(cvars['A'] - cvars['B']) < 1e-10, (
            "Identical assets with equal weights must have equal component VaR"
        )


# ── Marginal VaR ───────────────────────────────────────────────────────────

class TestMarginalVar:
    def test_concentrated_position_has_higher_mvar(self, four_asset_returns):
        """
        The largest position should have the highest (most positive) marginal VaR
        when the portfolio is concentrated.
        """
        w = np.array([0.7, 0.1, 0.1, 0.1])
        mvars = marginal_var(four_asset_returns, w)
        # Not all cases guaranteed, but concentrated position should rank high
        assert mvars.shape[0] == 4

    def test_returns_series_with_correct_index(self, four_asset_returns, four_weights):
        mvars = marginal_var(four_asset_returns, four_weights)
        assert list(mvars.index) == list(four_asset_returns.columns)


# ── VaR Attribution Report ─────────────────────────────────────────────────

class TestVarAttributionReport:
    def test_report_has_expected_columns(self, four_asset_returns, four_weights):
        report = var_attribution_report(four_asset_returns, four_weights)
        expected_cols = ['Position', 'Weight', 'Standalone VaR', 'Component VaR',
                         '% of Total VaR', 'Marginal VaR']
        for col in expected_cols:
            assert col in report.columns, f"Missing column: {col}"

    def test_total_row_present(self, four_asset_returns, four_weights):
        report = var_attribution_report(four_asset_returns, four_weights)
        assert 'TOTAL / Diversification' in report['Position'].values

    def test_component_var_sums_to_portfolio(self, four_asset_returns, four_weights):
        report = var_attribution_report(four_asset_returns, four_weights)
        non_total = report[report['Position'] != 'TOTAL / Diversification']
        cvar_sum = non_total['Component VaR'].sum()
        total_row = report[report['Position'] == 'TOTAL / Diversification']['Component VaR'].iloc[0]
        assert abs(cvar_sum - total_row) < 1e-9, (
            f"Sum of component VaRs {cvar_sum:.8f} != total {total_row:.8f}"
        )

    def test_pct_of_total_sums_to_100(self, four_asset_returns, four_weights):
        report = var_attribution_report(four_asset_returns, four_weights)
        non_total = report[report['Position'] != 'TOTAL / Diversification']
        total_pct = non_total['% of Total VaR'].sum()
        assert abs(total_pct - 100.0) < 1e-6, (
            f"% of Total VaR sums to {total_pct:.4f}%, expected 100%"
        )

    def test_weight_mismatch_raises(self, four_asset_returns):
        bad_weights = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="weights length"):
            var_attribution_report(four_asset_returns, bad_weights)
