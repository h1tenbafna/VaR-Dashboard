"""
Tests for src/backtesting.py
"""
import numpy as np
import pandas as pd
import pytest
from src.backtesting import (
    kupiec_test,
    christoffersen_test,
    basel_traffic_light,
    full_backtest_report,
)


def make_returns_and_var(n=500, true_rate=0.05, seed=0):
    """Synthetic series where violation rate = true_rate by construction."""
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0, 0.01, n))
    var_threshold = np.percentile(returns, true_rate * 100)
    var_series = pd.Series(var_threshold, index=returns.index)
    return returns, var_series


def make_clustered_violations(n=500, cluster_start=200, cluster_end=240, seed=1):
    """
    Series where violations cluster in [cluster_start, cluster_end].
    Outside the cluster: returns > var always (no violations).
    Inside the cluster: returns < var always (always violates).
    """
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0, 0.01, n))
    var_value = np.percentile(returns, 5)

    # Force no violations outside cluster
    returns.iloc[:cluster_start] = abs(returns.iloc[:cluster_start]) + 0.001
    returns.iloc[cluster_end:] = abs(returns.iloc[cluster_end:]) + 0.001
    # Force violations inside cluster
    returns.iloc[cluster_start:cluster_end] = -abs(returns.iloc[cluster_start:cluster_end]) - 0.001

    var_series = pd.Series(var_value, index=returns.index)
    return returns, var_series


# ── Kupiec ─────────────────────────────────────────────────────────────────

class TestKupiecTest:
    def test_correct_model_high_p_value(self):
        returns, var_series = make_returns_and_var(n=1000, true_rate=0.05)
        result = kupiec_test(returns, var_series, confidence_level=0.05)
        # Correctly calibrated model should not be rejected
        assert result['p_value'] > 0.05

    def test_too_many_violations_low_p_value(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0, 0.02, 500))
        # VaR set too conservatively: only blocks 1% of losses, not 5%
        var_series = pd.Series(np.percentile(returns, 1), index=returns.index)
        result = kupiec_test(returns, var_series, confidence_level=0.05)
        assert result['p_value'] < 0.05, "Over-violated model should fail Kupiec"

    def test_zero_violations(self):
        rng = np.random.default_rng(6)
        returns = pd.Series(rng.normal(0.1, 0.01, 250))  # all positive
        var_series = pd.Series(-0.5, index=returns.index)  # very conservative VaR
        result = kupiec_test(returns, var_series, confidence_level=0.05)
        assert result['n_violations'] == 0
        assert result['p_value'] == 1.0

    def test_returns_expected_keys(self):
        returns, var_series = make_returns_and_var()
        result = kupiec_test(returns, var_series)
        for key in ['n_observations', 'n_violations', 'violation_rate', 'lr_stat', 'p_value', 'passes']:
            assert key in result, f"Missing key: {key}"


# ── Christoffersen ─────────────────────────────────────────────────────────

class TestChristoffersenTest:
    def test_clustered_violations_fail_independence(self):
        returns, var_series = make_clustered_violations()
        result = christoffersen_test(returns, var_series, confidence_level=0.05)
        # Should fail independence test (violations cluster)
        assert result['p_value_ind'] < 0.05, (
            f"Clustered violations should fail independence. p_ind={result['p_value_ind']:.4f}"
        )

    def test_iid_violations_pass_independence(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        var_series = pd.Series(np.percentile(returns, 5), index=returns.index)
        result = christoffersen_test(returns, var_series, confidence_level=0.05)
        # i.i.d. violations should pass independence
        assert result['p_value_ind'] > 0.05, (
            f"IID violations should pass independence. p_ind={result['p_value_ind']:.4f}"
        )

    def test_transition_counts_sum_correctly(self):
        returns, var_series = make_returns_and_var(n=500)
        result = christoffersen_test(returns, var_series)
        total = result['n00'] + result['n01'] + result['n10'] + result['n11']
        assert total == result['n_observations'] - 1, (
            "Transition counts should sum to n_observations - 1"
        )

    def test_cc_stat_equals_uc_plus_ind(self):
        returns, var_series = make_returns_and_var(n=500)
        result = christoffersen_test(returns, var_series)
        assert abs(result['lr_cc'] - (result['lr_uc'] + result['lr_ind'])) < 1e-9

    def test_returns_all_keys(self):
        returns, var_series = make_returns_and_var()
        result = christoffersen_test(returns, var_series)
        required = [
            'n00', 'n01', 'n10', 'n11',
            'lr_uc', 'lr_ind', 'lr_cc',
            'p_value_uc', 'p_value_ind', 'p_value_cc',
            'passes_uc', 'passes_ind', 'passes_cc',
        ]
        for k in required:
            assert k in result, f"Missing key: {k}"


# ── Basel Traffic Light ────────────────────────────────────────────────────

class TestBaselTrafficLight:
    @pytest.mark.parametrize("n_viol,expected_zone", [
        (0, 'green'),
        (4, 'green'),
        (5, 'yellow'),
        (9, 'yellow'),
        (10, 'red'),
        (20, 'red'),
    ])
    def test_zone_boundaries(self, n_viol, expected_zone):
        result = basel_traffic_light(n_viol, n_observations=250)
        assert result['zone'] == expected_zone, (
            f"{n_viol} violations → expected '{expected_zone}', got '{result['zone']}'"
        )

    def test_green_no_multiplier(self):
        result = basel_traffic_light(3)
        assert result['capital_multiplier'] == 1.0

    def test_yellow_multiplier_increases(self):
        mult_5 = basel_traffic_light(5)['capital_multiplier']
        mult_9 = basel_traffic_light(9)['capital_multiplier']
        assert mult_9 > mult_5, "Yellow multiplier should increase with more violations"

    def test_red_max_multiplier(self):
        result = basel_traffic_light(15)
        assert result['capital_multiplier'] == 1.85

    def test_result_has_interpretation(self):
        result = basel_traffic_light(7)
        assert isinstance(result['interpretation'], str) and len(result['interpretation']) > 10


# ── Full Backtest Report ───────────────────────────────────────────────────

class TestFullBacktestReport:
    def test_combines_all_tests(self):
        returns, var_series = make_returns_and_var(n=500)
        report = full_backtest_report(returns, var_series, confidence_level=0.05)
        assert 'kupiec' in report
        assert 'christoffersen' in report
        assert 'basel_traffic_light' in report
        assert 'summary' in report

    def test_summary_zone_consistent_with_violations(self):
        returns, var_series = make_returns_and_var(n=250)
        report = full_backtest_report(returns, var_series, confidence_level=0.05)
        n_viol = report['summary']['n_violations']
        expected_zone = basel_traffic_light(n_viol, 250)['zone']
        assert report['summary']['zone'] == expected_zone
