"""
VaR estimation methods: Historical Simulation, Parametric (Normal/Student-t),
Filtered Historical Simulation, Monte Carlo, and Cornish-Fisher expansion.
"""

import numpy as np
import pandas as pd
from scipy import stats
from src.returns import compute_ewma_volatility


def historical_var(
    returns: pd.Series,
    confidence_level: float = 0.05,
    horizon: int = 1,
    scaling: str = "sqrt_t",
) -> float:
    """
    Non-parametric VaR from the empirical return distribution.

    For h-day horizon there are two approaches:
      'sqrt_t'    — scale 1-day VaR by sqrt(h). Valid only under i.i.d. returns;
                    underestimates tail risk when returns exhibit autocorrelation
                    or volatility clustering (e.g., during sustained stress periods).
      'overlapping' — compute h-day returns using overlapping windows, then take
                    the alpha-th percentile directly. More accurate but produces
                    fewer independent observations, increasing estimation noise.

    In practice, sqrt(h) scaling is the Basel III default for internal models,
    but regulators are aware it understates risk during crisis regimes.
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        raise ValueError("No valid returns after removing NaNs")

    if scaling == "overlapping" and horizon > 1:
        h_returns = returns_clean.rolling(horizon).sum().dropna()
        return float(np.percentile(h_returns, confidence_level * 100))

    var_1d = float(np.percentile(returns_clean, confidence_level * 100))
    return var_1d * np.sqrt(horizon)


def parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.05,
    horizon: int = 1,
    distribution: str = "normal",
) -> float:
    """
    Parametric VaR assuming a distribution for returns.

    For distribution='normal':
        VaR = mu - z_alpha * sigma
        where z_alpha = scipy.stats.norm.ppf(confidence_level)

    For distribution='student_t':
        Fit degrees of freedom via MLE on the return series.
        VaR = mu - t_alpha(nu) * sigma * sqrt((nu-2)/nu)
        The fatter tails of the Student-t typically give higher VaR than Normal,
        especially at the 99% confidence level.

    This is why Normal VaR was systematically underestimating risk pre-2008:
    equity returns exhibit excess kurtosis (fat tails) that the Normal
    distribution cannot capture, leading to consistent underestimation of
    extreme loss probabilities.
    """
    returns_clean = returns.dropna()
    mu = returns_clean.mean()
    sigma = returns_clean.std()

    if distribution == "normal":
        z = stats.norm.ppf(confidence_level)
        var_1d = mu + z * sigma

    elif distribution == "student_t":
        # MLE fit of Student-t degrees of freedom
        params = stats.t.fit(returns_clean, floc=mu)
        nu = params[0]  # degrees of freedom
        t_alpha = stats.t.ppf(confidence_level, df=nu)
        # Scale sigma to match Student-t variance: Var(t_nu) = nu/(nu-2) for nu>2
        sigma_t = sigma * np.sqrt((nu - 2) / nu) if nu > 2 else sigma
        var_1d = mu + t_alpha * sigma_t

    else:
        raise ValueError(f"distribution must be 'normal' or 'student_t', got '{distribution}'")

    return var_1d * np.sqrt(horizon)


def filtered_historical_simulation_var(
    returns: pd.Series,
    confidence_level: float = 0.05,
    lambda_: float = 0.94,
) -> float:
    """
    Filtered Historical Simulation (Hull-White, 1998).

    Key insight: standardise historical returns by their conditional volatility
    estimate, apply historical simulation on the standardised residuals, then
    scale back by current (most recent) EWMA volatility.

    Steps:
      1. Estimate EWMA volatility: sigma_t = sqrt(EWMA variance)
      2. Compute standardised residuals: z_t = r_t / sigma_t
      3. Apply historical simulation on z_t: q_alpha = percentile(z_t, alpha*100)
      4. Scale back: VaR = -sigma_{T+1} * q_alpha

    FHS addresses the main practical weakness of plain historical simulation:
    stale volatility estimates. When vol spikes suddenly (as in March 2020),
    historical returns scaled by quiet-period vol dramatically understate
    current tail risk. FHS decouples the empirical shape of the residual
    distribution from the current volatility level.

    FHS is the industry standard at most major banks for the internal models
    approach under Basel III precisely because it combines the distribution-free
    nature of historical simulation with time-varying volatility.
    """
    returns_clean = returns.dropna()

    ewma_vol = compute_ewma_volatility(returns_clean, lambda_=lambda_)

    # Replace any zero-vol estimates (can occur at series start) with series mean
    ewma_vol = ewma_vol.replace(0, ewma_vol[ewma_vol > 0].mean())

    z_t = returns_clean / ewma_vol
    z_t = z_t.dropna()

    q_alpha = float(np.percentile(z_t, confidence_level * 100))
    sigma_current = float(ewma_vol.iloc[-1])

    return sigma_current * q_alpha  # negative value = loss


def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = 0.05,
    n_simulations: int = 100_000,
    horizon: int = 1,
    use_ewma: bool = True,
    random_seed: int = 42,
) -> tuple:
    """
    Monte Carlo VaR with optional EWMA volatility.

    If use_ewma=True:
        Use current EWMA volatility estimate — more accurate than using
        full-period historical vol, especially after volatility regime shifts.
    If use_ewma=False:
        Use historical mean and std (replicates simple parametric MC).

    Returns (VaR estimate, standard error of estimate).

    Standard error formula:
        SE = sigma * z_alpha * sqrt(alpha*(1-alpha) / (n * f(z_alpha)^2))
    where f is the standard normal PDF at the quantile z_alpha.

    100,000 paths is the minimum recommended for 99% VaR: at 99% confidence
    we are estimating the 1st percentile, so we only have ~100 observations
    even with 10K paths — far too few for stable quantile estimation.
    Variance of the quantile estimator shrinks as 1/n, so 100K gives ~10x
    more stable estimates than 10K.
    """
    returns_clean = returns.dropna()
    mu = returns_clean.mean()

    if use_ewma:
        ewma_vol = compute_ewma_volatility(returns_clean, lambda_=0.94)
        sigma = float(ewma_vol.iloc[-1])
    else:
        sigma = float(returns_clean.std())

    rng = np.random.default_rng(random_seed)

    if horizon == 1:
        sim = rng.normal(mu, sigma, n_simulations)
    else:
        # Multi-period: sum of daily draws under i.i.d. assumption
        daily = rng.normal(mu, sigma, (n_simulations, horizon))
        sim = daily.sum(axis=1)

    var_estimate = float(np.percentile(sim, confidence_level * 100))

    # Standard error of the quantile estimate
    z_alpha = stats.norm.ppf(confidence_level)
    f_z = stats.norm.pdf(z_alpha)
    se = sigma * np.sqrt(confidence_level * (1 - confidence_level) / (n_simulations * f_z ** 2))

    return var_estimate, float(se)


def cornish_fisher_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Cornish-Fisher VaR: adjust the Normal quantile for skewness and excess kurtosis.

    Adjusted quantile:
        z_CF = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

    where:
        z = scipy.stats.norm.ppf(confidence_level)
        S = skewness of returns
        K = excess kurtosis of returns

    VaR_CF = -(mu + z_CF * sigma)

    Cornish-Fisher is a first-order correction for non-normality via the
    Edgeworth expansion. It works well for mild skewness/kurtosis but breaks
    down for severe tail events — for those, Student-t or FHS are more robust.

    Equity returns typically exhibit negative skewness (left tail is fatter
    than the right — crashes are more common than equivalent-magnitude rallies)
    and excess kurtosis (fat tails). Both make Normal VaR systematically
    optimistic. CF partially corrects this without requiring a full distributional
    assumption, making it a useful quick diagnostic for portfolio risk reports.
    """
    returns_clean = returns.dropna()
    mu = returns_clean.mean()
    sigma = returns_clean.std()
    S = float(returns_clean.skew())
    K = float(returns_clean.kurtosis())  # excess kurtosis (scipy default)

    z = stats.norm.ppf(confidence_level)

    z_cf = (
        z
        + (z ** 2 - 1) * S / 6
        + (z ** 3 - 3 * z) * K / 24
        - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    )

    return mu + z_cf * sigma  # negative value = loss


def portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_level: float = 0.05,
    method: str = "historical",
    **kwargs,
) -> float:
    """
    Portfolio-level VaR dispatcher: computes weighted portfolio returns
    then delegates to the chosen single-asset VaR method.
    """
    weights = np.array(weights)
    portfolio_returns = returns.values @ weights
    portfolio_series = pd.Series(portfolio_returns, index=returns.index)

    dispatch = {
        "historical": historical_var,
        "parametric": parametric_var,
        "fhs": filtered_historical_simulation_var,
        "monte_carlo": lambda r, cl, **kw: monte_carlo_var(r, cl, **kw)[0],
        "cornish_fisher": cornish_fisher_var,
    }

    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch.keys())}")

    return dispatch[method](portfolio_series, confidence_level, **kwargs)
