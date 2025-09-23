"""BetaBinomial probability distribution."""

import numpy as np
from pytensor_distributions import betabinomial as ptd_betabinomial

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class BetaBinomial(Discrete):
    R"""
    Beta-binomial distribution.

    Equivalent to binomial random variable with success probability
    drawn from a beta distribution.

    The pmf of this distribution is

    .. math::

       f(x \mid \alpha, \beta, n) =
           \binom{n}{x}
           \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs


        from preliz import BetaBinomial, style
        style.use('preliz-doc')
        alphas = [0.5, 1, 2.3]
        betas = [0.5, 1, 2]
        n = 10
        for a, b in zip(alphas, betas):
            BetaBinomial(a, b, n).plot_pdf()

    ========  =================================================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n \dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{n \alpha \beta (\alpha+\beta+n)}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
    ========  =================================================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.
    """

    def __init__(self, alpha=None, beta=None, n=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, n)

    def _parametrization(self, alpha=None, beta=None, n=None):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.params = (self.alpha, self.beta, self.n)
        self.param_names = ("alpha", "beta", "n")
        self.params_support = ((eps, np.inf), (eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, beta, n):
            self._update(alpha, beta, n)

    def _update(self, alpha, beta, n):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.n = np.int64(n)
        self.params = (self.alpha, self.beta, self.n)
        self.support = (0, self.n)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.n, self.alpha, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.n, self.alpha, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.n, self.alpha, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.n, self.alpha, self.beta)

    def entropy(self):
        return ptd_entropy(self.n, self.alpha, self.beta)

    def mean(self):
        return ptd_mean(self.n, self.alpha, self.beta)

    def mode(self):
        return ptd_mode(self.n, self.alpha, self.beta)

    def median(self):
        return ptd_median(self.n, self.alpha, self.beta)

    def var(self):
        return ptd_var(self.n, self.alpha, self.beta)

    def std(self):
        return ptd_std(self.n, self.alpha, self.beta)

    def skewness(self):
        return ptd_skewness(self.n, self.alpha, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.n, self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.n, self.alpha, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, n, alpha, beta):
    return ptd_betabinomial.pdf(x, n, alpha, beta)


@pytensor_jit
def ptd_cdf(x, n, alpha, beta):
    return ptd_betabinomial.cdf(x, n, alpha, beta)


@pytensor_jit
def ptd_ppf(q, n, alpha, beta):
    return ptd_betabinomial.ppf(q, n, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, n, alpha, beta):
    return ptd_betabinomial.logpdf(x, n, alpha, beta)


@pytensor_jit
def ptd_entropy(n, alpha, beta):
    return ptd_betabinomial.entropy(n, alpha, beta)


@pytensor_jit
def ptd_mean(n, alpha, beta):
    return ptd_betabinomial.mean(n, alpha, beta)


@pytensor_jit
def ptd_mode(n, alpha, beta):
    return ptd_betabinomial.mode(n, alpha, beta)


@pytensor_jit
def ptd_median(n, alpha, beta):
    return ptd_betabinomial.median(n, alpha, beta)


@pytensor_jit
def ptd_var(n, alpha, beta):
    return ptd_betabinomial.var(n, alpha, beta)


@pytensor_jit
def ptd_std(n, alpha, beta):
    return ptd_betabinomial.std(n, alpha, beta)


@pytensor_jit
def ptd_skewness(n, alpha, beta):
    return ptd_betabinomial.skewness(n, alpha, beta)


@pytensor_jit
def ptd_kurtosis(n, alpha, beta):
    return ptd_betabinomial.kurtosis(n, alpha, beta)


@pytensor_rng_jit
def ptd_rvs(n, alpha, beta, size, rng):
    return ptd_betabinomial.rvs(n, alpha, beta, size=size, random_state=rng)
