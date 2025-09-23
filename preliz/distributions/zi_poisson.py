import numpy as np
from pytensor_distributions import zi_poisson as ptd_zipoisson

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class ZeroInflatedPoisson(Discrete):
    R"""
    Zero-inflated Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
            \psi \frac{e^{-\mu}\mu^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::
        :context: close-figs

        from preliz import ZeroInflatedPoisson, style
        style.use('preliz-doc')
        psis = [0.7, 0.4]
        mus = [8, 4]
        for psi, mu in zip(psis, mus):
            ZeroInflatedPoisson(psi, mu).plot_pdf()

    ========  ================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi \mu`
    Variance  :math:`\psi \mu (1+(1-\psi) \mu`
    ========  ================================

    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    mu : float
        Expected number of occurrences during the given interval
        (mu >= 0).
    """

    def __init__(self, psi=None, mu=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(psi, mu)

    def _parametrization(self, psi=None, mu=None):
        self.psi = psi
        self.mu = mu
        self.params = (self.psi, self.mu)
        self.param_names = ("psi", "mu")
        self.params_support = ((eps, 1 - eps), (eps, np.inf))
        if all_not_none(psi, mu):
            self._update(psi, mu)

    def _update(self, psi, mu):
        self.psi = np.float64(psi)
        self.mu = np.float64(mu)
        self.params = (self.psi, self.mu)
        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)

    def pdf(self, x):
        x = np.asarray(x)
        result = ptd_pdf(x, self.psi, self.mu)
        # Return 0 for negative values and NaN for infinity, consistent with scipy.stats.poisson
        result = np.where(x < 0, 0, result)
        result = np.where(~np.isfinite(x), np.nan, result)
        return result

    def cdf(self, x):
        return ptd_cdf(x, self.psi, self.mu)

    def ppf(self, q):
        return ptd_ppf(q, self.psi, self.mu)

    def logpdf(self, x):
        return ptd_logpdf(x, self.psi, self.mu)

    def entropy(self):
        return ptd_entropy(self.psi, self.mu)

    def mean(self):
        return ptd_mean(self.psi, self.mu)

    def mode(self):
        return ptd_mode(self.psi, self.mu)

    def median(self):
        return ptd_median(self.psi, self.mu)

    def var(self):
        return ptd_var(self.psi, self.mu)

    def std(self):
        return ptd_std(self.psi, self.mu)

    def skewness(self):
        return ptd_skewness(self.psi, self.mu)

    def kurtosis(self):
        return ptd_kurtosis(self.psi, self.mu)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.psi, self.mu, size=size, rng=random_state)


@pytensor_jit
def ptd_pdf(x, psi, mu):
    return ptd_zipoisson.pdf(x, psi, mu)


@pytensor_jit
def ptd_cdf(x, psi, mu):
    return ptd_zipoisson.cdf(x, psi, mu)


@pytensor_jit
def ptd_ppf(q, psi, mu):
    return ptd_zipoisson.ppf(q, psi, mu)


@pytensor_jit
def ptd_logpdf(x, psi, mu):
    return ptd_zipoisson.logpdf(x, psi, mu)


@pytensor_jit
def ptd_entropy(psi, mu):
    return ptd_zipoisson.entropy(psi, mu)


@pytensor_jit
def ptd_mean(psi, mu):
    return ptd_zipoisson.mean(psi, mu)


@pytensor_jit
def ptd_mode(psi, mu):
    return ptd_zipoisson.mode(psi, mu)


@pytensor_jit
def ptd_median(psi, mu):
    return ptd_zipoisson.median(psi, mu)


@pytensor_jit
def ptd_var(psi, mu):
    return ptd_zipoisson.var(psi, mu)


@pytensor_jit
def ptd_std(psi, mu):
    return ptd_zipoisson.std(psi, mu)


@pytensor_jit
def ptd_skewness(psi, mu):
    return ptd_zipoisson.skewness(psi, mu)


@pytensor_jit
def ptd_kurtosis(psi, mu):
    return ptd_zipoisson.kurtosis(psi, mu)


@pytensor_rng_jit
def ptd_rvs(psi, mu, size, rng):
    return ptd_zipoisson.rvs(psi, mu, size=size, random_state=rng)
