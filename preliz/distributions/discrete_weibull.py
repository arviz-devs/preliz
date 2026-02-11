import numpy as np
from pytensor_distributions import discreteweibull as ptd_discreteweibull

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class DiscreteWeibull(Discrete):
    R"""
    Discrete Weibull distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x+1)^{\beta}}

    .. plot::
        :context: close-figs


        from preliz import DiscreteWeibull, style
        style.use('preliz-doc')
        qs = [0.1, 0.9, 0.9]
        betas = [0.5, 0.5, 2]
        for q, b in zip(qs, betas):
            DiscreteWeibull(q, b).plot_pdf(support=(0,10))

    ========  ===============================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu = \sum_{x = 1}^{\infty} q^{x^{\beta}}`
    Variance  :math:`2 \sum_{x = 1}^{\infty} x q^{x^{\beta}} - \mu - \mu^2`
    ========  ===============================================

    Parameters
    ----------
    q: float
        Shape parameter (0 < q < 1).
    beta: float
        Shape parameter (beta > 0).
    """

    def __init__(self, q=None, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(q, beta)

    def _parametrization(self, q=None, beta=None):
        self.q = q
        self.beta = beta
        self.params = (self.q, self.beta)
        self.param_names = ("q", "beta")
        self.params_support = ((eps, 1 - eps), (eps, np.inf))
        if all_not_none(q, beta):
            self._update(q, beta)

    def _update(self, q, beta):
        self.q = np.float64(q)
        self.beta = np.float64(beta)
        self.params = (self.q, self.beta)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.q, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.q, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.q, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.q, self.beta)

    def entropy(self):
        return ptd_entropy(self.q, self.beta)

    def mean(self):
        return ptd_mean(self.q, self.beta)

    def median(self):
        return ptd_median(self.q, self.beta)

    def var(self):
        return ptd_var(self.q, self.beta)

    def std(self):
        return ptd_std(self.q, self.beta)

    def skewness(self):
        return ptd_skewness(self.q, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.q, self.beta)

    def mode(self):
        return ptd_mode(self.q, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.q, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, q, beta):
    return ptd_discreteweibull.pdf(x, q, beta)


@pytensor_jit
def ptd_cdf(x, q, beta):
    return ptd_discreteweibull.cdf(x, q, beta)


@pytensor_jit
def ptd_ppf(p, q, beta):
    return ptd_discreteweibull.ppf(p, q, beta)


@pytensor_jit
def ptd_logpdf(x, q, beta):
    return ptd_discreteweibull.logpdf(x, q, beta)


@pytensor_jit
def ptd_entropy(q, beta):
    return ptd_discreteweibull.entropy(q, beta)


@pytensor_jit
def ptd_mean(q, beta):
    return ptd_discreteweibull.mean(q, beta)


@pytensor_jit
def ptd_mode(q, beta):
    return ptd_discreteweibull.mode(q, beta)


@pytensor_jit
def ptd_median(q, beta):
    return ptd_discreteweibull.median(q, beta)


@pytensor_jit
def ptd_var(q, beta):
    return ptd_discreteweibull.var(q, beta)


@pytensor_jit
def ptd_std(q, beta):
    return ptd_discreteweibull.std(q, beta)


@pytensor_jit
def ptd_skewness(q, beta):
    return ptd_discreteweibull.skewness(q, beta)


@pytensor_jit
def ptd_kurtosis(q, beta):
    return ptd_discreteweibull.kurtosis(q, beta)


@pytensor_rng_jit
def ptd_rvs(q, beta, size, rng):
    return ptd_discreteweibull.rvs(q, beta, size=size, random_state=rng)
