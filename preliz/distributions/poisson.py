import numpy as np
from pytensor_distributions import poisson as ptd_poisson

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import eps, pytensor_jit, pytensor_rng_jit


class Poisson(Discrete):
    R"""
    Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs


        from preliz import Poisson, style
        style.use('preliz-doc')
        for mu in [0.5, 3, 8]:
            Poisson(mu).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """

    def __init__(self, mu=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu)

    def _parametrization(self, mu=None):
        self.mu = mu
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.params_support = ((eps, np.inf),)
        if mu is not None:
            self._update(mu)

    def _update(self, mu):
        self.mu = np.float64(mu)
        self.params = (self.mu,)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu)

    def cdf(self, x):
        return ptd_cdf(x, self.mu)

    def ppf(self, q):
        return ptd_ppf(q, self.mu)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu)

    def entropy(self):
        return ptd_entropy(self.mu)

    def mean(self):
        return ptd_mean(self.mu)

    def mode(self):
        return ptd_mode(self.mu)

    def median(self):
        return ptd_median(self.mu)

    def var(self):
        return ptd_var(self.mu)

    def std(self):
        return ptd_std(self.mu)

    def skewness(self):
        return ptd_skewness(self.mu)

    def kurtosis(self):
        return ptd_kurtosis(self.mu)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma=None):
        self._update(mean)

    def _fit_mle(self, sample):
        self._update(np.mean(sample))


@pytensor_jit
def ptd_pdf(x, mu):
    return ptd_poisson.pdf(x, mu)


@pytensor_jit
def ptd_cdf(x, mu):
    return ptd_poisson.cdf(x, mu)


@pytensor_jit
def ptd_ppf(q, mu):
    return ptd_poisson.ppf(q, mu)


@pytensor_jit
def ptd_logpdf(x, mu):
    return ptd_poisson.logpdf(x, mu)


@pytensor_jit
def ptd_entropy(mu):
    return ptd_poisson.entropy(mu)


@pytensor_jit
def ptd_mean(mu):
    return ptd_poisson.mean(mu)


@pytensor_jit
def ptd_mode(mu):
    return ptd_poisson.mode(mu)


@pytensor_jit
def ptd_median(mu):
    return ptd_poisson.median(mu)


@pytensor_jit
def ptd_var(mu):
    return ptd_poisson.var(mu)


@pytensor_jit
def ptd_std(mu):
    return ptd_poisson.std(mu)


@pytensor_jit
def ptd_skewness(mu):
    return ptd_poisson.skewness(mu)


@pytensor_jit
def ptd_kurtosis(mu):
    return ptd_poisson.kurtosis(mu)


@pytensor_rng_jit
def ptd_rvs(mu, size, rng):
    return ptd_poisson.rvs(mu, size=size, random_state=rng)
