import numpy as np
from pytensor_distributions import gumbel as ptd_gumbel

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Gumbel(Continuous):
    r"""
    Gumbel distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}

    .. plot::
        :context: close-figs


        from preliz import Gumbel, style
        style.use('preliz-doc')
        mus = [0., 4., -1.]
        betas = [1., 2., 4.]
        for mu, beta in zip(mus, betas):
            Gumbel(mu, beta).plot_pdf(support=(-10,20))

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \beta\gamma`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, mu=None, beta=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, beta)

    def _parametrization(self, mu=None, beta=None):
        self.mu = mu
        self.beta = beta
        self.params = (self.mu, self.beta)
        self.param_names = ("mu", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(self.mu, self.beta):
            self._update(self.mu, self.beta)

    def _update(self, mu, beta):
        self.mu = np.float64(mu)
        self.beta = np.float64(beta)
        self.params = (self.mu, self.beta)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.beta)

    def entropy(self):
        return ptd_entropy(self.mu, self.beta)

    def mean(self):
        return ptd_mean(self.mu, self.beta)

    def mode(self):
        return ptd_mode(self.mu, self.beta)

    def median(self):
        return ptd_median(self.mu, self.beta)

    def var(self):
        return ptd_var(self.mu, self.beta)

    def std(self):
        return ptd_std(self.mu, self.beta)

    def skewness(self):
        return ptd_skewness(self.mu, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        beta = sigma / np.pi * 6**0.5
        mu = mean - beta * np.euler_gamma
        self._update(mu, beta)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, beta):
    return ptd_gumbel.pdf(x, mu, beta)


@pytensor_jit
def ptd_cdf(x, mu, beta):
    return ptd_gumbel.cdf(x, mu, beta)


@pytensor_jit
def ptd_ppf(q, mu, beta):
    return ptd_gumbel.ppf(q, mu, beta)


@pytensor_jit
def ptd_logpdf(x, mu, beta):
    return ptd_gumbel.logpdf(x, mu, beta)


@pytensor_jit
def ptd_entropy(mu, beta):
    return ptd_gumbel.entropy(mu, beta)


@pytensor_jit
def ptd_mean(mu, beta):
    return ptd_gumbel.mean(mu, beta)


@pytensor_jit
def ptd_mode(mu, beta):
    return ptd_gumbel.mode(mu, beta)


@pytensor_jit
def ptd_median(mu, beta):
    return ptd_gumbel.median(mu, beta)


@pytensor_jit
def ptd_var(mu, beta):
    return ptd_gumbel.var(mu, beta)


@pytensor_jit
def ptd_std(mu, beta):
    return ptd_gumbel.std(mu, beta)


@pytensor_jit
def ptd_skewness(mu, beta):
    return ptd_gumbel.skewness(mu, beta)


@pytensor_jit
def ptd_kurtosis(mu, beta):
    return ptd_gumbel.kurtosis(mu, beta)


@pytensor_rng_jit
def ptd_rvs(mu, beta, size, rng):
    return ptd_gumbel.rvs(mu, beta, size=size, random_state=rng)
