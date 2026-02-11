import numpy as np
from pytensor_distributions import moyal as ptd_moyal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Moyal(Continuous):
    r"""
    Moyal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu,\sigma) =
            \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(z + e^{-z}\right)},

    where

    .. math::

       z = \frac{x-\mu}{\sigma}

    .. plot::
        :context: close-figs


        from preliz import Moyal, style
        style.use('preliz-doc')
        mus = [-1., 0., 4.]
        sigmas = [2., 1., 4.]
        for mu, sigma in zip(mus, sigmas):
            Moyal(mu, sigma).plot_pdf(support=(-10,20))

    ========  ==============================================================
    Support   :math:`x \in (-\infty, \infty)`
    Mean      :math:`\mu + \sigma\left(\gamma + \log 2\right)`, where
              :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^{2}}{2}\sigma^{2}`
    ========  ==============================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, sigma):
            self._update(self.mu, self.sigma)

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.mu, self.sigma)

    def mode(self):
        return ptd_mode(self.mu, self.sigma)

    def median(self):
        return ptd_median(self.mu, self.sigma)

    def var(self):
        return ptd_var(self.mu, self.sigma)

    def std(self):
        return ptd_std(self.mu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        sigma = sigma / np.pi * 2**0.5
        mu = mean - sigma * (np.euler_gamma + np.log(2))
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, sigma):
    return ptd_moyal.pdf(x, mu, sigma)


@pytensor_jit
def ptd_cdf(x, mu, sigma):
    return ptd_moyal.cdf(x, mu, sigma)


@pytensor_jit
def ptd_ppf(q, mu, sigma):
    return ptd_moyal.ppf(q, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, mu, sigma):
    return ptd_moyal.logpdf(x, mu, sigma)


@pytensor_jit
def ptd_entropy(mu, sigma):
    return ptd_moyal.entropy(mu, sigma)


@pytensor_jit
def ptd_mean(mu, sigma):
    return ptd_moyal.mean(mu, sigma)


@pytensor_jit
def ptd_mode(mu, sigma):
    return ptd_moyal.mode(mu, sigma)


@pytensor_jit
def ptd_median(mu, sigma):
    return ptd_moyal.median(mu, sigma)


@pytensor_jit
def ptd_var(mu, sigma):
    return ptd_moyal.var(mu, sigma)


@pytensor_jit
def ptd_std(mu, sigma):
    return ptd_moyal.std(mu, sigma)


@pytensor_jit
def ptd_skewness(mu, sigma):
    return ptd_moyal.skewness(mu, sigma)


@pytensor_jit
def ptd_kurtosis(mu, sigma):
    return ptd_moyal.kurtosis(mu, sigma)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, size, rng):
    return ptd_moyal.rvs(mu, sigma, size=size, random_state=rng)
