import numba as nb
import numpy as np
from scipy.special import erf, erfinv, zeta

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import erf, erfinv, ppf_bounds_cont  # noqa: F811


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
        x = np.asarray(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return nb_logpdf(x, self.mu, self.sigma)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapezoid(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return self.mu + self.sigma * (np.euler_gamma + np.log(2))

    def mode(self):
        return self.mu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.sigma**2 * (np.pi**2) / 2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 28 * np.sqrt(2) * zeta(3) / np.pi**3

    def kurtosis(self):
        return 4

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.random(size))

    def _fit_moments(self, mean, sigma):
        sigma = sigma / np.pi * 2**0.5
        mu = mean - sigma * (np.euler_gamma + np.log(2))
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    return 1 - erf(np.exp(-z_val / 2) * (2**-0.5))


@nb.njit(cache=True)
def nb_ppf(q, mu, sigma):
    x_val = sigma * -np.log(2.0 * erfinv(1 - q) ** 2) + mu
    return ppf_bounds_cont(x_val, q, -np.inf, np.inf)


@nb.njit(cache=True)
def nb_entropy(sigma):
    return 0.5 * (np.log(2 * np.pi * np.e * sigma**2))


@nb.njit(cache=True)
def nb_logpdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    return -(1 / 2) * (z_val + np.exp(-z_val)) - np.log(sigma) - (1 / 2) * np.log(2 * np.pi)


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma):
    return -(nb_logpdf(x, mu, sigma)).sum()
