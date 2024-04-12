# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from scipy.special import zeta

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.optimization import optimize_ml
from ..internal.special import cdf_bounds, ppf_bounds_cont


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

        import arviz as az
        from preliz import Gumbel
        az.style.use('arviz-doc')
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu, self.beta))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.beta, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.beta, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.beta)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.beta)

    def entropy(self):
        return nb_entropy(self.beta)

    def mean(self):
        return self.mu + self.beta * np.euler_gamma

    def median(self):
        return self.mu - self.beta * np.log(np.log(2))

    def var(self):
        return np.pi**2 / 6 * self.beta**2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 12 * 6**0.5 * zeta(3) / np.pi**3

    def kurtosis(self):
        return 12 / 5

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return nb_ppf(random_state.uniform(size=size), self.mu, self.beta, -np.inf, np.inf)

    def _fit_moments(self, mean, sigma):
        beta = sigma / np.pi * 6**0.5
        mu = mean - beta * np.euler_gamma
        self._update(mu, beta)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, mu, beta, lower, upper):
    prob = np.exp(-np.exp(-(x - mu) / beta))
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, mu, beta, lower, upper):
    x_val = mu - beta * np.log(-np.log(q))
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(beta):
    return np.log(beta) + 1 + np.euler_gamma


@nb.njit(cache=True)
def nb_logpdf(x, mu, beta):
    zval = (x - mu) / beta
    return -(zval + np.exp(-zval) + np.log(beta))


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, beta):
    return -(nb_logpdf(x, mu, beta)).sum()
