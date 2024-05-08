# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import pdtr, pdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.distribution_helper import eps
from ..internal.special import gammaln, xlogy, cdf_bounds, ppf_bounds_disc


class Poisson(Discrete):
    R"""
    Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Poisson
        az.style.use('arviz-doc')
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        return nb_cdf(x, self.mu, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        return nb_ppf(q, self.mu, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu)

    def entropy(self):
        x = np.arange(0, self.ppf(0.9999) + 1, dtype=int)
        logpdf = self.logpdf(x)
        return -np.sum(np.exp(logpdf) * logpdf)

    def mean(self):
        return self.mu

    def median(self):
        return np.floor(self.mu + 1 / 3 - 0.02 / self.mu)

    def var(self):
        return self.mu

    def std(self):
        return self.mu**0.5

    def skewness(self):
        return 1 / self.mu**0.5

    def kurtosis(self):
        return 1 / self.mu

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.poisson(self.mu, size=size)

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        self._update(nb_fit_mle(sample))


# @nb.jit
# pdtr not supported by numba
def nb_cdf(x, mu, lower, upper):
    prob = pdtr(x, mu)
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
# pdtr not supported by numba
def nb_ppf(q, mu, lower, upper):
    q = np.asarray(q)
    vals = np.ceil(pdtrik(q, mu))
    vals1 = np.maximum(vals - 1, 0)
    temp = pdtr(vals1, mu)
    x_vals = np.where(temp >= q, vals1, vals)
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.njit(cache=True)
def nb_fit_mle(sample):
    return np.mean(sample)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu):
    if x < 0:
        return -np.inf
    else:
        return xlogy(x, mu) - gammaln(x + 1) - mu


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu):
    return -(nb_logpdf(x, mu)).sum()
