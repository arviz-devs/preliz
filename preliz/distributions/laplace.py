# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from ..internal.distribution_helper import all_not_none, eps

from .distributions import Continuous


class Laplace(Continuous):
    r"""
    Laplace distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Laplace
        az.style.use('arviz-doc')
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            Laplace(mu, b).plot_pdf(support=(-10,10))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    """

    def __init__(self, mu=None, b=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, b)

    def _parametrization(self, mu=None, b=None):
        self.mu = mu
        self.b = b
        self.params = (self.mu, self.b)
        self.param_names = ("mu", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, b):
            self._update(mu, b)

    def _update(self, mu, b):
        self.mu = np.float64(mu)
        self.b = np.float64(b)
        self.params = (self.mu, self.b)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu, self.b))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.b)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.b)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.b)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.b)

    def entropy(self):
        return nb_entropy(self.b)

    def median(self):
        return self.mu

    def mean(self):
        return self.mu

    def std(self):
        return self.var() ** 0.5

    def var(self):
        return 2 * self.b**2

    def skewness(self):
        return 0.0

    def kurtosis(self):
        return 3.0

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.laplace(self.mu, self.b, size)

    def _fit_moments(self, mean, sigma):
        b = (sigma / 2) * (2**0.5)
        self._update(mean, b)

    def _fit_mle(self, sample, **kwargs):
        mu, b = nb_fit_mle(sample)
        self._update(mu, b)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, mu, b):
    x = (x - mu) / b
    if x > 0:
        return 1.0 - 0.5 * np.exp(-x)
    return 0.5 * np.exp(x)


@nb.vectorize(nopython=True, cache=True)
def nb_ppf(q, mu, b):
    if q > 0.5:
        q = -np.log(2 * (1 - q))
    else:
        q = np.log(2 * q)
    return q * b + mu


@nb.njit(cache=True)
def nb_logpdf(x, mu, b):
    x = (x - mu) / b
    return np.log(0.5) - np.abs(x) - np.log(b)


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, b):
    return (-nb_logpdf(x, mu, b)).sum()


@nb.njit(cache=True)
def nb_entropy(b):
    return np.log(2) + 1 + np.log(b)


@nb.njit(cache=True)
def nb_fit_mle(sample):
    median = np.median(sample)
    scale = np.sum(np.abs(sample - median)) / len(sample)
    return median, scale
