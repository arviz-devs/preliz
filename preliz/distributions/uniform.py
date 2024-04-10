# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import all_not_none
from ..internal.special import cdf_bounds, ppf_bounds_cont


class Uniform(Continuous):
    r"""
    Uniform distribution.

    The pdf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Uniform
        az.style.use('arviz-doc')
        ls = [1, -2]
        us = [6, 2]
        for l, u in zip(ls, us):
            ax = Uniform(l, u).plot_pdf()
        ax.set_ylim(0, 0.3)

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  =====================================

    Parameters
    ----------
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self._parametrization(lower, upper)

    def _parametrization(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self.params = (self.lower, self.upper)
        self.param_names = ("lower", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf))
        if lower is None:
            self.lower = -np.inf
        if upper is None:
            self.upper = np.inf
        self.support = (self.lower, self.upper)
        if all_not_none(lower, upper):
            self._update(lower, upper)
        else:
            self.lower = lower
            self.upper = upper

    def _update(self, lower, upper):
        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
        self.params = (self.lower, self.upper)
        self.support = (self.lower, self.upper)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_pdf(x, self.lower, self.upper)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.lower, self.upper)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.lower, self.upper)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.lower, self.upper)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.lower, self.upper)

    def entropy(self):
        return nb_entropy(self.lower, self.upper)

    def mean(self):
        return (self.upper + self.lower) / 2

    def median(self):
        return (self.upper + self.lower) / 2

    def var(self):
        return (self.upper - self.lower) ** 2 / 12

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 0

    def kurtosis(self):
        return -6 / 5

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.uniform(self.lower, self.upper, size)

    def _fit_moments(self, mean, sigma):
        lower = mean - 1.73205 * sigma
        upper = mean + 1.73205 * sigma
        self._update(lower, upper)

    def _fit_mle(self, sample):
        lower = np.min(sample)
        upper = np.max(sample)
        self._update(lower, upper)


@nb.njit(cache=True)
def nb_cdf(x, lower, upper):
    prob = (x - lower) / (upper - lower)
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, lower, upper):
    x_vals = lower + q * (upper - lower)
    return ppf_bounds_cont(x_vals, q, lower, upper)


@nb.vectorize(nopython=True, cache=True)
def nb_pdf(x, lower, upper):
    if lower <= x <= upper:
        return 1 / (upper - lower)
    else:
        return 0


@nb.njit(cache=True)
def nb_entropy(lower, upper):
    return np.log(upper - lower)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, lower, upper):
    if lower <= x <= upper:
        return -np.log(upper - lower)
    else:
        return -np.inf


@nb.njit(cache=True)
def nb_neg_logpdf(x, lower, upper):
    return -(nb_logpdf(x, lower, upper)).sum()
