# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from ..internal.optimization import optimize_ml
from ..internal.special import ppf_bounds_cont
from ..internal.distribution_helper import all_not_none, eps
from .distributions import Continuous


class Pareto(Continuous):
    r"""
    Pareto distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Pareto
        az.style.use('arviz-doc')
        alphas = [1., 5., 5.]
        ms = [1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            Pareto(alpha, m).plot_pdf(support=(0,4))

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m \alpha}{(\alpha - 1)^2 (\alpha - 2)}` for :math:`\alpha > 2`
    ========  =============================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    m : float
        Scale parameter (m > 0).
    """

    def __init__(self, alpha=None, m=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, m)

    def _parametrization(self, alpha=None, m=None):
        self.alpha = alpha
        self.m = m
        self.params = (self.alpha, self.m)
        self.param_names = ("alpha", "m")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, m):
            self._update(alpha, m)

    def _update(self, alpha, m):
        self.alpha = np.float64(alpha)
        self.m = np.float64(m)
        self.support = (self.m, np.inf)
        self.params = (self.alpha, self.m)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.alpha, self.m))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.alpha, self.m)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.alpha, self.m, 1, np.inf)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.alpha, self.m)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.alpha, self.m)

    def entropy(self):
        return nb_entropy(self.alpha, self.m)

    def mean(self):
        return np.where(self.alpha > 1, self.alpha * self.m / (self.alpha - 1), np.inf)

    def median(self):
        return self.m * 2 ** (1 / self.alpha)

    def var(self):
        return np.where(
            self.alpha > 2,
            self.m**2 * self.alpha / ((self.alpha - 1) ** 2 * (self.alpha - 2)),
            np.inf,
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return np.where(
            self.alpha > 3,
            2 * (1 + self.alpha) / (self.alpha - 3) * (1 - 2 / self.alpha) ** 0.5,
            np.nan,
        )

    def kurtosis(self):
        return np.where(
            self.alpha > 4,
            6
            * (self.alpha**3 + self.alpha**2 - 6 * self.alpha - 2)
            / (self.alpha * (self.alpha - 3) * (self.alpha - 4)),
            np.nan,
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(0, 1, size)
        return nb_rvs(random_samples, self.alpha, self.m)

    def _fit_moments(self, mean, sigma):
        alpha = 1 + (1 + (mean / sigma) ** 2) ** (1 / 2)
        m = (alpha - 1) * mean / alpha
        self._update(alpha, m)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, alpha, m):
    if x < m:
        return 0
    return 1 - (m / x) ** alpha


@nb.njit(cache=True)
def nb_ppf(q, alpha, m, lower, upper):
    return ppf_bounds_cont(m * (1 - q) ** (-1 / alpha), q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(alpha, m):
    return np.log((m / alpha) * np.exp(1 + 1 / alpha))


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, alpha, m):
    if x < m:
        return -np.inf
    return np.log(alpha) + alpha * np.log(m) - (alpha + 1) * np.log(x)


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, m):
    return -(nb_logpdf(x, alpha, m)).sum()


@nb.njit(cache=True)
def nb_rvs(random_samples, alpha, m):
    return m / (1 - random_samples) ** (1 / alpha)
