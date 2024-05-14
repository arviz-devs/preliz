# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
# pylint: disable=invalid-name
import numba as nb
import numpy as np
from .distributions import Discrete

from ..internal.special import betaln, cdf_bounds
from ..internal.optimization import optimize_ml, optimize_moments, find_ppf
from ..internal.distribution_helper import all_not_none

eps = np.finfo(float).eps


class HyperGeometric(Discrete):
    R"""
    Discrete hypergeometric distribution.

    The probability of :math:`x` successes in a sequence of :math:`n` bernoulli
    trials taken without replacement from a population of :math:`N` objects,
    containing :math:`k` good (or successful or Type I) objects.
    The pmf of this distribution is

    .. math:: f(x \mid N, n, k) = \frac{\binom{k}{x}\binom{N-k}{n-x}}{\binom{N}{n}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HyperGeometric
        az.style.use('arviz-doc')
        N = 50
        k = 10
        for n in [20, 25]:
            HyperGeometric(N, k, n).plot_pdf(support=(1,15))

    ========  =============================
    Support   :math:`x \in \left[\max(0, n - N + k), \min(k, n)\right]`
    Mean      :math:`\dfrac{nk}{N}`
    Variance  :math:`\dfrac{(N-n)nk(N-k)}{(N-1)N^2}`
    ========  =============================

    Parameters
    ----------
    N : int
        Total size of the population (N > 0)
    k : int
        Number of successful individuals in the population (0 <= k <= N)
    n : int
        Number of samples drawn from the population (0 <= n <= N)
    """

    def __init__(self, N=None, k=None, n=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(N, k, n)

    def _parametrization(self, N=None, k=None, n=None):
        self.N = N
        self.k = k
        self.n = n
        self.param_names = ("N", "k", "n")
        self.params_support = ((eps, np.inf), (eps, self.N), (eps, self.N))
        if all_not_none(self.N, self.k, self.n):
            self._update(N, k, n)

    def _update(self, N, k, n):
        self.N = np.int64(N)
        self.k = np.int64(k)
        self.n = np.int64(n)
        self.params = (self.N, self.k, self.n)
        self.support = (max(0, n - N + k), min(k, n))
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        if isinstance(x, (np.ndarray, list, tuple)):
            cdf_values = np.zeros_like(x, dtype=float)
            for i, val in enumerate(x):
                x_vals = np.arange(self.support[0], val + 1)
                cdf_values[i] = np.sum(self.pdf(x_vals))
            return cdf_bounds(cdf_values, x, *self.support)
        else:
            x_vals = np.arange(self.support[0], x + 1)
            return cdf_bounds(np.sum(self.pdf(x_vals)), x, *self.support)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return find_ppf(self, q)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.N, self.k, self.n, *self.support)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.N, self.k, self.n, *self.support)

    def entropy(self):
        x_values = self.xvals("full")
        logpdf = self.logpdf(x_values)
        return -np.sum(np.exp(logpdf) * logpdf)

    def mean(self):
        return self.n * self.k / self.N

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return (
            self.n * self.k / self.N * (self.N - self.k) / self.N * (self.N - self.n) / (self.N - 1)
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        numerator = (self.N - 2 * self.k) * (self.N - 1) ** 0.5 * (self.N - 2 * self.n)
        denominator = (self.n * self.k * (self.N - self.k) * (self.N - self.n)) ** 0.5 * (
            self.N - 2
        )
        return numerator / denominator

    def kurtosis(self):
        return (
            1
            / (
                self.n
                * self.k
                * (self.N - self.k)
                * (self.N - self.n)
                * (self.N - 2)
                * (self.N - 3)
            )
            * (
                (self.N - 1)
                * self.N**2
                * (
                    self.N * (self.N + 1)
                    - 6 * self.k * (self.N - self.k)
                    - 6 * self.n * (self.N - self.n)
                )
                + 6 * self.n * self.k * (self.N - self.k) * (self.N - self.n) * (5 * self.N - 6)
            )
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.hypergeometric(self.k, self.N - self.k, self.n, size=size)

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, N, k, n, lower, upper):
    if x < lower:
        return -np.inf
    if x > upper:
        return -np.inf
    else:
        good = k
        bad = N - k
        tot = good + bad
        result = (
            betaln(good + 1, 1)
            + betaln(bad + 1, 1)
            + betaln(tot - n + 1, n + 1)
            - betaln(x + 1, good - x + 1)
            - betaln(n - x + 1, bad - n + x + 1)
            - betaln(tot + 1, 1)
        )
        return result


@nb.njit(cache=True)
def nb_neg_logpdf(x, N, k, n, lower, upper):
    return -(nb_logpdf(x, N, k, n, lower, upper)).sum()
