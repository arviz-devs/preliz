# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import bdtr, bdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.optimization import optimize_moments
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import cdf_bounds, ppf_bounds_disc, gammaln, mean_and_std, xlogy, xlog1py


class Binomial(Discrete):
    R"""
    Binomial distribution.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.

    The pmf of this distribution is

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Binomial
        az.style.use('arviz-doc')
        ns = [5, 10, 10]
        ps = [0.5, 0.5, 0.7]
        for n, p in zip(ns, ps):
            Binomial(n, p).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n p`
    Variance  :math:`n p (1 - p)`
    ========  ==========================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n=None, p=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(n, p)

    def _parametrization(self, n=None, p=None):
        self.n = n
        self.p = p
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.params_support = ((eps, np.inf), (eps, 1 - eps))
        if all_not_none(n, p):
            self._update(n, p)

    def _update(self, n, p):
        self.n = np.int64(n)
        self.p = np.float64(p)
        self._q = 1 - self.p
        self.params = (self.n, self.p)
        self.support = (0, self.n)
        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p
        n = mean + sigma * 2
        p = mean / n
        params = n, p
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        self._update(*nb_fit_mle(sample))

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
        return nb_cdf(x, self.n, self.p, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        return nb_ppf(q, self.n, self.p, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.n, self.p)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.n, self.p)

    def entropy(self):
        return nb_entropy(self.n, self.p)

    def mean(self):
        return self.n * self.p

    def median(self):
        return np.ceil(self.n * self.p)

    def var(self):
        return self.n * self.p * self._q

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (self._q - self.p) / self.std()

    def kurtosis(self):
        return (1 - 6 * self.p * self._q) / (self.n * self.p * self._q)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.binomial(self.n, self.p, size=size)

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p
        n = mean + sigma * 2
        p = mean / n
        params = n, p
        return optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        self._update(*nb_fit_mle(sample))


# @nb.jit
# bdtr not supported by numba
def nb_cdf(x, n, p, lower, upper):
    x = np.asarray(x)
    prob = np.asarray(bdtr(x, n, p))
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
def nb_ppf(q, n, p, lower, upper):
    q = np.asarray(q)
    x_vals = np.ceil(bdtrik(q, n, p))
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(n, p):
    return 0.5 * np.log(2 * np.pi * np.e * n * p * (1 - p))


@nb.njit(cache=True)
def nb_fit_mle(sample):
    # see https://doi.org/10.1016/j.jspi.2004.02.019 for details
    x_bar, x_std = mean_and_std(sample)
    x_max = np.max(sample)
    n = np.ceil(x_max ** (1.5) * x_std / (x_bar**0.5 * (x_max - x_bar) ** 0.5))
    p = x_bar / n
    return n, p


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, n, p):
    if x < 0:
        return -np.inf
    elif x > n:
        return -np.inf
    else:
        return (
            gammaln(n + 1)
            - (gammaln(x + 1) + gammaln(n - x + 1))
            + xlogy(x, p)
            + xlog1py(n - x, -p)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, n, p):
    return -(nb_logpdf(x, n, p)).sum()
