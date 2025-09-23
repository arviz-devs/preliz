import numpy as np
from pytensor_distributions import hypergeometric as ptd_hypergeometric

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml

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


        from preliz import HyperGeometric, style
        style.use('preliz-doc')
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
        return ptd_pdf(x, self.N, self.k, self.n)

    def cdf(self, x):
        return ptd_cdf(x, self.N, self.k, self.n)

    def ppf(self, q):
        return ptd_ppf(q, self.N, self.k, self.n)

    def logpdf(self, x):
        return ptd_logpdf(x, self.N, self.k, self.n)

    def entropy(self):
        return ptd_entropy(self.N, self.k, self.n)

    def mean(self):
        return ptd_mean(self.N, self.k, self.n)

    def median(self):
        return ptd_median(self.N, self.k, self.n)

    def var(self):
        return ptd_var(self.N, self.k, self.n)

    def std(self):
        return ptd_std(self.N, self.k, self.n)

    def skewness(self):
        return ptd_skewness(self.N, self.k, self.n)

    def kurtosis(self):
        return ptd_kurtosis(self.N, self.k, self.n)

    def mode(self):
        return ptd_mode(self.N, self.k, self.n)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.N, self.k, self.n, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, N, k, n):
    return ptd_hypergeometric.pdf(x, N, k, n)


@pytensor_jit
def ptd_cdf(x, N, k, n):
    return ptd_hypergeometric.cdf(x, N, k, n)


@pytensor_jit
def ptd_ppf(q, N, k, n):
    return ptd_hypergeometric.ppf(q, N, k, n)


@pytensor_jit
def ptd_logpdf(x, N, k, n):
    return ptd_hypergeometric.logpdf(x, N, k, n)


@pytensor_jit
def ptd_entropy(N, k, n):
    return ptd_hypergeometric.entropy(N, k, n)


@pytensor_jit
def ptd_mean(N, k, n):
    return ptd_hypergeometric.mean(N, k, n)


@pytensor_jit
def ptd_mode(N, k, n):
    return ptd_hypergeometric.mode(N, k, n)


@pytensor_jit
def ptd_median(N, k, n):
    return ptd_hypergeometric.median(N, k, n)


@pytensor_jit
def ptd_var(N, k, n):
    return ptd_hypergeometric.var(N, k, n)


@pytensor_jit
def ptd_std(N, k, n):
    return ptd_hypergeometric.std(N, k, n)


@pytensor_jit
def ptd_skewness(N, k, n):
    return ptd_hypergeometric.skewness(N, k, n)


@pytensor_jit
def ptd_kurtosis(N, k, n):
    return ptd_hypergeometric.kurtosis(N, k, n)


@pytensor_rng_jit
def ptd_rvs(N, k, n, size, rng):
    return ptd_hypergeometric.rvs(N, k, n, size=size, random_state=rng)
