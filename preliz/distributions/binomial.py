import numba as nb
import numpy as np
from pytensor_distributions import binomial as ptd_binomial

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma
from preliz.internal.special import mean_and_std


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


        from preliz import Binomial, style
        style.use('preliz-doc')
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
        optimize_mean_sigma(self, mean, sigma, params)

    def _fit_mle(self, sample):
        self._update(*nb_fit_mle(sample))

    def pdf(self, x):
        return ptd_pdf(x, self.n, self.p)

    def cdf(self, x):
        return ptd_cdf(x, self.n, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.n, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.n, self.p)

    def entropy(self):
        return ptd_entropy(self.n, self.p)

    def mean(self):
        return ptd_mean(self.n, self.p)

    def mode(self):
        return ptd_mode(self.n, self.p)

    def median(self):
        return ptd_median(self.n, self.p)

    def var(self):
        return ptd_var(self.n, self.p)

    def std(self):
        return ptd_std(self.n, self.p)

    def skewness(self):
        return ptd_skewness(self.n, self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.n, self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.n, self.p, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p
        n = mean + sigma * 2
        p = mean / n
        params = n, p
        return optimize_mean_sigma(self, mean, sigma, params)

    def _fit_mle(self, sample):
        self._update(*nb_fit_mle(sample))


@pytensor_jit
def ptd_pdf(x, n, p):
    return ptd_binomial.pdf(x, n, p)


@pytensor_jit
def ptd_cdf(x, n, p):
    return ptd_binomial.cdf(x, n, p)


@pytensor_jit
def ptd_ppf(q, n, p):
    return ptd_binomial.ppf(q, n, p)


@pytensor_jit
def ptd_logpdf(x, n, p):
    return ptd_binomial.logpdf(x, n, p)


@pytensor_jit
def ptd_entropy(n, p):
    return ptd_binomial.entropy(n, p)


@pytensor_jit
def ptd_mean(n, p):
    return ptd_binomial.mean(n, p)


@pytensor_jit
def ptd_mode(n, p):
    return ptd_binomial.mode(n, p)


@pytensor_jit
def ptd_median(n, p):
    return ptd_binomial.median(n, p)


@pytensor_jit
def ptd_var(n, p):
    return ptd_binomial.var(n, p)


@pytensor_jit
def ptd_std(n, p):
    return ptd_binomial.std(n, p)


@pytensor_jit
def ptd_skewness(n, p):
    return ptd_binomial.skewness(n, p)


@pytensor_jit
def ptd_kurtosis(n, p):
    return ptd_binomial.kurtosis(n, p)


@pytensor_rng_jit
def ptd_rvs(n, p, size, rng):
    return ptd_binomial.rvs(n, p, size=size, random_state=rng)


@nb.njit(cache=True)
def nb_fit_mle(sample):
    # see https://doi.org/10.1016/j.jspi.2004.02.019 for details
    x_bar, x_std = mean_and_std(sample)
    x_max = np.max(sample)
    n = np.ceil(x_max ** (1.5) * x_std / (x_bar**0.5 * (x_max - x_bar) ** 0.5))
    p = x_bar / n
    return n, p
