import numpy as np
from pytensor_distributions import geometric as ptd_geometric

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import eps, pytensor_jit, pytensor_rng_jit


class Geometric(Discrete):
    R"""
    Geometric distribution.

    The probability that the first success in a sequence of Bernoulli trials
    occurs on the x'th trial.
    The pmf of this distribution is

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    .. plot::
        :context: close-figs


        from preliz import Geometric, style
        style.use('preliz-doc')
        for p in [0.1, 0.25, 0.75]:
            Geometric(p).plot_pdf(support=(1,10))

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{>0}`
    Mean      :math:`\dfrac{1}{p}`
    Variance  :math:`\dfrac{1 - p}{p^2}`
    ========  =============================

    Parameters
    ----------
    p : float
        Probability of success on an individual trial (0 < p <= 1).
    """

    def __init__(self, p=None):
        super().__init__()
        self.support = (1, np.inf)
        self._parametrization(p)

    def _parametrization(self, p=None):
        self.p = p
        self.param_names = "p"
        self.params_support = ((eps, 1),)
        if self.p is not None:
            self._update(self.p)

    def _update(self, p):
        self.p = np.float64(p)
        self.params = (self.p,)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.p)

    def cdf(self, x):
        return ptd_cdf(x, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.p)

    def entropy(self):
        return ptd_entropy(self.p)

    def mean(self):
        return ptd_mean(self.p)

    def mode(self):
        return ptd_mode(self.p)

    def median(self):
        return ptd_median(self.p)

    def var(self):
        return ptd_var(self.p)

    def std(self):
        return ptd_std(self.p)

    def skewness(self):
        return ptd_skewness(self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.p, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        p = 1 / mean
        self._update(p)

    def _fit_mle(self, sample):
        p = 1 / np.mean(sample)
        self._update(p)


@pytensor_jit
def ptd_pdf(x, p):
    return ptd_geometric.pdf(x, p)


@pytensor_jit
def ptd_cdf(x, p):
    return ptd_geometric.cdf(x, p)


@pytensor_jit
def ptd_ppf(q, p):
    return ptd_geometric.ppf(q, p)


@pytensor_jit
def ptd_logpdf(x, p):
    return ptd_geometric.logpdf(x, p)


@pytensor_jit
def ptd_entropy(p):
    return ptd_geometric.entropy(p)


@pytensor_jit
def ptd_mean(p):
    return ptd_geometric.mean(p)


@pytensor_jit
def ptd_mode(p):
    return ptd_geometric.mode(p)


@pytensor_jit
def ptd_median(p):
    return ptd_geometric.median(p)


@pytensor_jit
def ptd_var(p):
    return ptd_geometric.var(p)


@pytensor_jit
def ptd_std(p):
    return ptd_geometric.std(p)


@pytensor_jit
def ptd_skewness(p):
    return ptd_geometric.skewness(p)


@pytensor_jit
def ptd_kurtosis(p):
    return ptd_geometric.kurtosis(p)


@pytensor_rng_jit
def ptd_rvs(p, size, rng):
    return ptd_geometric.rvs(p, size=size, random_state=rng)
