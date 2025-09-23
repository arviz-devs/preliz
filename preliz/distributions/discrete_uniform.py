import numpy as np
from pytensor_distributions import discreteuniform as ptd_discreteuniform

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, pytensor_jit, pytensor_rng_jit


class DiscreteUniform(Discrete):
    R"""
    Discrete Uniform distribution.

    The pmf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower+1}

    .. plot::
        :context: close-figs


        from preliz import DiscreteUniform, style
        style.use('preliz-doc')
        ls = [1, -2]
        us = [6, 2]
        for l, u in zip(ls, us):
            ax = DiscreteUniform(l, u).plot_pdf()
            ax.set_ylim(0, 0.25)

    ========  ===============================================
    Support   :math:`x \in {lower, lower + 1, \ldots, upper}`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower + 1)^2 - 1}{12}`
    ========  ===============================================

    Parameters
    ----------
    lower: int
        Lower limit.
    upper: int
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
        self.lower = np.floor(lower)
        self.upper = np.ceil(upper)
        self._n = self.upper - self.lower + 1
        self.params = (self.lower, self.upper)
        self.support = (self.lower, self.upper)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.lower, self.upper)

    def cdf(self, x):
        return ptd_cdf(x, self.lower, self.upper)

    def ppf(self, q):
        return ptd_ppf(q, self.lower, self.upper)

    def logpdf(self, x):
        return ptd_logpdf(x, self.lower, self.upper)

    def entropy(self):
        return ptd_entropy(self.lower, self.upper)

    def mean(self):
        return ptd_mean(self.lower, self.upper)

    def mode(self):
        return ptd_mode(self.lower, self.upper)

    def median(self):
        return ptd_median(self.lower, self.upper)

    def var(self):
        return ptd_var(self.lower, self.upper)

    def std(self):
        return ptd_std(self.lower, self.upper)

    def skewness(self):
        return ptd_skewness(self.lower, self.upper)

    def kurtosis(self):
        return ptd_kurtosis(self.lower, self.upper)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.lower, self.upper, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        spr = (12 * sigma**2 + 1) ** 0.5
        lower = 0.5 * (2 * mean - spr + 1)
        upper = 0.5 * (2 * mean + spr - 1)
        self._update(lower, upper)

    def _fit_mle(self, sample):
        lower = np.min(sample)
        upper = np.max(sample)
        self._update(lower, upper)


@pytensor_jit
def ptd_pdf(x, lower, upper):
    return ptd_discreteuniform.pdf(x, lower, upper)


@pytensor_jit
def ptd_cdf(x, lower, upper):
    return ptd_discreteuniform.cdf(x, lower, upper)


@pytensor_jit
def ptd_ppf(q, lower, upper):
    return ptd_discreteuniform.ppf(q, lower, upper)


@pytensor_jit
def ptd_logpdf(x, lower, upper):
    return ptd_discreteuniform.logpdf(x, lower, upper)


@pytensor_jit
def ptd_entropy(lower, upper):
    return ptd_discreteuniform.entropy(lower, upper)


@pytensor_jit
def ptd_mean(lower, upper):
    return ptd_discreteuniform.mean(lower, upper)


@pytensor_jit
def ptd_mode(lower, upper):
    return ptd_discreteuniform.mode(lower, upper)


@pytensor_jit
def ptd_median(lower, upper):
    return ptd_discreteuniform.median(lower, upper)


@pytensor_jit
def ptd_var(lower, upper):
    return ptd_discreteuniform.var(lower, upper)


@pytensor_jit
def ptd_std(lower, upper):
    return ptd_discreteuniform.std(lower, upper)


@pytensor_jit
def ptd_skewness(lower, upper):
    return ptd_discreteuniform.skewness(lower, upper)


@pytensor_jit
def ptd_kurtosis(lower, upper):
    return ptd_discreteuniform.kurtosis(lower, upper)


@pytensor_rng_jit
def ptd_rvs(lower, upper, size, rng):
    return ptd_discreteuniform.rvs(lower, upper, size=size, random_state=rng)
