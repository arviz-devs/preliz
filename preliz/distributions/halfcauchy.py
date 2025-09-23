import numpy as np
from pytensor_distributions import halfcauchy as ptd_halfcauchy

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class HalfCauchy(Continuous):
    r"""
    HalfCauchy Distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \beta) =
            \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::
        :context: close-figs


        from preliz import HalfCauchy, style
        style.use('preliz-doc')
        for beta in [.5, 1., 2.]:
            HalfCauchy(beta).plot_pdf(support=(0,5))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      undefined
    Variance  undefined
    ========  ==========================================

    Parameters
    ----------
    beta : float
        Scale parameter :math:`\beta` (``beta`` > 0)
    """

    def __init__(self, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(beta)

    def _parametrization(self, beta=None):
        self.beta = beta
        self.params = (self.beta,)
        self.param_names = ("beta",)
        self.params_support = ((eps, np.inf),)
        if self.beta is not None:
            self._update(self.beta)

    def _update(self, beta):
        self.beta = np.float64(beta)
        self.params = (self.beta,)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.beta)

    def entropy(self):
        return ptd_entropy(self.beta)

    def mean(self):
        return ptd_mean(self.beta)

    def mode(self):
        return ptd_mode(self.beta)

    def median(self):
        return ptd_median(self.beta)

    def var(self):
        return ptd_var(self.beta)

    def std(self):
        return ptd_std(self.beta)

    def skewness(self):
        return ptd_skewness(self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        self._update(sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, beta):
    return ptd_halfcauchy.pdf(x, beta)


@pytensor_jit
def ptd_cdf(x, beta):
    return ptd_halfcauchy.cdf(x, beta)


@pytensor_jit
def ptd_ppf(q, beta):
    return ptd_halfcauchy.ppf(q, beta)


@pytensor_jit
def ptd_logpdf(x, beta):
    return ptd_halfcauchy.logpdf(x, beta)


@pytensor_jit
def ptd_entropy(beta):
    return ptd_halfcauchy.entropy(beta)


@pytensor_jit
def ptd_mean(beta):
    return ptd_halfcauchy.mean(beta)


@pytensor_jit
def ptd_mode(beta):
    return ptd_halfcauchy.mode(beta)


@pytensor_jit
def ptd_median(beta):
    return ptd_halfcauchy.median(beta)


@pytensor_jit
def ptd_var(beta):
    return ptd_halfcauchy.var(beta)


@pytensor_jit
def ptd_std(beta):
    return ptd_halfcauchy.std(beta)


@pytensor_jit
def ptd_skewness(beta):
    return ptd_halfcauchy.skewness(beta)


@pytensor_jit
def ptd_kurtosis(beta):
    return ptd_halfcauchy.kurtosis(beta)


@pytensor_rng_jit
def ptd_rvs(beta, size, rng):
    return ptd_halfcauchy.rvs(beta, size=size, random_state=rng)
