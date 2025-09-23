import numpy as np
from pytensor_distributions import pareto as ptd_pareto

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Pareto(Continuous):
    r"""
    Pareto distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::
        :context: close-figs


        from preliz import Pareto, style
        style.use('preliz-doc')
        alphas = [1., 5., 5.]
        ms = [1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            Pareto(alpha, m).plot_pdf(support=(0,4))

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m^2 \alpha}{(\alpha - 1)^2 (\alpha - 2)}` for :math:`\alpha > 2`
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
        return ptd_pdf(x, self.alpha, self.m)

    def cdf(self, x):
        return ptd_cdf(x, self.alpha, self.m)

    def ppf(self, q):
        return ptd_ppf(q, self.alpha, self.m)

    def logpdf(self, x):
        return ptd_logpdf(x, self.alpha, self.m)

    def entropy(self):
        return ptd_entropy(self.alpha, self.m)

    def mean(self):
        return ptd_mean(self.alpha, self.m)

    def mode(self):
        return ptd_mode(self.alpha, self.m)

    def median(self):
        return ptd_median(self.alpha, self.m)

    def var(self):
        return ptd_var(self.alpha, self.m)

    def std(self):
        return ptd_std(self.alpha, self.m)

    def skewness(self):
        return ptd_skewness(self.alpha, self.m)

    def kurtosis(self):
        return ptd_kurtosis(self.alpha, self.m)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.m, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        alpha = 1 + (1 + (mean / sigma) ** 2) ** (1 / 2)
        m = (alpha - 1) * mean / alpha
        self._update(alpha, m)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, alpha, m):
    return ptd_pareto.pdf(x, alpha, m)


@pytensor_jit
def ptd_cdf(x, alpha, m):
    return ptd_pareto.cdf(x, alpha, m)


@pytensor_jit
def ptd_ppf(q, alpha, m):
    return ptd_pareto.ppf(q, alpha, m)


@pytensor_jit
def ptd_logpdf(x, alpha, m):
    return ptd_pareto.logpdf(x, alpha, m)


@pytensor_jit
def ptd_entropy(alpha, m):
    return ptd_pareto.entropy(alpha, m)


@pytensor_jit
def ptd_mean(alpha, m):
    return ptd_pareto.mean(alpha, m)


@pytensor_jit
def ptd_mode(alpha, m):
    return ptd_pareto.mode(alpha, m)


@pytensor_jit
def ptd_median(alpha, m):
    return ptd_pareto.median(alpha, m)


@pytensor_jit
def ptd_var(alpha, m):
    return ptd_pareto.var(alpha, m)


@pytensor_jit
def ptd_std(alpha, m):
    return ptd_pareto.std(alpha, m)


@pytensor_jit
def ptd_skewness(alpha, m):
    return ptd_pareto.skewness(alpha, m)


@pytensor_jit
def ptd_kurtosis(alpha, m):
    return ptd_pareto.kurtosis(alpha, m)


@pytensor_rng_jit
def ptd_rvs(alpha, m, size, rng):
    return ptd_pareto.rvs(alpha, m, size=size, random_state=rng)
