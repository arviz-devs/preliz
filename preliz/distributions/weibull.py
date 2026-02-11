import numpy as np
from pytensor_distributions import weibull as ptd_weibull

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import (
    garcia_approximation,
    mean_and_std,
)


class Weibull(Continuous):
    r"""
    Weibull distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::
        :context: close-figs


        from preliz import Weibull, style
        style.use('preliz-doc')
        alphas = [1., 2, 5.]
        betas = [1., 1., 2.]
        for a, b in zip(alphas, betas):
            Weibull(a, b).plot_pdf(support=(0,5))

    ========  ====================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2/\beta^2)`
    ========  ====================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.alpha, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.alpha, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.alpha, self.beta)

    def entropy(self):
        return ptd_entropy(self.alpha, self.beta)

    def mean(self):
        return ptd_mean(self.alpha, self.beta)

    def mode(self):
        return ptd_mode(self.alpha, self.beta)

    def median(self):
        return ptd_median(self.alpha, self.beta)

    def var(self):
        return ptd_var(self.alpha, self.beta)

    def std(self):
        return ptd_std(self.alpha, self.beta)

    def skewness(self):
        return ptd_skewness(self.alpha, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        alpha, beta = garcia_approximation(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, alpha, beta):
    return ptd_weibull.pdf(x, alpha, beta)


@pytensor_jit
def ptd_cdf(x, alpha, beta):
    return ptd_weibull.cdf(x, alpha, beta)


@pytensor_jit
def ptd_ppf(q, alpha, beta):
    return ptd_weibull.ppf(q, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, alpha, beta):
    return ptd_weibull.logpdf(x, alpha, beta)


@pytensor_jit
def ptd_entropy(alpha, beta):
    return ptd_weibull.entropy(alpha, beta)


@pytensor_jit
def ptd_mean(alpha, beta):
    return ptd_weibull.mean(alpha, beta)


@pytensor_jit
def ptd_mode(alpha, beta):
    return ptd_weibull.mode(alpha, beta)


@pytensor_jit
def ptd_median(alpha, beta):
    return ptd_weibull.median(alpha, beta)


@pytensor_jit
def ptd_var(alpha, beta):
    return ptd_weibull.var(alpha, beta)


@pytensor_jit
def ptd_std(alpha, beta):
    return ptd_weibull.std(alpha, beta)


@pytensor_jit
def ptd_skewness(alpha, beta):
    return ptd_weibull.skewness(alpha, beta)


@pytensor_jit
def ptd_kurtosis(alpha, beta):
    return ptd_weibull.kurtosis(alpha, beta)


@pytensor_rng_jit
def ptd_rvs(alpha, beta, size, rng):
    return ptd_weibull.rvs(alpha, beta, size=size, random_state=rng)
