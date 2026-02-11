import numpy as np
from pytensor_distributions import loglogistic as ptd_loglogistic

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class LogLogistic(Continuous):
    r"""
    Log-Logistic distribution.

    Also known as the Fisk distribution is a continuous non-negative distribution used in survival
    analysis as a parametric model for events whose rate increases initially and decreases later

    The pdf of this distribution is

    .. math::

        f(x\mid \alpha, \beta) =
            \frac{ (\beta/\alpha)(x/\alpha)^{\beta-1}}
            {\left( 1+(x/\alpha)^{\beta} \right)^2}

    .. plot::
        :context: close-figs


        from preliz import LogLogistic, style
        style.use('preliz-doc')
        alphas = [1, 1, 2]
        betas = [4, 8, 8]
        for alpha, beta in zip(alphas, betas):
            LogLogistic(alpha,beta).plot_pdf(support=(0, 6))

    ========  ==========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`{\alpha\,\pi/\beta \over \sin(\pi/\beta)}`
    Variance  :math:`\alpha^2 \left(2b / \sin 2b -b^2 / \sin^2 b \right), \quad \beta>2`
    ========  ==========================================================================

    Parameters
    ----------
    alpha : float
        Scale parameter. (alpha > 0))
    beta : float
        Shape parameter. (beta > 0)).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
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
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, alpha, beta):
    return ptd_loglogistic.pdf(x, alpha, beta)


@pytensor_jit
def ptd_cdf(x, alpha, beta):
    return ptd_loglogistic.cdf(x, alpha, beta)


@pytensor_jit
def ptd_ppf(q, alpha, beta):
    return ptd_loglogistic.ppf(q, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, alpha, beta):
    return ptd_loglogistic.logpdf(x, alpha, beta)


@pytensor_jit
def ptd_entropy(alpha, beta):
    return ptd_loglogistic.entropy(alpha, beta)


@pytensor_jit
def ptd_mean(alpha, beta):
    return ptd_loglogistic.mean(alpha, beta)


@pytensor_jit
def ptd_mode(alpha, beta):
    return ptd_loglogistic.mode(alpha, beta)


@pytensor_jit
def ptd_median(alpha, beta):
    return ptd_loglogistic.median(alpha, beta)


@pytensor_jit
def ptd_var(alpha, beta):
    return ptd_loglogistic.var(alpha, beta)


@pytensor_jit
def ptd_std(alpha, beta):
    return ptd_loglogistic.std(alpha, beta)


@pytensor_jit
def ptd_skewness(alpha, beta):
    return ptd_loglogistic.skewness(alpha, beta)


@pytensor_jit
def ptd_kurtosis(alpha, beta):
    return ptd_loglogistic.kurtosis(alpha, beta)


@pytensor_rng_jit
def ptd_rvs(alpha, beta, size, rng):
    return ptd_loglogistic.rvs(alpha, beta, size=size, random_state=rng)
