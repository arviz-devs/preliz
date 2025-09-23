import numpy as np
from pytensor_distributions import cauchy as ptd_cauchy

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Cauchy(Continuous):
    r"""
    Cauchy Distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \alpha, \beta) =
            \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::
        :context: close-figs


        from preliz import Cauchy, style
        style.use('preliz-doc')
        alphas = [0., 0., -2.]
        betas = [.5, 1., 1.]
        for alpha, beta in zip(alphas, betas):
            Cauchy(alpha, beta).plot_pdf(support=(-5,5))

    ========  ==============================================================
    Support   :math:`x \in \mathbb{R}`
    Mean      undefined
    Variance  undefined
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        Location parameter.
    beta : float
        Scale parameter > 0.
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ("alpha", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.params = (self.alpha, self.beta)
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

    def logcdf(self, x):
        return ptd_logcdf(x, self.alpha, self.beta)

    def logsf(self, x):
        return ptd_logsf(x, self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, alpha, beta):
    return ptd_cauchy.pdf(x, alpha, beta)


@pytensor_jit
def ptd_cdf(x, alpha, beta):
    return ptd_cauchy.cdf(x, alpha, beta)


@pytensor_jit
def ptd_ppf(q, alpha, beta):
    return ptd_cauchy.ppf(q, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, alpha, beta):
    return ptd_cauchy.logpdf(x, alpha, beta)


@pytensor_jit
def ptd_entropy(alpha, beta):
    return ptd_cauchy.entropy(alpha, beta)


@pytensor_jit
def ptd_mean(alpha, beta):
    return ptd_cauchy.mean(alpha, beta)


@pytensor_jit
def ptd_mode(alpha, beta):
    return ptd_cauchy.mode(alpha, beta)


@pytensor_jit
def ptd_median(alpha, beta):
    return ptd_cauchy.median(alpha, beta)


@pytensor_jit
def ptd_var(alpha, beta):
    return ptd_cauchy.var(alpha, beta)


@pytensor_jit
def ptd_std(alpha, beta):
    return ptd_cauchy.std(alpha, beta)


@pytensor_jit
def ptd_skewness(alpha, beta):
    return ptd_cauchy.skewness(alpha, beta)


@pytensor_jit
def ptd_kurtosis(alpha, beta):
    return ptd_cauchy.kurtosis(alpha, beta)


@pytensor_jit
def ptd_logcdf(x, alpha, beta):
    return ptd_cauchy.logcdf(x, alpha, beta)


@pytensor_jit
def ptd_logsf(x, alpha, beta):
    return ptd_cauchy.logsf(x, alpha, beta)


@pytensor_rng_jit
def ptd_rvs(alpha, beta, size, rng):
    return ptd_cauchy.rvs(alpha, beta, size=size, random_state=rng)
