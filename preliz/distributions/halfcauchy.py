# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from ..internal.special import cdf_bounds, ppf_bounds_cont
from ..internal.optimization import optimize_ml
from ..internal.distribution_helper import eps
from .distributions import Continuous


class HalfCauchy(Continuous):
    r"""
    HalfCauchy Distribution

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
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.beta))

    def cdf(self, x):
        x = np.asarray(x)
        return nb_cdf(x, self.beta, 0, np.inf)

    def ppf(self, q):
        q = np.asarray(q)
        return nb_ppf(q, self.beta, 0, np.inf)

    def logpdf(self, x):
        return nb_logpdf(x, self.beta)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.beta)

    def entropy(self):
        return nb_entropy(self.beta)

    def mean(self):
        return np.inf

    def mode(self):
        return np.zeros_like(self.beta)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return np.inf

    def std(self):
        return np.inf

    def skewness(self):
        return np.nan

    def kurtosis(self):
        return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(0, 1, size)
        return nb_rvs(random_samples, self.beta)

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        self._update(sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, beta, lower, upper):
    prob = 2 / np.pi * np.arctan(x / beta)
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, beta, lower, upper):
    x_val = beta * np.tan(np.pi / 2 * q)
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(beta):
    return np.log(2 * beta * np.pi)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, beta):
    if x < 0:
        return -np.inf
    else:
        return np.log(2) - np.log(np.pi * beta) - np.log(1 + (x / beta) ** 2)


@nb.njit(cache=True)
def nb_neg_logpdf(x, beta):
    return (-nb_logpdf(x, beta)).sum()


@nb.njit(cache=True)
def nb_rvs(random_samples, beta):
    return beta * np.tan(np.pi / 2 * random_samples)
