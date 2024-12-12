# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from ..internal.optimization import optimize_ml
from ..internal.special import ppf_bounds_cont
from ..internal.distribution_helper import all_not_none, eps
from .distributions import Continuous


class Cauchy(Continuous):
    r"""
    Cauchy Distribution

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
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.alpha, self.beta))

    def cdf(self, x):
        x = np.asarray(x)
        return nb_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        q = np.asarray(q)
        return nb_ppf(q, self.alpha, self.beta, -np.inf, np.inf)

    def logpdf(self, x):
        return nb_logpdf(x, self.alpha, self.beta)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.alpha, self.beta)

    def entropy(self):
        return nb_entropy(self.beta)

    def mean(self):
        return np.nan

    def mode(self):
        return self.alpha

    def median(self):
        return self.alpha

    def var(self):
        return np.nan

    def std(self):
        return np.nan

    def skewness(self):
        return np.nan

    def kurtosis(self):
        return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(0, 1, size)
        return nb_rvs(random_samples, self.alpha, self.beta)

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, alpha, beta):
    return 1 / np.pi * np.arctan((x - alpha) / beta) + 0.5


@nb.njit(cache=True)
def nb_ppf(q, alpha, beta, lower, upper):
    x_val = alpha + beta * np.tan(np.pi * (q - 0.5))
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(beta):
    return np.log(4 * np.pi * beta)


@nb.njit(cache=True)
def nb_logpdf(x, alpha, beta):
    return -np.log(np.pi) - np.log(beta) - np.log(1 + ((x - alpha) / beta) ** 2)


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()


@nb.njit(cache=True)
def nb_rvs(random_samples, alpha, beta):
    return alpha + beta * np.tan(np.pi * (random_samples - 0.5))
