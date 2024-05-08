# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.optimization import optimize_ml
from ..internal.special import (
    garcia_approximation,
    gamma,
    mean_and_std,
    cdf_bounds,
    ppf_bounds_cont,
    xlogy,
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

        import arviz as az
        from preliz import Weibull
        az.style.use('arviz-doc')
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.alpha, self.beta))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.alpha, self.beta, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.alpha, self.beta, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.alpha, self.beta)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.alpha, self.beta)

    def entropy(self):
        return nb_entropy(self.alpha, self.beta)

    def mean(self):
        return self.beta * gamma(1 + 1 / self.alpha)

    def median(self):
        return self.beta * np.log(2) ** (1 / self.alpha)

    def var(self):
        return self.beta**2 * gamma(1 + 2 / self.alpha) - self.mean() ** 2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mu = self.mean()
        sigma = self.std()
        m_s = mu / sigma
        return gamma(1 + 3 / self.alpha) * (self.beta / sigma) ** 3 - 3 * m_s - m_s**3

    def kurtosis(self):
        mu = self.mean()
        sigma = self.std()
        skew = self.skewness()
        m_s = mu / sigma
        return (
            (self.beta / sigma) ** 4 * gamma(1 + 4 / self.alpha)
            - 4 * skew * m_s
            - 6 * m_s**2
            - m_s**4
            - 3
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.weibull(self.alpha, size) * self.beta

    def _fit_moments(self, mean, sigma):
        alpha, beta = garcia_approximation(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, alpha, beta, lower, upper):
    prob = 1 - np.exp(-((x / beta) ** alpha))
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, alpha, beta, lower, upper):
    x_val = beta * (-np.log(1 - q)) ** (1 / alpha)
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(alpha, beta):
    return np.euler_gamma * (1 - 1 / alpha) + np.log(beta / alpha) + 1


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, alpha, beta):
    if x < 0:
        return -np.inf
    else:
        x_b = x / beta
        return np.log(alpha / beta) + xlogy((alpha - 1), x_b) - x_b**alpha


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()
