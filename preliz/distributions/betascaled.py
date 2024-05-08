# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.optimization import optimize_ml
from ..internal.special import (
    betaln,
    betainc,
    betaincinv,
    digamma,
    cdf_bounds,
    ppf_bounds_cont,
    xlogy,
)


class BetaScaled(Continuous):
    r"""
    Scaled Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{(x-\text{lower})^{\alpha - 1} (\text{upper} - x)^{\beta - 1}}
           {(\text{upper}-\text{lower})^{\alpha+\beta-1} B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import BetaScaled
        az.style.use('arviz-doc')
        alphas = [2, 2]
        betas = [2, 5]
        lowers = [-0.5, -1]
        uppers = [1.5, 2]
        for alpha, beta, lower, upper in zip(alphas, betas, lowers, uppers):
            BetaScaled(alpha, beta, lower, upper).plot_pdf()

    ========  ==============================================================
    Support   :math:`x \in (lower, upper)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta} (upper-lower) + lower`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)} (upper-lower)`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, alpha=None, beta=None, lower=0, upper=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.support = (lower, upper)
        self._parametrization(self.alpha, self.beta, self.lower, self.upper)

    def _parametrization(self, alpha=None, beta=None, lower=None, upper=None):
        self.param_names = ("alpha", "beta", "lower", "upper")
        self.params_support = ((eps, np.inf), (eps, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta, lower, upper)

    def _update(self, alpha, beta, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta, self.lower, self.upper)
        self.support = self.lower, self.upper
        self.is_frozen = True

    def _from_mu_sigma(self, mu, sigma):
        nu = mu * (1 - mu) / sigma**2 - 1
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def _from_mu_nu(self, mu, nu):
        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        alpha_plus_beta = alpha + beta
        mu = alpha / alpha_plus_beta
        sigma = (alpha * beta) ** 0.5 / alpha_plus_beta / (alpha_plus_beta + 1) ** 0.5
        return mu, sigma

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.alpha, self.beta, self.lower, self.upper)

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
        return nb_logpdf(x, self.alpha, self.beta, self.lower, self.upper)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.alpha, self.beta, self.lower, self.upper)

    def entropy(self):
        return nb_entropy(self.alpha, self.beta, self.lower, self.upper)

    def mean(self):
        return (self.alpha * self.upper + self.beta * self.lower) / (self.alpha + self.beta)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return (
            (self.alpha * self.beta)
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
            * (self.lower - self.upper) ** 2
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        if self.alpha == self.beta:
            return np.zeros_like(self.alpha)
        else:
            psc = self.alpha + self.beta
            return (2 * (self.beta - self.alpha) * np.sqrt(psc + 1)) / (
                (psc + 2) * np.sqrt(self.alpha * self.beta)
            )

    def kurtosis(self):
        psc = self.alpha + self.beta
        prod = self.alpha * self.beta
        return (
            6
            * (np.abs(self.alpha - self.beta) ** 2 * (psc + 1) - prod * (psc + 2))
            / (prod * (psc + 2) * (psc + 3))
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return (
            random_state.beta(self.alpha, self.beta, size) * (self.upper - self.lower) + self.lower
        )

    def _fit_moments(self, mean, sigma):
        mean = (mean - self.lower) / (self.upper - self.lower)
        sigma = sigma / (self.upper - self.lower)
        kappa = mean * (1 - mean) / sigma**2 - 1
        alpha = max(0.5, kappa * mean)
        beta = max(0.5, kappa * (1 - mean))
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        self._update(None, None, np.min(sample), np.max(sample))
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, alpha, beta, lower, upper):
    prob = betainc(alpha, beta, (x - lower) / (upper - lower))
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, alpha, beta, lower, upper):
    x_val = betaincinv(alpha, beta, q) * (upper - lower) + lower
    return ppf_bounds_cont(x_val, q, lower, upper)


# @nb.njit(cache=True)
def nb_entropy(alpha, beta, lower, upper):
    psc = alpha + beta
    return (
        betaln(alpha, beta)
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (psc - 2) * digamma(psc)
        + np.log(upper - lower)
    )


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, alpha, beta, lower, upper):
    if x < lower or x > upper:
        return -np.inf
    else:
        return (xlogy((alpha - 1), (x - lower)) + xlogy((beta - 1), (upper - x))) - (
            xlogy((alpha + beta - 1), (upper - lower)) + betaln(alpha, beta)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta, lower, upper):
    return -(nb_logpdf(x, alpha, beta, lower, upper)).sum()
