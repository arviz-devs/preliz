# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from scipy.special import gammaincc, gammainccinv  # pylint: disable=no-name-in-module

from ..internal.distribution_helper import all_not_none, any_not_none, eps
from ..internal.special import gammaln, digamma, cdf_bounds, ppf_bounds_cont
from ..internal.optimization import optimize_ml
from .distributions import Continuous


class InverseGamma(Continuous):
    r"""
    Inverse gamma distribution, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import InverseGamma
        az.style.use('arviz-doc')
        alphas = [1., 2., 3.]
        betas = [1., 1., .5]
        for alpha, beta in zip(alphas, betas):
            InverseGamma(alpha, beta).plot_pdf(support=(0, 3))

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}` for :math:`\alpha > 2`
    ========  ===============================

    Inverse gamma distribution has 2 alternative parameterization. In terms of alpha and
    beta or mu (mean) and sigma (standard deviation).

    The link between the 2 alternatives is given by

    .. math::

       \alpha &= \frac{\mu^2}{\sigma^2} + 2 \\
       \beta  &= \frac{\mu^3}{\sigma^2} + \mu

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    mu : float
        Mean (mu > 0).
    sigma : float
        Standard deviation (sigma > 0)
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, sigma)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None):
        if any_not_none(alpha, beta) and any_not_none(mu, sigma):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            if all_not_none(mu, sigma):
                alpha, beta = _from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu = _to_mu(self.alpha, self.beta)
        self.sigma = _to_sigma(self.alpha, self.beta)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)

        self.is_frozen = True

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
        return nb_cdf(x, self.alpha, self.beta, 0, np.inf)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.alpha, self.beta, 0, np.inf)

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
        return np.where(self.alpha > 1, self.beta / (self.alpha - 1), np.inf)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return np.where(
            self.alpha > 2, self.beta**2 / ((self.alpha - 1) ** 2 * (self.alpha - 2)), np.inf
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return np.where(self.alpha > 3, 4 * (self.alpha - 2) ** 0.5 / (self.alpha - 3), np.nan)

    def kurtosis(self):
        return np.where(
            self.alpha > 4,
            6 * (5 * self.alpha - 11) / ((self.alpha - 3) * (self.alpha - 4)),
            np.nan,
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return 1 / random_state.gamma(self.alpha, 1 / self.beta, size)

    def _fit_moments(self, mean, sigma):
        alpha, beta = _from_mu_sigma(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


def nb_cdf(x, alpha, beta, lower, upper):
    prob = gammaincc(alpha, beta / x)
    return cdf_bounds(prob, x, lower, upper)


def nb_ppf(q, alpha, beta, lower, upper):
    x_val = beta / gammainccinv(alpha, q)
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(alpha, beta):
    return alpha + gammaln(alpha) - (1 + alpha) * digamma(alpha) + np.log(beta)


@nb.njit(cache=True)
def nb_logpdf(x, alpha, beta):
    return alpha * np.log(beta) - gammaln(alpha) - (alpha + 1) * np.log(x) - beta / x


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()


def _from_mu_sigma(mu, sigma):
    alpha = mu**2 / sigma**2 + 2
    beta = mu**3 / sigma**2 + mu
    return alpha, beta


@nb.vectorize(nopython=True, cache=True)
def _to_mu(alpha, beta):
    if alpha > 1:
        return beta / (alpha - 1)
    else:
        return np.nan


@nb.vectorize(nopython=True, cache=True)
def _to_sigma(alpha, beta):
    if alpha > 2:
        return beta / ((alpha - 1) * (alpha - 2) ** 0.5)
    else:
        return np.nan
