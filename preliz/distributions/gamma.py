# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from scipy.special import gammainc, gammaincinv  # pylint: disable=no-name-in-module
from ..internal.distribution_helper import all_not_none, any_not_none, eps
from ..internal.special import cdf_bounds, digamma, gammaln, ppf_bounds_cont, xlogy
from ..internal.optimization import optimize_ml

from .distributions import Continuous


class Gamma(Continuous):
    r"""
    Gamma distribution.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has rate beta.

    The pdf of this distribution is

    .. math::

        f(x \mid \alpha, \beta) =
            \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gamma
        az.style.use('arviz-doc')
        alphas = [1., 3., 7.5]
        betas = [.5, 1., 1.]
        for alpha, beta in zip(alphas, betas):
            Gamma(alpha, beta).plot_pdf()

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Gamma distribution has 2 alternative parameterizations. In terms of alpha and
    beta or mu (mean) and sigma (standard deviation).

    The link between the 2 alternatives is given by

    .. math::

        \alpha &= \frac{\mu^2}{\sigma^2} \\
        \beta  &= \frac{\mu}{\sigma^2}

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).
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
                alpha, beta = self._from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)

        self.is_frozen = True

    def _from_mu_sigma(self, mu, sigma):
        alpha = mu**2 / sigma**2
        beta = mu / sigma**2
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        mu = alpha / beta
        sigma = alpha**0.5 / beta
        return mu, sigma

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
        return self.alpha / self.beta

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.alpha / self.beta**2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 2 / self.alpha**0.5

    def kurtosis(self):
        return 6 / self.alpha

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.gamma(self.alpha, 1 / self.beta, size)

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


def nb_cdf(x, alpha, beta, lower, upper):
    prob = gammainc(alpha, x / (1 / beta))
    return cdf_bounds(prob, x, lower, upper)


def nb_ppf(q, alpha, beta, lower, upper):
    x_val = gammaincinv(alpha, q) * (1 / beta)
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_logpdf(x, alpha, beta):
    x = x / (1 / beta)
    return xlogy(alpha - 1.0, x) - x - gammaln(alpha) - np.log(1 / beta)


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()


@nb.njit(cache=True)
def nb_entropy(alpha, beta):
    return alpha - np.log(beta) + gammaln(alpha) + (1 - alpha) * digamma(alpha)
