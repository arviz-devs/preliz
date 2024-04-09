# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, any_not_none, all_not_none
from ..internal.optimization import optimize_ml
from ..internal.special import (
    betaln,
    betainc,
    betaincinv,
    digamma,
    gammaln,
    cdf_bounds,
    ppf_bounds_cont,
    mean_and_std,
)


class Beta(Continuous):
    r"""
    Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Beta
        az.style.use('arviz-doc')
        alphas = [.5, 5., 2.]
        betas = [.5, 5., 5.]
        for alpha, beta in zip(alphas, betas):
            ax = Beta(alpha, beta).plot_pdf()
        ax.set_ylim(0, 5)

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Beta distribution has 3 alternative parameterizations. In terms of alpha and
    beta, mean and sigma (standard deviation) or mean and nu (concentration).

    The link between the 3 alternatives is given by

    .. math::

       \alpha &= \mu \nu \\
       \beta  &= (1 - \mu) \nu

       \text{where } \nu = \frac{\mu(1-\mu)}{\sigma^2} - 1


    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    mu : float
        mean (0 < ``mu`` < 1).
    sigma : float
        standard deviation (``sigma`` < sqrt(``mu`` * (1 - ``mu``))).
    nu : float
        concentration > 0
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None, nu=None):
        super().__init__()
        self.support = (0, 1)
        self._parametrization(alpha, beta, mu, sigma, nu)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None, nu=None):
        if any_not_none(alpha, beta) and any_not_none(mu, sigma, nu) or all_not_none(sigma, nu):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta, or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            self.params_support = ((eps, 1 - eps), (eps, 1 - eps))
            if all_not_none(mu, sigma):
                alpha, beta = self._from_mu_sigma(mu, sigma)

        if any_not_none(mu, nu) and sigma is None:
            self.mu = mu
            self.nu = nu
            self.param_names = ("mu", "nu")
            self.params_support = ((eps, 1 - eps), (eps, np.inf))
            if all_not_none(mu, nu):
                alpha, beta = self._from_mu_nu(mu, nu)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

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

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)
        self.nu = self.mu * (1 - self.mu) / self.sigma**2 - 1

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "nu":
            self.params = (self.mu, self.nu)

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
        return self.alpha / (self.alpha + self.beta)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
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
        return random_state.beta(self.alpha, self.beta, size)

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, alpha, beta, lower, upper):
    prob = betainc(alpha, beta, x)
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, alpha, beta, lower, upper):
    x_val = betaincinv(alpha, beta, q)
    return ppf_bounds_cont(x_val, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(alpha, beta):
    psc = alpha + beta
    return (
        betaln(alpha, beta)
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (psc - 2) * digamma(psc)
    )


@nb.njit(cache=True)
def nb_logpdf(x, alpha, beta):
    beta_ = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) - beta_


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()
