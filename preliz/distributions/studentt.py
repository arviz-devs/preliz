# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from preliz.distributions.normal import Normal
from .distributions import Continuous
from ..internal.distribution_helper import eps, to_precision, from_precision, all_not_none
from ..internal.special import (
    beta,
    betainc,
    betaincinv,
    digamma,
    gammaln,
    erf,
    erfinv,
    ppf_bounds_cont,
)
from ..internal.optimization import optimize_ml


class StudentT(Continuous):
    r"""
    StudentT's distribution.

    Describes a normal variable whose precision is gamma distributed.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu, \mu, \sigma) =
           \frac{\Gamma \left(\frac{\nu+1}{2} \right)} {\sqrt{\nu\pi}\
           \Gamma \left(\frac{\nu}{2} \right)} \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import StudentT
        az.style.use('arviz-doc')
        nus = [2., 5., 5.]
        mus = [0., 0.,  -4.]
        sigmas = [1., 1., 2.]
        for nu, mu, sigma in zip(nus, mus, sigmas):
            StudentT(nu, mu, sigma).plot_pdf(support=(-10,6))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu` for :math:`\nu > 1`, otherwise undefined
    Variance  :math:`\frac{\nu}{\nu-2}` for :math:`\nu > 2`,
              :math:`\infty` for :math:`1 < \nu \le 2`, otherwise undefined
    ========  ========================

    StudentT distribution has 2 alternative parameterization. In terms of nu, mu and
    sigma (standard deviation as nu increases) or nu, mu and lam (precision as nu increases).

    The link between the 2 alternatives is given by

    .. math::

        \lambda = \frac{1}{\sigma^2}

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu increases.
    """

    def __init__(self, nu=None, mu=None, sigma=None, lam=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))
        self._parametrization(nu, mu, sigma, lam)

    def _parametrization(self, nu=None, mu=None, sigma=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError(
                "Incompatible parametrization. Either use nu, mu and sigma, or nu, mu and lam."
            )

        self.param_names = ("nu", "mu", "sigma")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))

        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("nu", "mu", "lam")

        self.nu = nu
        self.mu = mu
        self.sigma = sigma

        if all_not_none(self.nu, self.mu, self.sigma):
            self._update(self.nu, self.mu, self.sigma)

    def _update(self, nu, mu, sigma):
        self.nu = np.float64(nu)
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(sigma)

        if self.param_names[2] == "sigma":
            self.params = (self.nu, self.mu, self.sigma)
        elif self.param_names[2] == "lam":
            self.params = (self.nu, self.mu, self.lam)

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
        return nb_cdf(x, self.nu, self.mu, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.nu, self.mu, self.sigma)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.nu, self.mu, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.nu, self.mu, self.sigma)

    def entropy(self):
        return nb_entropy(self.nu, self.sigma)

    def mean(self):
        return self.mu

    def median(self):
        return self.mu

    def var(self):
        if self.nu > 2:
            return self.sigma**2 * self.nu / (self.nu - 2)
        elif self.nu > 1:
            return np.inf
        else:
            return np.nan

    def std(self):
        if self.nu > 2:
            return self.var() ** 0.5
        elif self.nu > 1:
            return np.inf
        else:
            return np.nan

    def skewness(self):
        if self.nu > 3:
            return 0
        else:
            return np.nan

    def kurtosis(self):
        if self.nu > 4:
            return 6 / (self.nu - 4)
        elif self.nu > 2:
            return np.inf
        else:
            return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return np.where(
            self.nu > 1e10,
            Normal(self.mu, self.sigma).rvs(size, random_state),
            random_state.standard_t(self.nu, size) * self.sigma + self.mu,
        )

    def _fit_moments(self, mean, sigma):
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu is None:
            nu = 100
        elif nu <= 2:
            nu = 2.1
        else:
            sigma = sigma / (nu / (nu - 2)) ** 0.5

        self._update(nu, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, nu, mu, sigma):
    x = (x - mu) / sigma
    factor = 0.5 * betainc(0.5 * nu, 0.5, nu / (x**2 + nu))
    x_vals = np.where(x < 0, factor, 1 - factor)
    return np.where(nu > 1e10, 0.5 * (1 + erf((x - mu) / (sigma * 2**0.5))), x_vals)


@nb.njit(cache=True)
def nb_ppf(p, nu, mu, sigma):
    q = np.where(p < 0.5, p, 1 - p)
    x = betaincinv(0.5 * nu, 0.5, 2 * q)
    x = np.where(p < 0.5, -np.sqrt(nu * (1 - x) / x), np.sqrt(nu * (1 - x) / x))
    vals = np.where(nu > 1e10, mu + sigma * 2**0.5 * erfinv(2 * p - 1), mu + sigma * x)
    return ppf_bounds_cont(vals, p, -np.inf, np.inf)


@nb.vectorize(nopython=True, cache=True)
def nb_entropy(nu, sigma):
    if nu > 1e10:
        return 0.5 * (np.log(2 * np.pi * np.e * sigma**2))
    else:
        return (
            np.log(sigma)
            + 0.5 * (nu + 1) * (digamma(0.5 * (nu + 1)) - digamma(0.5 * nu))
            + np.log(np.sqrt(nu) * beta(0.5 * nu, 0.5))
        )


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, nu, mu, sigma):
    if nu > 1e10:
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) / sigma) ** 2
    else:
        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * sigma**2)
            - 0.5 * (nu + 1) * np.log(1 + ((x - mu) / sigma) ** 2 / nu)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, nu, mu, sigma):
    return -(nb_logpdf(x, nu, mu, sigma)).sum()
