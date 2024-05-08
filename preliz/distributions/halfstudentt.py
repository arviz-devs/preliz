# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, to_precision, from_precision, all_not_none
from ..internal.special import (
    digamma,
    gamma,
    gammaln,
    beta,
    betainc,
    betaincinv,
    cdf_bounds,
    ppf_bounds_cont,
)
from ..internal.optimization import optimize_ml


class HalfStudentT(Continuous):
    r"""
    HalfStudentT Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfStudentT
        az.style.use('arviz-doc')
        sigmas = [1., 2., 2.]
        nus = [3, 3., 10.]
        for sigma, nu in zip(sigmas, nus):
            HalfStudentT(nu, sigma).plot_pdf(support=(0,10))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      .. math::
                  2\sigma\sqrt{\frac{\nu}{\pi}}\
                  \frac{\Gamma\left(\frac{\nu+1}{2}\right)}\
                  {\Gamma\left(\frac{\nu}{2}\right)(\nu-1)}\, \text{for } \nu > 2
    Variance  .. math::
                  \sigma^2\left(\frac{\nu}{\nu - 2}-\
                  \frac{4\nu}{\pi(\nu-1)^2}\left(\frac{\Gamma\left(\frac{\nu+1}{2}\right)}\
                  {\Gamma\left(\frac{\nu}{2}\right)}\right)^2\right) \text{for } \nu > 2\, \infty\
                  \text{for } 1 < \nu \le 2\, \text{otherwise undefined}
    ========  ==========================================

    HalfStudentT distribution has 2 alternative parameterizations. In terms of nu and
    sigma (standard deviation as nu increases) or nu lam (precision as nu increases).

    The link between the 2 alternatives is given by

    .. math::

        \lambda = \frac{1}{\sigma^2}

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu increases.
    """

    def __init__(self, nu=None, sigma=None, lam=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(nu, sigma, lam)

    def _parametrization(self, nu=None, sigma=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError(
                "Incompatible parametrization. Either use nu and sigma, or nu and lam."
            )

        self.param_names = ("nu", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("nu", "lam")

        self.nu = nu
        self.sigma = sigma
        if all_not_none(self.nu, self.sigma):
            self._update(self.nu, self.sigma)

    def _update(self, nu, sigma):
        self.nu = np.float64(nu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(self.sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.nu, self.sigma)
        elif self.param_names[1] == "lam":
            self.params = (self.nu, self.lam)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.nu, self.sigma))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.nu, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.nu, self.sigma)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.nu, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.nu, self.sigma)

    def entropy(self):
        return nb_entropy(self.nu, self.sigma)

    def mean(self):
        if self.nu > 1:
            gamma0 = gamma((self.nu + 1) / 2)
            gamma1 = gamma(self.nu / 2)
            if np.isfinite(gamma0) and np.isfinite(gamma1):
                mean = (
                    2 * self.sigma * (self.nu / np.pi) ** 0.5 * (gamma0 / (gamma1 * (self.nu - 1)))
                )
            else:
                mean = self.sigma * (2 / np.pi) ** 0.5
        return mean

    def median(self):
        return self.ppf(0.5)

    def var(self):
        gamma0 = gamma((self.nu + 1) / 2)
        gamma1 = gamma(self.nu / 2)
        if self.nu > 2:
            if np.isfinite(gamma0) and np.isfinite(gamma1):
                var = self.sigma**2 * (
                    (self.nu / (self.nu - 2))
                    - ((4 * self.nu) / (np.pi * (self.nu - 1) ** 2)) * (gamma0 / gamma1) ** 2
                )
            else:
                # assume nu is large enough that the std of the halfnormal is a good approximation
                var = self.sigma**2 * (1 - 2.0 / np.pi)
        return var

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return NotImplemented

    def kurtosis(self):
        return NotImplemented

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return np.abs(random_state.standard_t(self.nu, size) * self.sigma)

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu is None:
            nu = 100
        elif nu <= 2:
            nu = 2.1

        gamma0 = gamma((nu + 1) / 2)
        gamma1 = gamma(nu / 2)
        if np.isfinite(gamma0) and np.isfinite(gamma1):
            sigma = (
                sigma**2
                / ((nu / (nu - 2)) - ((4 * nu) / (np.pi * (nu - 1) ** 2)) * (gamma0 / gamma1) ** 2)
            ) ** 0.5
        else:
            # we assume a Gaussian for large nu
            sigma = sigma / (1 - 2 / np.pi) ** 0.5
        self._update(nu, sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, nu, sigma):
    x = x / sigma
    factor = 0.5 * betainc(0.5 * nu, 0.5, nu / (x**2 + nu))
    return cdf_bounds(np.where(x < 0, factor, 1 - factor) * 2 - 1, x, 0, np.inf)


@nb.njit(cache=True)
def nb_ppf(p, nu, sigma):
    p_factor = (p + 1) / 2
    inv_factor = np.where(
        p_factor < 0.5,
        betaincinv(0.5 * nu, 0.5, 2 * p_factor),
        np.sqrt(nu / betaincinv(0.5 * nu, 0.5, 2 - 2 * p_factor) - nu),
    )
    return ppf_bounds_cont(inv_factor * sigma, p, 0, np.inf)


@nb.njit(cache=True)
def nb_entropy(nu, sigma):
    return (
        np.log(sigma)
        + 0.5 * (nu + 1) * (digamma(0.5 * (nu + 1)) - digamma(0.5 * nu))
        + np.log(np.sqrt(nu) * beta(0.5 * nu, 0.5))
        - np.log(2)
    )


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, nu, sigma):
    if x < 0:
        return -np.inf
    else:
        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * sigma**2)
            - 0.5 * (nu + 1) * np.log(1 + (x / sigma) ** 2 / nu)
            + np.log(2)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, nu, sigma):
    return -(nb_logpdf(x, nu, sigma)).sum()
