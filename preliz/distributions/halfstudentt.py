import numpy as np
from pytensor_distributions import halfstudentt as ptd_halfstudentt

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import (
    gamma,
)


class HalfStudentT(Continuous):
    r"""
    HalfStudentT Distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        from preliz import HalfStudentT, style
        style.use('preliz-doc')
        sigmas = [1., 2., 2.]
        nus = [3, 3., 10.]
        for sigma, nu in zip(sigmas, nus):
            HalfStudentT(nu, sigma).plot_pdf(support=(0,10))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      .. math::
                  2\sigma\sqrt{\frac{\nu}{\pi}}\
                  \frac{\Gamma\left(\frac{\nu+1}{2}\right)}
                  {\Gamma\left(\frac{\nu}{2}\right)(\nu-1)}\, \text{for } \nu > 2
    Variance  .. math::
                  \sigma^2\left(\frac{\nu}{\nu - 2}-\
                  \frac{4\nu}{\pi(\nu-1)^2}\left(\frac{\Gamma\left(\frac{\nu+1}{2}\right)}
                  {\Gamma\left(\frac{\nu}{2}\right)}\right)^2\right) \text{for } \nu > 2\, \infty\
                  \text{for } 1 < \nu \le 2\, \text{otherwise undefined}
    ========  ==========================================

    HalfStudentT distribution has 2 alternative parameterizations. In terms of nu and
    sigma (standard deviation as nu increases) or nu and lam (precision as nu increases).

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
        return ptd_pdf(x, self.nu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.nu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.nu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.nu, self.sigma)

    def mean(self):
        return ptd_mean(self.nu, self.sigma)

    def mode(self):
        return ptd_mode(self.nu, self.sigma)

    def median(self):
        return ptd_median(self.nu, self.sigma)

    def var(self):
        return ptd_var(self.nu, self.sigma)

    def std(self):
        return ptd_std(self.nu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.nu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.nu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
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

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, nu, sigma):
    return ptd_halfstudentt.pdf(x, nu, sigma)


@pytensor_jit
def ptd_cdf(x, nu, sigma):
    return ptd_halfstudentt.cdf(x, nu, sigma)


@pytensor_jit
def ptd_ppf(q, nu, sigma):
    return ptd_halfstudentt.ppf(q, nu, sigma)


@pytensor_jit
def ptd_logpdf(x, nu, sigma):
    return ptd_halfstudentt.logpdf(x, nu, sigma)


@pytensor_jit
def ptd_entropy(nu, sigma):
    return ptd_halfstudentt.entropy(nu, sigma)


@pytensor_jit
def ptd_mean(nu, sigma):
    return ptd_halfstudentt.mean(nu, sigma)


@pytensor_jit
def ptd_mode(nu, sigma):
    return ptd_halfstudentt.mode(nu, sigma)


@pytensor_jit
def ptd_median(nu, sigma):
    return ptd_halfstudentt.median(nu, sigma)


@pytensor_jit
def ptd_var(nu, sigma):
    return ptd_halfstudentt.var(nu, sigma)


@pytensor_jit
def ptd_std(nu, sigma):
    return ptd_halfstudentt.std(nu, sigma)


@pytensor_jit
def ptd_skewness(nu, sigma):
    return ptd_halfstudentt.skewness(nu, sigma)


@pytensor_jit
def ptd_kurtosis(nu, sigma):
    return ptd_halfstudentt.kurtosis(nu, sigma)


@pytensor_rng_jit
def ptd_rvs(nu, sigma, size, rng):
    return ptd_halfstudentt.rvs(nu, sigma, size=size, random_state=rng)
