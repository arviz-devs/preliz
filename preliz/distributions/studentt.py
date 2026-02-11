import numpy as np
from pytensor_distributions import studentt as ptd_studentt

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

        from preliz import StudentT, style
        style.use('preliz-doc')
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
        return ptd_pdf(x, self.nu, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.nu, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.nu, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.nu, self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.nu, self.mu, self.sigma)

    def mode(self):
        return ptd_mode(self.nu, self.mu, self.sigma)

    def median(self):
        return ptd_median(self.nu, self.mu, self.sigma)

    def var(self):
        return ptd_var(self.nu, self.mu, self.sigma)

    def std(self):
        return ptd_std(self.nu, self.mu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.nu, self.mu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.nu, self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, self.mu, self.sigma, size=size, rng=random_state)

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


@pytensor_jit
def ptd_pdf(x, nu, mu, sigma):
    return ptd_studentt.pdf(x, nu, mu, sigma)


@pytensor_jit
def ptd_cdf(x, nu, mu, sigma):
    return ptd_studentt.cdf(x, nu, mu, sigma)


@pytensor_jit
def ptd_ppf(q, nu, mu, sigma):
    return ptd_studentt.ppf(q, nu, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, nu, mu, sigma):
    return ptd_studentt.logpdf(x, nu, mu, sigma)


@pytensor_jit
def ptd_entropy(nu, mu, sigma):
    return ptd_studentt.entropy(nu, mu, sigma)


@pytensor_jit
def ptd_mean(nu, mu, sigma):
    return ptd_studentt.mean(nu, mu, sigma)


@pytensor_jit
def ptd_mode(nu, mu, sigma):
    return ptd_studentt.mode(nu, mu, sigma)


@pytensor_jit
def ptd_median(nu, mu, sigma):
    return ptd_studentt.median(nu, mu, sigma)


@pytensor_jit
def ptd_var(nu, mu, sigma):
    return ptd_studentt.var(nu, mu, sigma)


@pytensor_jit
def ptd_std(nu, mu, sigma):
    return ptd_studentt.std(nu, mu, sigma)


@pytensor_jit
def ptd_skewness(nu, mu, sigma):
    return ptd_studentt.skewness(nu, mu, sigma)


@pytensor_jit
def ptd_kurtosis(nu, mu, sigma):
    return ptd_studentt.kurtosis(nu, mu, sigma)


@pytensor_rng_jit
def ptd_rvs(nu, mu, sigma, size, rng):
    return ptd_studentt.rvs(nu, mu, sigma, size=size, random_state=rng)
