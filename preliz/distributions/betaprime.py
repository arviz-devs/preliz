"""BetaPrime distribution."""

import numpy as np
from pytensor_distributions import betaprime as ptd_betaprime

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    any_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import mean_and_std


class BetaPrime(Continuous):
    r"""
    Beta prime distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 + x)^{-\alpha - \beta}}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        from preliz import BetaPrime, style
        style.use('preliz-doc')
        alphas = [.5, 5., 2.]
        betas = [.5, 5., 5.]
        for alpha, beta in zip(alphas, betas):
            ax = BetaPrime(alpha, beta).plot_pdf(support=(0, 5))

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta - 1}` for :math:`\beta > 1`
    Variance  :math:`\dfrac{\alpha(\alpha + \beta - 1)}
              {(\beta - 1)^2(\beta - 2)}` for :math:`\beta > 2`
    ========  ==============================================================

    Beta prime distribution has 2 alternative parameterization. In terms of alpha and
    beta, or mean and nu (precision).

    The link between the 2 alternatives is given by

    .. math::

       \alpha &= \mu (1 + \nu) \\
       \beta  &= 2 + \nu

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0
    mu : float
        mean (``mu`` > 0).
    nu : float
        precision (``nu`` > 0).
    """

    parametrizations = [("alpha", "beta"), ("mu", "nu")]

    def __init__(self, alpha=None, beta=None, mu=None, nu=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, nu)

    def _parametrization(self, alpha=None, beta=None, mu=None, nu=None):
        if any_not_none(alpha, beta) and any_not_none(mu, nu):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta, or mu and nu."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, nu):
            self.mu = mu
            self.nu = nu
            self.param_names = ("mu", "nu")
            self.params_support = ((eps, np.inf), (eps, np.inf))
            if all_not_none(mu, nu):
                alpha, beta = self._from_mu_nu()

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _from_mu_nu(self):
        return ptd_from_mu_nu(self.mu, self.nu)

    def _to_mu_nu(self):
        return ptd_to_mu_nu(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        mu, nu = self._to_mu_nu()
        self.mu, self.nu = np.float64(mu), np.float64(nu)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "nu":
            self.params = (self.mu, self.nu)

        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        nu = mean * (1 + mean) / sigma**2
        self.mu, self.nu = mean, nu
        alpha, beta = self._from_mu_nu()
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)

    def pdf(self, x):
        return ptd_pdf(x, self.alpha, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.alpha, self.beta)

    def sf(self, x):
        return ptd_sf(x, self.alpha, self.beta)

    def isf(self, q):
        return ptd_isf(q, self.alpha, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.alpha, self.beta)

    def logcdf(self, x):
        return ptd_logcdf(x, self.alpha, self.beta)

    def logsf(self, x):
        return ptd_logsf(x, self.alpha, self.beta)

    def logisf(self, q):
        return ptd_logisf(q, self.alpha, self.beta)

    def entropy(self):
        return ptd_entropy(self.alpha, self.beta)

    def mean(self):
        return ptd_mean(self.alpha, self.beta)

    def mode(self):
        return ptd_mode(self.alpha, self.beta)

    def median(self):
        return ptd_median(self.alpha, self.beta)

    def var(self):
        return ptd_var(self.alpha, self.beta)

    def std(self):
        return ptd_std(self.alpha, self.beta)

    def skewness(self):
        return ptd_skewness(self.alpha, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.alpha, self.beta)

    def lmoment1(self):
        return ptd_lmoment1(self.alpha, self.beta)

    def lmoment2(self):
        return ptd_lmoment2(self.alpha, self.beta)

    def lmoment3(self):
        return ptd_lmoment3(self.alpha, self.beta)

    def lmoment4(self):
        return ptd_lmoment4(self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.beta, size=size, rng=random_state)


@pytensor_jit
def ptd_pdf(x, alpha, beta):
    return ptd_betaprime.pdf(x, alpha, beta)


@pytensor_jit
def ptd_cdf(x, alpha, beta):
    return ptd_betaprime.cdf(x, alpha, beta)


@pytensor_jit
def ptd_ppf(q, alpha, beta):
    return ptd_betaprime.ppf(q, alpha, beta)


@pytensor_jit
def ptd_sf(x, alpha, beta):
    return ptd_betaprime.sf(x, alpha, beta)


@pytensor_jit
def ptd_isf(q, alpha, beta):
    return ptd_betaprime.isf(q, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, alpha, beta):
    return ptd_betaprime.logpdf(x, alpha, beta)


@pytensor_jit
def ptd_logcdf(x, alpha, beta):
    return ptd_betaprime.logcdf(x, alpha, beta)


@pytensor_jit
def ptd_logsf(x, alpha, beta):
    return ptd_betaprime.logsf(x, alpha, beta)


@pytensor_jit
def ptd_logisf(q, alpha, beta):
    return ptd_betaprime.logisf(q, alpha, beta)


@pytensor_jit
def ptd_entropy(alpha, beta):
    return ptd_betaprime.entropy(alpha, beta)


@pytensor_jit
def ptd_mean(alpha, beta):
    return ptd_betaprime.mean(alpha, beta)


@pytensor_jit
def ptd_mode(alpha, beta):
    return ptd_betaprime.mode(alpha, beta)


@pytensor_jit
def ptd_median(alpha, beta):
    return ptd_betaprime.median(alpha, beta)


@pytensor_jit
def ptd_var(alpha, beta):
    return ptd_betaprime.var(alpha, beta)


@pytensor_jit
def ptd_std(alpha, beta):
    return ptd_betaprime.std(alpha, beta)


@pytensor_jit
def ptd_skewness(alpha, beta):
    return ptd_betaprime.skewness(alpha, beta)


@pytensor_jit
def ptd_kurtosis(alpha, beta):
    return ptd_betaprime.kurtosis(alpha, beta)


@pytensor_jit
def ptd_lmoment1(alpha, beta):
    return ptd_betaprime.lmoment1(alpha, beta)


@pytensor_jit
def ptd_lmoment2(alpha, beta):
    return ptd_betaprime.lmoment2(alpha, beta)


@pytensor_jit
def ptd_lmoment3(alpha, beta):
    return ptd_betaprime.lmoment3(alpha, beta)


@pytensor_jit
def ptd_lmoment4(alpha, beta):
    return ptd_betaprime.lmoment4(alpha, beta)


@pytensor_rng_jit
def ptd_rvs(alpha, beta, size, rng):
    return ptd_betaprime.rvs(alpha, beta, size=size, random_state=rng)


@pytensor_jit
def ptd_from_mu_nu(mu, nu):
    return ptd_betaprime.from_mu_nu(mu, nu)


@pytensor_jit
def ptd_to_mu_nu(alpha, beta):
    return ptd_betaprime.to_mu_nu(alpha, beta)
