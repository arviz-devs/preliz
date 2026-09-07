"""Fréchet distribution."""

import numpy as np
from pytensor_distributions import frechet as ptd_frechet

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import brentq, optimize_ml
from preliz.internal.special import gamma, mean_and_std


class Frechet(Continuous):
    r"""
    Fréchet distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \sigma) =
           \frac{\alpha}{\sigma} \left(\frac{x}{\sigma}\right)^{-1-\alpha}
           e^{-(x/\sigma)^{-\alpha}}

    .. plot::
        :context: close-figs

        from preliz import Frechet, style
        style.use('preliz-doc')
        alphas = [1., 2., 3.]
        sigmas = [1., 1., 1.]
        for alpha, sigma in zip(alphas, sigmas):
            ax = Frechet(alpha, sigma).plot_pdf(support=(0, 10))

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma \Gamma(1 - 1/\alpha) \text{ for } \alpha > 1`
    Variance  :math:`\sigma^2 \left(\Gamma(1 - 2/\alpha) - \Gamma(1 - 1/\alpha)^2\right)
                      \text{ for } \alpha > 2`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (``alpha`` > 0).
    sigma : float
        Scale parameter (``sigma`` > 0).
    """

    parametrizations = [("alpha", "sigma")]

    def __init__(self, alpha=None, sigma=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, sigma)

    def _parametrization(self, alpha=None, sigma=None):
        self.param_names = ("alpha", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.alpha = alpha
        self.sigma = sigma
        if all_not_none(self.alpha, self.sigma):
            self._update(self.alpha, self.sigma)

    def _update(self, alpha, sigma):
        self.alpha = np.float64(alpha)
        self.sigma = np.float64(sigma)
        self.params = (self.alpha, self.sigma)
        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        def cv2_error(alpha):
            return gamma(1 - 2 / alpha) / gamma(1 - 1 / alpha) ** 2 - 1 - (sigma / mean) ** 2
        alpha = brentq(cv2_error, np.nextafter(2.0, np.inf), 100.0)
        sigma = mean / gamma(1 - 1 / alpha)
        self._update(alpha, sigma)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        self._fit_moments(mean, std)
        optimize_ml(self, sample)

    def pdf(self, x):
        return ptd_pdf(x, self.alpha, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.alpha, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.alpha, self.sigma)

    def sf(self, x):
        return ptd_sf(x, self.alpha, self.sigma)

    def isf(self, q):
        return ptd_isf(q, self.alpha, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.alpha, self.sigma)

    def logcdf(self, x):
        return ptd_logcdf(x, self.alpha, self.sigma)

    def logsf(self, x):
        return ptd_logsf(x, self.alpha, self.sigma)

    def entropy(self):
        return ptd_entropy(self.alpha, self.sigma)

    def mean(self):
        return ptd_mean(self.alpha, self.sigma)

    def mode(self):
        return ptd_mode(self.alpha, self.sigma)

    def median(self):
        return ptd_median(self.alpha, self.sigma)

    def var(self):
        return ptd_var(self.alpha, self.sigma)

    def std(self):
        return ptd_std(self.alpha, self.sigma)

    def skewness(self):
        return ptd_skewness(self.alpha, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.alpha, self.sigma)

    def lmoment1(self):
        return ptd_lmoment1(self.alpha, self.sigma)

    def lmoment2(self):
        return ptd_lmoment2(self.alpha, self.sigma)

    def lmoment3(self):
        return ptd_lmoment3(self.alpha, self.sigma)

    def lmoment4(self):
        return ptd_lmoment4(self.alpha, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.sigma, size=size, rng=random_state)


@pytensor_jit
def ptd_pdf(x, alpha, sigma):
    return ptd_frechet.pdf(x, alpha, sigma)


@pytensor_jit
def ptd_cdf(x, alpha, sigma):
    return ptd_frechet.cdf(x, alpha, sigma)


@pytensor_jit
def ptd_ppf(q, alpha, sigma):
    return ptd_frechet.ppf(q, alpha, sigma)


@pytensor_jit
def ptd_sf(x, alpha, sigma):
    return ptd_frechet.sf(x, alpha, sigma)


@pytensor_jit
def ptd_isf(q, alpha, sigma):
    return ptd_frechet.isf(q, alpha, sigma)


@pytensor_jit
def ptd_logpdf(x, alpha, sigma):
    return ptd_frechet.logpdf(x, alpha, sigma)


@pytensor_jit
def ptd_logcdf(x, alpha, sigma):
    return ptd_frechet.logcdf(x, alpha, sigma)


@pytensor_jit
def ptd_logsf(x, alpha, sigma):
    return ptd_frechet.logsf(x, alpha, sigma)


@pytensor_jit
def ptd_entropy(alpha, sigma):
    return ptd_frechet.entropy(alpha, sigma)


@pytensor_jit
def ptd_mean(alpha, sigma):
    return ptd_frechet.mean(alpha, sigma)


@pytensor_jit
def ptd_mode(alpha, sigma):
    return ptd_frechet.mode(alpha, sigma)


@pytensor_jit
def ptd_median(alpha, sigma):
    return ptd_frechet.median(alpha, sigma)


@pytensor_jit
def ptd_var(alpha, sigma):
    return ptd_frechet.var(alpha, sigma)


@pytensor_jit
def ptd_std(alpha, sigma):
    return ptd_frechet.std(alpha, sigma)


@pytensor_jit
def ptd_skewness(alpha, sigma):
    return ptd_frechet.skewness(alpha, sigma)


@pytensor_jit
def ptd_kurtosis(alpha, sigma):
    return ptd_frechet.kurtosis(alpha, sigma)


@pytensor_jit
def ptd_lmoment1(alpha, sigma):
    return ptd_frechet.lmoment1(alpha, sigma)


@pytensor_jit
def ptd_lmoment2(alpha, sigma):
    return ptd_frechet.lmoment2(alpha, sigma)


@pytensor_jit
def ptd_lmoment3(alpha, sigma):
    return ptd_frechet.lmoment3(alpha, sigma)


@pytensor_jit
def ptd_lmoment4(alpha, sigma):
    return ptd_frechet.lmoment4(alpha, sigma)


@pytensor_rng_jit
def ptd_rvs(alpha, sigma, size, rng):
    return ptd_frechet.rvs(alpha, sigma, size=size, random_state=rng)
