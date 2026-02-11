import numpy as np
from pytensor_distributions import lognormal as ptd_lognormal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.special import (
    mean_and_std,
)


class LogNormal(Continuous):
    r"""
    Log-normal distribution.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma) =
          \frac{1}{x \sigma \sqrt{2\pi}}
           \exp\left\{ -\frac{(\ln(x)-\mu)^2}{2\sigma^2} \right\}

    .. plot::
        :context: close-figs


        from preliz import LogNormal, style
        style.use('preliz-doc')
        mus = [ 0., 0.]
        sigmas = [.5, 1.]
        for mu, sigma in zip(mus, sigmas):
            LogNormal(mu, sigma).plot_pdf(support=(0,5))

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\exp\left(\mu+\frac{\sigma^2}{2}\right)`
    Variance  :math:`[\exp(\sigma^2)-1] \exp(2\mu+\sigma^2)`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Standard deviation. (sigma > 0)).
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, sigma):
            self._update(mu, sigma)

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.mu, self.sigma)

    def mode(self):
        return ptd_mode(self.mu, self.sigma)

    def median(self):
        return ptd_median(self.mu, self.sigma)

    def var(self):
        return ptd_var(self.mu, self.sigma)

    def std(self):
        return ptd_std(self.mu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        mu, sigma = mean_and_std(np.log(sample))
        self._update(mu, sigma)


@pytensor_jit
def ptd_pdf(x, mu, sigma):
    return ptd_lognormal.pdf(x, mu, sigma)


@pytensor_jit
def ptd_cdf(x, mu, sigma):
    return ptd_lognormal.cdf(x, mu, sigma)


@pytensor_jit
def ptd_ppf(q, mu, sigma):
    return ptd_lognormal.ppf(q, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, mu, sigma):
    return ptd_lognormal.logpdf(x, mu, sigma)


@pytensor_jit
def ptd_entropy(mu, sigma):
    return ptd_lognormal.entropy(mu, sigma)


@pytensor_jit
def ptd_mean(mu, sigma):
    return ptd_lognormal.mean(mu, sigma)


@pytensor_jit
def ptd_mode(mu, sigma):
    return ptd_lognormal.mode(mu, sigma)


@pytensor_jit
def ptd_median(mu, sigma):
    return ptd_lognormal.median(mu, sigma)


@pytensor_jit
def ptd_var(mu, sigma):
    return ptd_lognormal.var(mu, sigma)


@pytensor_jit
def ptd_std(mu, sigma):
    return ptd_lognormal.std(mu, sigma)


@pytensor_jit
def ptd_skewness(mu, sigma):
    return ptd_lognormal.skewness(mu, sigma)


@pytensor_jit
def ptd_kurtosis(mu, sigma):
    return ptd_lognormal.kurtosis(mu, sigma)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, size, rng):
    return ptd_lognormal.rvs(mu, sigma, size=size, random_state=rng)
