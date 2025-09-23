import numpy as np
from pytensor_distributions import exgaussian as ptd_exgaussian
from scipy.stats import skew

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.special import mean_and_std


class ExGaussian(Continuous):
    r"""
    Exponentially modified Gaussian (EMG) Distribution.

    Results from the convolution of a normal distribution with an exponential
    distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \sigma, \nu) =
            \frac{1}{\nu}\;
            \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
            \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    .. plot::
        :context: close-figs


        from preliz import ExGaussian, style
        style.use('preliz-doc')
        mus = [0., 0., -3.]
        sigmas = [1., 3., 1.]
        nus = [1., 1., 4.]
        for mu, sigma, nu in zip(mus, sigmas, nus):
            ExGaussian(mu, sigma, nu).plot_pdf(support=(-6,9))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \nu`
    Variance  :math:`\sigma^2 + \nu^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution (sigma > 0).
    nu : float
        Mean of the exponential distribution (nu > 0).
    """

    def __init__(self, mu=None, sigma=None, nu=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, nu)

    def _parametrization(self, mu=None, sigma=None, nu=None):
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.param_names = ("mu", "sigma", "nu")
        self.params = (mu, sigma, nu)
        #  if nu is too small we get a non-smooth distribution
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (1e-4, np.inf))
        if all_not_none(mu, sigma, nu):
            self._update(mu, sigma, nu)

    def _update(self, mu, sigma, nu):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.nu = np.float64(nu)
        self.params = (self.mu, self.sigma, self.nu)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma, self.nu)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma, self.nu)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma, self.nu)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma, self.nu)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma, self.nu)

    def mean(self):
        return ptd_mean(self.mu, self.sigma, self.nu)

    def median(self):
        return ptd_median(self.mu, self.sigma, self.nu)

    def var(self):
        return ptd_var(self.mu, self.sigma, self.nu)

    def std(self):
        return ptd_std(self.mu, self.sigma, self.nu)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma, self.nu)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma, self.nu)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, self.nu, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        # Just assume this is approximately Gaussian
        self._update(mean, sigma, 1e-4)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        skweness = max(1e-4, skew(sample))
        nu = std * (skweness / 2) ** (1 / 3)
        mu = mean - nu
        var = std**2 * (1 - (skweness / 2) ** (2 / 3))
        self._update(mu, var**0.5, nu)


@pytensor_jit
def ptd_pdf(x, mu, sigma, nu):
    return ptd_exgaussian.pdf(x, mu, sigma, nu)


@pytensor_jit
def ptd_cdf(x, mu, sigma, nu):
    return ptd_exgaussian.cdf(x, mu, sigma, nu)


@pytensor_jit
def ptd_ppf(q, mu, sigma, nu):
    return ptd_exgaussian.ppf(q, mu, sigma, nu)


@pytensor_jit
def ptd_logpdf(x, mu, sigma, nu):
    return ptd_exgaussian.logpdf(x, mu, sigma, nu)


@pytensor_jit
def ptd_entropy(mu, sigma, nu):
    return ptd_exgaussian.entropy(mu, sigma, nu)


@pytensor_jit
def ptd_mean(mu, sigma, nu):
    return ptd_exgaussian.mean(mu, sigma, nu)


@pytensor_jit
def ptd_mode(mu, sigma, nu):
    return ptd_exgaussian.mode(mu, sigma, nu)


@pytensor_jit
def ptd_median(mu, sigma, nu):
    return ptd_exgaussian.median(mu, sigma, nu)


@pytensor_jit
def ptd_var(mu, sigma, nu):
    return ptd_exgaussian.var(mu, sigma, nu)


@pytensor_jit
def ptd_std(mu, sigma, nu):
    return ptd_exgaussian.std(mu, sigma, nu)


@pytensor_jit
def ptd_skewness(mu, sigma, nu):
    return ptd_exgaussian.skewness(mu, sigma, nu)


@pytensor_jit
def ptd_kurtosis(mu, sigma, nu):
    return ptd_exgaussian.kurtosis(mu, sigma, nu)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, nu, size, rng):
    return ptd_exgaussian.rvs(mu, sigma, nu, size=size, random_state=rng)
