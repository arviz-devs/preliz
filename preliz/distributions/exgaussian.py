# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.stats import skew

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import erf, erfc, erfcx, mean_and_std
from ..internal.optimization import find_ppf


class ExGaussian(Continuous):
    r"""
    Exponentially modified Gaussian (EMG) Distribution

    Results from the convolution of a normal distribution with an exponential
    distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \sigma, \tau) =
            \frac{1}{\nu}\;
            \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
            \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ExGaussian
        az.style.use('arviz-doc')
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
        return nb_cdf(x, self.mu, self.sigma, self.nu)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return find_ppf(self, q)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.sigma, self.nu)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma, self.nu)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return self.mu + self.nu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.sigma**2 + self.nu**2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        nus2 = (self.nu / self.sigma) ** 2
        opnus2 = 1.0 + nus2
        return 2 * (self.nu / self.sigma) ** 3 * opnus2 ** (-1.5)

    def kurtosis(self):
        nus2 = (self.nu / self.sigma) ** 2
        opnus2 = 1.0 + nus2
        return 6.0 * nus2 * nus2 * opnus2 ** (-2)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.normal(self.mu, self.sigma, size) + random_state.exponential(
            1 / self.nu, size
        )

    def _fit_moments(self, mean, sigma):
        # Just assume this is a approximately Gaussian
        self._update(mean, sigma, 1e-4)

    def _fit_mle(self, sample):
        mean, std = mean_and_std(sample)
        skweness = skew(sample)
        nu = std * (skweness / 2) ** (1 / 3)
        mu = mean - nu
        var = std**2 * (1 - (skweness / 2) ** (2 / 3))
        self._update(mu, var**0.5, 1 / nu)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, mu, sigma, nu):
    cdf_n = 0.5 * (1 + erf((x - mu) / (sigma * 2**0.5)))
    if x == -np.inf:
        return 0
    elif nu > 0.05 * sigma:
        return cdf_n - 0.5 * np.exp(0.5 / nu * (2 * mu + sigma**2 / nu - 2 * x)) * (
            1 + erf((x - (mu + (sigma**2) / nu)) / (sigma * 2**0.5))
        )
    else:
        return cdf_n


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, sigma, nu):
    if nu > 0.05 * sigma:
        return (
            -np.log(nu)
            + (mu - x) / nu
            + 0.5 * (sigma / nu) ** 2
            + normal_lcdf(x, mu + (sigma**2) / nu, sigma)
        )
    else:
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) / sigma) ** 2


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma, nu):
    return -(nb_logpdf(x, mu, sigma, nu)).sum()


@nb.vectorize(nopython=True, cache=True)
def normal_lcdf(x, mu, sigma):
    z_val = (x - mu) / sigma
    if z_val < -1:
        return np.log(erfcx(-z_val / 2**0.5) / 2) - abs(z_val) ** 2 / 2
    else:
        return np.log1p(-erfc(z_val / 2**0.5) / 2)
