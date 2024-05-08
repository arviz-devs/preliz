# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import erf, erfinv  # pylint: disable=no-name-in-module

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import erf, erfinv, mean_and_std, ppf_bounds_cont, cdf_bounds


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

        import arviz as az
        from preliz import LogNormal
        az.style.use('arviz-doc')
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
        return nb_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        return nb_entropy(self.mu, self.sigma)

    def mean(self):
        return np.exp(self.mu + self.sigma**2 / 2)

    def median(self):
        return np.exp(self.mu)

    def var(self):
        return (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (np.exp(self.sigma**2) + 2) * (np.exp(self.sigma**2) - 1) ** 0.5

    def kurtosis(self):
        return (
            np.exp(4 * self.sigma**2)
            + 2 * np.exp(3 * self.sigma**2)
            + 3 * np.exp(2 * self.sigma**2)
            - 6
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.lognormal(self.mu, self.sigma, size)

    def _fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        mu, sigma = mean_and_std(np.log(sample))
        self._update(mu, sigma)


@nb.njit(cache=True)
def nb_cdf(x, mu, sigma):
    return cdf_bounds(0.5 * (1 + erf((np.log(x) - mu) / (sigma * 2**0.5))), x, 0, np.inf)


@nb.njit(cache=True)
def nb_ppf(q, mu, sigma):
    return ppf_bounds_cont(np.exp(mu + sigma * 2**0.5 * erfinv(2 * q - 1)), q, 0, np.inf)


@nb.njit(cache=True)
def nb_entropy(mu, sigma):
    return np.log((2 * np.pi) ** 0.5 * sigma * np.exp(mu + 0.5))


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, sigma):
    if x <= 0:
        return -np.inf
    else:
        return -(np.log(x) + np.log(sigma) + 0.5 * np.log(2 * np.pi)) - ((np.log(x) - mu) ** 2) / (
            2 * sigma**2
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma):
    return -(nb_logpdf(x, mu, sigma)).sum()
