# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.stats import skew  # pylint: disable=no-name-in-module
from scipy.special import owens_t  # pylint: disable=no-name-in-module

from .distributions import Continuous
from ..internal.distribution_helper import eps, to_precision, from_precision, all_not_none
from ..internal.special import erf, norm_logcdf
from ..internal.optimization import find_ppf, optimize_ml, optimize_moments


class SkewNormal(Continuous):
    r"""
    SkewNormal distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \tau, \alpha) =
        2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import SkewNormal
        az.style.use('arviz-doc')
        for alpha in [-6, 0, 6]:
            SkewNormal(mu=0, sigma=1, alpha=alpha).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac {\alpha }{{\sqrt {1+\alpha ^{2}}}}`
    Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
    ========  ==========================================

    SkewNormal distribution has 2 alternative parameterizations. In terms of mu, sigma (standard
    deviation) and alpha, or mu, tau (precision) and alpha.

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    alpha : float
        Skewness parameter.
    tau : float
        Precision (tau > 0).

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    and sigma the standard deviation. In the limit of alpha approaching
    plus/minus infinite we get a half-normal distribution.
    """

    def __init__(self, mu=None, sigma=None, alpha=None, tau=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, alpha, tau)

    def _parametrization(self, mu=None, sigma=None, alpha=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu, sigma and alpha,"
                " or mu, tau and alpha."
            )

        self.param_names = ("mu", "sigma", "alpha")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (-np.inf, np.inf))

        if tau is not None:
            self.tau = tau
            sigma = from_precision(tau)
            self.param_names = ("mu", "tau", "alpha")

        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        if all_not_none(self.mu, self.sigma, self.alpha):
            self._update(self.mu, self.sigma, self.alpha)

    def _update(self, mu, sigma, alpha):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.alpha = np.float64(alpha)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma, self.alpha)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau, self.alpha)

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
        return nb_cdf(x, self.mu, self.sigma, self.alpha)

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
        return nb_logpdf(x, self.mu, self.sigma, self.alpha)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma, self.alpha)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return self.mu + self.sigma * np.sqrt(2 / np.pi) * self.alpha / np.sqrt(1 + self.alpha**2)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        delta = self.alpha / (1 + self.alpha**2) ** 0.5
        return self.sigma**2 * (1 - 2 * delta**2 / np.pi)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        delta = self.alpha / (1 + self.alpha**2) ** 0.5
        mean_z = (2 / np.pi) ** 0.5 * delta
        return ((4 - np.pi) / 2) * (mean_z**3 / (1 - mean_z**2) ** (3 / 2))

    def kurtosis(self):
        delta = self.alpha / (1 + self.alpha**2) ** 0.5
        return (
            2
            * (np.pi - 3)
            * ((delta * np.sqrt(2 / np.pi)) ** 4 / (1 - 2 * (delta**2) / np.pi) ** 2)
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        u_0 = random_state.normal(size=size)
        v = random_state.normal(size=size)
        d = self.alpha / np.sqrt(1 + self.alpha**2)
        u_1 = d * u_0 + v * np.sqrt(1 - d**2)
        return np.sign(u_0) * u_1 * self.sigma + self.mu

    def _fit_moments(self, mean, sigma):
        if self.alpha is None:
            self.alpha = 0
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        skewness = skew(sample)
        self.alpha = skewness / (1 - skewness**2) ** 0.5
        optimize_ml(self, sample)


def nb_cdf(x, mu, sigma, alpha):
    return 0.5 * (1 + erf((x - mu) / (sigma * 2**0.5))) - 2 * owens_t((x - mu) / sigma, alpha)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, sigma, alpha):
    if x == np.inf:
        return -np.inf
    elif x == -np.inf:
        return -np.inf
    else:
        z_val = (x - mu) / sigma
        return (
            np.log(2)
            - np.log(sigma)
            - z_val**2 / 2.0
            - np.log((2 * np.pi) ** 0.5)
            + norm_logcdf(alpha * z_val)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma, alpha):
    return -(nb_logpdf(x, mu, sigma, alpha)).sum()
