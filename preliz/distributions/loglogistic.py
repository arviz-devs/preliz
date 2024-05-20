# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import ppf_bounds_cont, cdf_bounds
from ..internal.optimization import optimize_ml, optimize_moments


class LogLogistic(Continuous):
    r"""
    Log-Logistic distribution.

    Also known as the Fisk distribution is a continuous non-negative distribution used in survival
    analysis as a parametric model for events whose rate increases initially and decreases later

    The pdf of this distribution is

    .. math::

        f(x\mid \alpha, \beta) =
            \frac{ (\beta/\alpha)(x/\alpha)^{\beta-1}}
            {\left( 1+(x/\alpha)^{\beta} \right)^2}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import LogLogistic
        az.style.use('arviz-doc')
        mus = [1, 1, 2]
        sigmas = [4, 8, 8]
        for mu, sigma in zip(mus, sigmas):
            LogLogistic(mu, sigma).plot_pdf(support=(0, 6))

    ========  ==========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`{\alpha\,\pi/\beta \over \sin(\pi/\beta)}`
    Variance  :math:`\alpha^2 \left(2b / \sin 2b -b^2 / \sin^2 b \right), \quad \beta>2`
    ========  ==========================================================================

    Parameters
    ----------
    alpha : float
        Scale parameter. (alpha > 0))
    beta : float
        Shape parameter. (beta > 0)).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta)
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
        return nb_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.alpha, self.beta)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.alpha, self.beta)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.alpha, self.beta)

    def entropy(self):
        return nb_entropy(self.alpha, self.beta)

    def mean(self):
        return np.where(
            self.beta > 1, self.alpha * np.pi / self.beta / np.sin(np.pi / self.beta), np.nan
        )

    def median(self):
        return self.alpha

    def var(self):
        pib = np.pi / self.beta
        return np.where(
            self.beta > 2,
            self.alpha**2 * (2 * pib / np.sin(2 * pib) - pib**2 / np.sin(pib) ** 2),
            np.nan,
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.where(
            self.beta > 3, np.trapz(((x_values - mean) / std) ** 3 * pdf, x_values), np.nan
        )

    def kurtosis(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.where(
            self.beta > 4, np.trapz(((x_values - mean) / std) ** 4 * pdf, x_values) - 3, np.nan
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        u_val = random_state.random(size)
        return self.alpha * (u_val / (1 - u_val)) ** (1 / self.beta)

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, alpha, beta):
    return cdf_bounds(1 / (1 + (x / alpha) ** (-beta)), x, 0, np.inf)


@nb.njit(cache=True)
def nb_ppf(q, alpha, beta):
    return ppf_bounds_cont(alpha * (q / (1 - q)) ** (1 / beta), q, 0, np.inf)


@nb.njit(cache=True)
def nb_entropy(alpha, beta):
    return np.log(alpha) - np.log(beta) + 2


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, alpha, beta):
    if x <= 0:
        return -np.inf
    elif x == np.inf:
        return -np.inf
    else:
        return (
            -2 * np.log1p((x / alpha) ** beta)
            + (-1 + beta) * (np.log(x) - np.log(alpha))
            - np.log(alpha)
            + np.log(beta)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, alpha, beta):
    return -(nb_logpdf(x, alpha, beta)).sum()
