# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import gammaln, xlogy, pdtr, pdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete


class Poisson(Discrete):
    R"""
    Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Poisson
        az.style.use('arviz-white')
        for mu in [0.5, 3, 8]:
            Poisson(mu).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """

    def __init__(self, mu=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu)

    def _parametrization(self, mu=None):
        self.mu = mu
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.params_support = ((eps, np.inf),)
        if mu is not None:
            self._update(mu)

    def _update(self, mu):
        self.mu = np.float64(mu)
        self.params = (self.mu,)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        return nb_pdf(x, self.mu)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        return nb_cdf(x, self.mu)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        return nb_ppf(q, self.mu)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return _logpdf(x, self.mu)

    def entropy(self):
        if self.mu < 50:
            x = np.arange(0, self.ppf(0.9999) + 1, dtype=int)
            logpdf = self.logpdf(x)
            return -np.sum(np.exp(logpdf) * logpdf)
        else:
            return (
                0.5 * np.log(2 * np.pi * np.e * self.mu)
                - 1 / (12 * self.mu)
                - 1 / (24 * self.mu**2)
                - 19 / (360 * self.mu**3)
            )

    def mean(self):
        return self.mu

    def median(self):
        return np.floor(self.mu + 1 / 3 - 0.02 / self.mu)

    def var(self):
        return self.mu

    def std(self):
        return self.mu**0.5

    def skewness(self):
        return 1 / self.mu**0.5

    def kurtosis(self):
        return 1 / self.mu

    def rvs(self, size=1, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.poisson(self.mu, size=size)

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        self._update(nb_fit_mle(sample))


# @nb.jit
# pdtr not supported by numba
def nb_cdf(x, mu):
    x = np.floor(x)
    return np.nan_to_num(pdtr(x, mu))


# @nb.jit
# pdtr not supported by numba
def nb_ppf(q, mu):
    vals = np.ceil(pdtrik(q, mu))
    vals1 = np.maximum(vals - 1, 0)
    temp = pdtr(vals1, mu)
    output = np.where(temp >= q, vals1, vals)
    output[np.isnan(output)] = np.inf
    output[output == 0] = -1
    return output


# @nb.njit
def nb_pdf(x, mu):
    return np.exp(_logpdf(x, mu))


@nb.njit
def nb_fit_mle(sample):
    return np.mean(sample)


# @nb.njit
# xlogy and gammaln not supported by numba
def _logpdf(x, mu):
    x = np.asarray(x)
    return xlogy(x, mu) - gammaln(x + 1) - mu
