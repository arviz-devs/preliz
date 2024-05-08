# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import pdtr, pdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.distribution_helper import eps, all_not_none
from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.special import gammaln, xlogy, cdf_bounds, ppf_bounds_disc


class ZeroInflatedPoisson(Discrete):
    R"""
    Zero-inflated Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
            \psi \frac{e^{-\mu}\mu^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ZeroInflatedPoisson
        az.style.use('arviz-doc')
        psis = [0.7, 0.4]
        mus = [8, 4]
        for psi, mu in zip(psis, mus):
            ZeroInflatedPoisson(psi, mu).plot_pdf()

    ========  ================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi \mu`
    Variance  :math:`\psi \mu (1+(1-\psi) \mu`
    ========  ================================

    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    mu : float
        Expected number of occurrences during the given interval
        (mu >= 0).
    """

    def __init__(self, psi=None, mu=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(psi, mu)

    def _parametrization(self, psi=None, mu=None):
        self.psi = psi
        self.mu = mu
        self.params = (self.psi, self.mu)
        self.param_names = ("psi", "mu")
        self.params_support = ((eps, 1 - eps), (eps, np.inf))
        if all_not_none(psi, mu):
            self._update(psi, mu)

    def _update(self, psi, mu):
        self.psi = np.float64(psi)
        self.mu = np.float64(mu)
        self.params = (self.psi, self.mu)
        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.psi, self.mu))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        return nb_cdf(x, self.psi, self.mu, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        return nb_ppf(q, self.psi, self.mu, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.psi, self.mu)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.psi, self.mu)

    def entropy(self):
        if self.mu < 50:
            x = self.xvals("full", 5000)
            logpdf = self.logpdf(x)
            return -np.sum(np.exp(logpdf) * logpdf)
        else:
            poisson_entropy = (
                0.5 * np.log(2 * np.pi * np.e * self.mu)
                - 1 / (12 * self.mu)
                - 1 / (24 * self.mu**2)
                - 19 / (360 * self.mu**3)
            )
            if self.psi == 1:
                return poisson_entropy
            else:
                # The var can be 0 with probability 1-psi or something else with probability psi
                zero_entropy = -(1 - self.psi) * np.logp(1 - self.psi) - self.psi * np.log(self.psi)
                # The total entropy is the weighted sum of the two entropies
                return (1 - self.psi) * zero_entropy + self.psi * poisson_entropy

    def mean(self):
        return self.psi * self.mu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.psi * self.mu * (1 + (1 - self.psi) * self.mu)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return np.nan

    def kurtosis(self):
        return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        zeros = random_state.uniform(size=size) > (1 - self.psi)
        poisson = random_state.poisson(self.mu, size=size)
        return zeros * poisson


# @nb.jit
# pdtr not supported by numba
def nb_cdf(x, psi, mu, lower, upper):
    p_prob = pdtr(x, mu)
    prob = (1 - psi) + psi * p_prob
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
# pdtr not supported by numba
def nb_ppf(q, psi, mu, lower, upper):
    q = np.asarray(q)
    vals = np.ceil(pdtrik(q, mu))
    vals1 = np.maximum(vals - 1, 0)
    temp = pdtr(vals1, mu)
    p_vals = np.where(temp >= q, vals1, vals)
    x_vals = (1 - psi) + psi * p_vals
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, psi, mu):
    if x < 0:
        return -np.inf
    elif x == 0:
        return np.log(np.exp(-mu) * psi - psi + 1)
    else:
        return np.log(psi) + xlogy(x, mu) - gammaln(x + 1) - mu


@nb.njit(cache=True)
def nb_neg_logpdf(x, psi, mu):
    return -(nb_logpdf(x, psi, mu)).sum()
