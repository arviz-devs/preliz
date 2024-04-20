# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import bdtr, bdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.optimization import optimize_moments, optimize_ml
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import cdf_bounds, ppf_bounds_disc, gammaln


class ZeroInflatedBinomial(Discrete):
    R"""
    Zero-inflated Binomial distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
            (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
            \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
            \end{array} \right.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ZeroInflatedBinomial
        az.style.use('arviz-doc')
        ns = [10, 20]
        ps = [0.5, 0.7]
        psis = [0.7, 0.4]
        for n, p, psi in zip(ns, ps, psis):
            ZeroInflatedBinomial(psi, n, p).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi n p`
    Variance  :math:`\psi n p (1 - p) + n^2 p^2 (\psi - \psi^2)`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of Binomial variates (0 < psi < 1)
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, psi=None, n=None, p=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(psi, n, p)

    def _parametrization(self, psi=None, n=None, p=None):
        self.psi = psi
        self.n = n
        self.p = p
        self.params = (self.psi, self.n, self.p)
        self.param_names = ("psi", "n", "p")
        self.params_support = ((eps, 1 - eps), (eps, np.inf), (eps, 1 - eps))
        if all_not_none(psi, n, p):
            self._update(psi, n, p)

    def _update(self, psi, n, p):
        self.psi = np.float64(psi)
        self.n = np.int64(n)
        self.p = np.float64(p)
        self.params = (self.psi, self.n, self.p)
        if self.psi == 0:
            self.support = (0, 0)
        else:
            self.support = (0, self.n)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(self.psi, self.n, x, self.p))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        return nb_cdf(x, self.psi, self.n, self.p, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        return nb_ppf(q, self.psi, self.n, self.p, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(self.psi, self.n, x, self.p)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(self.psi, self.n, x, self.p)

    def entropy(self):
        binomial_entropy = 0.5 * np.log(2 * np.pi * np.e * self.n * self.p * (1 - self.p))
        if self.psi == 1:
            return binomial_entropy
        else:
            zero_entropy = -(1 - self.psi) * np.log(1 - self.psi) - self.psi * np.log(self.psi)
            return (1 - self.psi) * zero_entropy + self.psi * binomial_entropy

    def mean(self):
        return self.psi * self.n * self.p

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.mean() * (1 - self.p) + self.n**2 * self.p**2 * (self.psi - self.psi**2)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        # implement skewness
        return np.nan

    def kurtosis(self):
        # implement kurtosis
        return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        zeros = random_state.uniform(size=size) > (1 - self.psi)
        binomial = random_state.binomial(self.n, self.p, size=size)
        return zeros * binomial

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


# @nb.jit
# bdtr not supported by numba
def nb_cdf(x, psi, n, p, lower, upper):
    x = np.asarray(x)
    b_prob = np.asarray(bdtr(x, n, p))
    prob = (1 - psi) + psi * b_prob
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
def nb_ppf(q, psi, n, p, lower, upper):
    q = np.asarray(q)
    n_vals = np.ceil(bdtrik(q, n, p))
    x_vals = (1 - psi) + psi * n_vals
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(psi, n, y, p):
    if y == 0:
        return np.log((1 - psi) + psi * (1 - p) ** n)
    if y > n:
        return -np.inf
    else:
        return (
            np.log(psi)
            + gammaln(n + 1)
            - (gammaln(y + 1) + gammaln(n - y + 1))
            + y * np.log(p)
            + (n - y) * np.log1p(-p)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(psi, n, y, p):
    return -(nb_logpdf(psi, n, y, p)).sum()
