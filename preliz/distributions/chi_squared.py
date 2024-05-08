# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from scipy.special import gammainc, gammaincinv  # pylint: disable=no-name-in-module

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import cdf_bounds, ppf_bounds_cont, gammaln, digamma, xlogy
from ..internal.optimization import optimize_ml


class ChiSquared(Continuous):
    r"""
    Chi squared  distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) =
                \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ChiSquared
        az.style.use('arviz-doc')
        nus = [1., 3., 9.]
        for nu in nus:
                ax = ChiSquared(nu).plot_pdf(support=(0,20))
                ax.set_ylim(0, 0.6)

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu`
    Variance  :math:`2 \nu`
    ========  ===============================

    Parameters
    ----------
    nu : float
        Degrees of freedom (nu > 0).
    """

    def __init__(self, nu=None):
        super().__init__()
        self.nu = nu
        self.support = (0, np.inf)
        self._parametrization(nu)

    def _parametrization(self, nu=None):
        self.nu = nu
        self.param_names = ("nu",)
        self.params_support = ((eps, np.inf),)
        self.params = (self.nu,)
        if self.nu is not None:
            self._update(self.nu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.nu)
        return frozen

    def _update(self, nu):
        self.nu = np.float64(nu)
        self.params = (self.nu,)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.nu))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.nu)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.nu)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.nu)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.nu)

    def entropy(self):
        return nb_entropy(self.nu)

    def mean(self):
        return self.nu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.nu * 2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (8 / self.nu) ** 0.5

    def kurtosis(self):
        return 12 / self.nu

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.chisquare(self.nu, size)

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


# @nb.njit(cache=True)
def nb_cdf(x, nu):
    return cdf_bounds(gammainc(nu / 2, x / 2), x, 0, np.inf)


# @nb.njit(cache=True)
def nb_ppf(q, nu):
    vals = 2 * gammaincinv(nu / 2, q)
    return ppf_bounds_cont(vals, q, 0, np.inf)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, nu):
    if x < 0:
        return -np.inf
    else:
        return xlogy(nu / 2 - 1, x) - x / 2 - gammaln(nu / 2) - (nu * np.log(2)) / 2


@nb.njit(cache=True)
def nb_neg_logpdf(x, lam):
    return (-nb_logpdf(x, lam)).sum()


@nb.njit(cache=True)
def nb_entropy(nu):
    h_nu = nu / 2
    return h_nu + np.log(2) + gammaln(h_nu) + (1 - h_nu) * digamma(h_nu)
