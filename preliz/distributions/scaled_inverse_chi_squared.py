import numba as nb
import numpy as np
from scipy.special import gammaincc, gammaincinv

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import cdf_bounds, digamma, gammaln, ppf_bounds_cont


class ScaledInverseChiSquared(Continuous):
    r"""
    Scaled Inverse Chi squared  distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu, \tau^2) =
                    \frac{(\tau^2\nu/2)^{\nu/2}}{\Gamma(\nu/2)}~
                    \frac{\exp\left[ \frac{-\nu \tau^2}{2 x}\right]}{x^{1+\nu/2}}


    .. plot::
        :context: close-figs


        from preliz import ScaledInverseChiSquared, style
        style.use('preliz-doc')
        nus =  [4., 4, 10.]
        tau2s = [1., 2, 1.]
        for nu, tau2 in zip(nus, tau2s):
            ScaledInverseChiSquared(nu, tau2).plot_pdf(support=(0, 5))

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu \tau^2 / (\nu - 2)` for :math:`\nu > 2`, else :math:`\infty`
    Variance  :math:`\frac{2 \nu^2 \tau^4}{(\nu - 2)^2 (\nu - 4)}`
              for :math:`\nu > 4`, else :math:`\infty`
    ========  ===============================


    Parameters
    ----------
    nu : float
        Degrees of freedom (nu > 0).
    tau2 : float
        Scale (tau2 > 0).
    """

    def __init__(self, nu=None, tau2=None):
        super().__init__()
        self.nu = nu
        self.tau2 = tau2
        self.support = (0, np.inf)
        self._parametrization(nu, tau2)

    def _parametrization(self, nu=None, tau2=None):
        self.nu = nu
        self.tau2 = tau2
        self.param_names = ("nu", "tau2")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.params = (self.nu, self.tau2)
        if all_not_none(nu, tau2):
            self._update(self.nu, self.tau2)

    def _update(self, nu, tau2):
        self.nu = np.float64(nu)
        self.tau2 = np.float64(tau2)
        self.params = (self.nu, self.tau2)
        self.is_frozen = True

    def pdf(self, x):
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.nu, self.tau2))

    def cdf(self, x):
        x = np.asarray(x)
        return nb_cdf(x, self.nu, self.tau2)

    def ppf(self, q):
        q = np.asarray(q)
        return nb_ppf(q, self.nu, self.tau2)

    def logpdf(self, x):
        return nb_logpdf(x, self.nu, self.tau2)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.nu, self.tau2)

    def entropy(self):
        return nb_entropy(self.nu, self.tau2)

    def mean(self):
        return np.where(self.nu > 2, self.nu * self.tau2 / (self.nu - 2), np.inf)

    def mode(self):
        return self.nu * self.tau2 / (self.nu + 2)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return np.where(
            self.nu > 4,
            2 * self.nu**2 * self.tau2**2 / ((self.nu - 2) ** 2 * (self.nu - 4)),
            np.inf,
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return np.where(self.nu > 6, 4 / (self.nu - 6) * np.sqrt(2 * (self.nu - 4)), np.inf)

    def kurtosis(self):
        return np.where(
            self.nu > 8, (12 * (5 * self.nu - 22)) / ((self.nu - 6) * (self.nu - 8)), np.inf
        )

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return (self.nu * self.tau2) / random_state.chisquare(self.nu, size)

    def _fit_moments(self, mean, sigma):
        cv2 = sigma**2 / mean**2
        nu_hat = 4 + 2 / cv2
        tau2_hat = mean * (nu_hat - 2) / nu_hat
        self._update(nu_hat, tau2_hat)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


# @nb.njit(cache=True)
def nb_cdf(x, nu, tau2):
    h_nu = nu / 2
    return cdf_bounds(gammaincc(h_nu, h_nu * tau2 / x), x, 0, np.inf)


# @nb.njit(cache=True)
def nb_ppf(q, nu, tau2):
    h_nu = nu / 2
    vals = h_nu * tau2 / gammaincinv(h_nu, 1 - q)
    return ppf_bounds_cont(vals, q, 0, np.inf)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, nu, tau2):
    if x < 0:
        return -np.inf
    else:
        h_nu = nu / 2
        return (
            -(np.log(x) * (h_nu + 1))
            - (h_nu * tau2) / x
            + np.log(tau2) * h_nu
            - gammaln(h_nu)
            + np.log(h_nu) * h_nu
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, nu, tau2):
    return (-nb_logpdf(x, nu, tau2)).sum()


@nb.njit(cache=True)
def nb_entropy(nu, tau2):
    h_nu = nu / 2
    return h_nu + np.log(h_nu * tau2) + gammaln(h_nu) - (1 + h_nu) * digamma(h_nu)
