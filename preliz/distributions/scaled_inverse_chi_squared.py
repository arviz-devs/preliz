import numpy as np
from pytensor_distributions import scaledinversechisquared as ptd_scaledinversechisquared

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


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
        return ptd_pdf(x, self.nu, self.tau2)

    def cdf(self, x):
        return ptd_cdf(x, self.nu, self.tau2)

    def ppf(self, q):
        return ptd_ppf(q, self.nu, self.tau2)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu, self.tau2)

    def entropy(self):
        return ptd_entropy(self.nu, self.tau2)

    def mean(self):
        return ptd_mean(self.nu, self.tau2)

    def mode(self):
        return ptd_mode(self.nu, self.tau2)

    def median(self):
        return ptd_median(self.nu, self.tau2)

    def var(self):
        return ptd_var(self.nu, self.tau2)

    def std(self):
        return ptd_std(self.nu, self.tau2)

    def skewness(self):
        return ptd_skewness(self.nu, self.tau2)

    def kurtosis(self):
        return ptd_kurtosis(self.nu, self.tau2)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, self.tau2, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        cv2 = sigma**2 / mean**2
        nu_hat = 4 + 2 / cv2
        tau2_hat = mean * (nu_hat - 2) / nu_hat
        self._update(nu_hat, tau2_hat)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, nu, tau2):
    return ptd_scaledinversechisquared.pdf(x, nu, tau2)


@pytensor_jit
def ptd_cdf(x, nu, tau2):
    return ptd_scaledinversechisquared.cdf(x, nu, tau2)


@pytensor_jit
def ptd_ppf(q, nu, tau2):
    return ptd_scaledinversechisquared.ppf(q, nu, tau2)


@pytensor_jit
def ptd_logpdf(x, nu, tau2):
    return ptd_scaledinversechisquared.logpdf(x, nu, tau2)


@pytensor_jit
def ptd_entropy(nu, tau2):
    return ptd_scaledinversechisquared.entropy(nu, tau2)


@pytensor_jit
def ptd_mean(nu, tau2):
    return ptd_scaledinversechisquared.mean(nu, tau2)


@pytensor_jit
def ptd_mode(nu, tau2):
    return ptd_scaledinversechisquared.mode(nu, tau2)


@pytensor_jit
def ptd_median(nu, tau2):
    return ptd_scaledinversechisquared.median(nu, tau2)


@pytensor_jit
def ptd_var(nu, tau2):
    return ptd_scaledinversechisquared.var(nu, tau2)


@pytensor_jit
def ptd_std(nu, tau2):
    return ptd_scaledinversechisquared.std(nu, tau2)


@pytensor_jit
def ptd_skewness(nu, tau2):
    return ptd_scaledinversechisquared.skewness(nu, tau2)


@pytensor_jit
def ptd_kurtosis(nu, tau2):
    return ptd_scaledinversechisquared.kurtosis(nu, tau2)


@pytensor_rng_jit
def ptd_rvs(nu, tau2, size, rng):
    return ptd_scaledinversechisquared.rvs(nu, tau2, size=size, random_state=rng)
