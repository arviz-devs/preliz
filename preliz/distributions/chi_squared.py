import numpy as np
from pytensor_distributions import chisquared as ptd_chisquared

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class ChiSquared(Continuous):
    r"""
    Chi squared  distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) =
                \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs


        from preliz import ChiSquared, style
        style.use('preliz-doc')
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

    def _update(self, nu):
        self.nu = np.float64(nu)
        self.params = (self.nu,)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.nu)

    def cdf(self, x):
        return ptd_cdf(x, self.nu)

    def ppf(self, q):
        return ptd_ppf(q, self.nu)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu)

    def entropy(self):
        return ptd_entropy(self.nu)

    def mean(self):
        return ptd_mean(self.nu)

    def mode(self):
        return ptd_mode(self.nu)

    def median(self):
        return ptd_median(self.nu)

    def var(self):
        return ptd_var(self.nu)

    def std(self):
        return ptd_std(self.nu)

    def skewness(self):
        return ptd_skewness(self.nu)

    def kurtosis(self):
        return ptd_kurtosis(self.nu)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma=None):
        self._update(mean)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, nu):
    return ptd_chisquared.pdf(x, nu)


@pytensor_jit
def ptd_cdf(x, nu):
    return ptd_chisquared.cdf(x, nu)


@pytensor_jit
def ptd_ppf(q, nu):
    return ptd_chisquared.ppf(q, nu)


@pytensor_jit
def ptd_logpdf(x, nu):
    return ptd_chisquared.logpdf(x, nu)


@pytensor_jit
def ptd_entropy(nu):
    return ptd_chisquared.entropy(nu)


@pytensor_jit
def ptd_mean(nu):
    return ptd_chisquared.mean(nu)


@pytensor_jit
def ptd_mode(nu):
    return ptd_chisquared.mode(nu)


@pytensor_jit
def ptd_median(nu):
    return ptd_chisquared.median(nu)


@pytensor_jit
def ptd_var(nu):
    return ptd_chisquared.var(nu)


@pytensor_jit
def ptd_std(nu):
    return ptd_chisquared.std(nu)


@pytensor_jit
def ptd_skewness(nu):
    return ptd_chisquared.skewness(nu)


@pytensor_jit
def ptd_kurtosis(nu):
    return ptd_chisquared.kurtosis(nu)


@pytensor_rng_jit
def ptd_rvs(nu, size, rng):
    return ptd_chisquared.rvs(nu, size=size, random_state=rng)
