import numpy as np
from pytensor_distributions import halfnormal as ptd_halfnormal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)


class HalfNormal(Continuous):
    r"""
    HalfNormal Distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \sigma) =
           \sqrt{\frac{2}{\pi\sigma^2}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. plot::
        :context: close-figs


        from preliz import HalfNormal, style
        style.use('preliz-doc')
        for sigma in [0.4,  2.]:
            HalfNormal(sigma).plot_pdf(support=(0,5))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    HalfNormal distribution has 2 alternative parameterizations. In terms of sigma (standard
    deviation) or tau (precision).

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`\sigma` (``sigma`` > 0).
    tau : float
        Precision :math:`\tau` (``tau`` > 0).
    """

    def __init__(self, sigma=None, tau=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(sigma, tau)

    def _parametrization(self, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError("Incompatible parametrization. Either use sigma or tau.")

        self.param_names = ("sigma",)
        self.params_support = ((eps, np.inf),)

        if tau is not None:
            sigma = from_precision(tau)
            self.param_names = ("tau",)

        self.sigma = sigma
        self.tau = tau
        if self.sigma is not None:
            self._update(self.sigma)

    def _update(self, sigma):
        self.sigma = np.float64(sigma)
        self.tau = to_precision(self.sigma)

        if self.param_names[0] == "sigma":
            self.params = (self.sigma,)
        elif self.param_names[0] == "tau":
            self.params = (self.tau,)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.sigma)

    def entropy(self):
        return ptd_entropy(self.sigma)

    def mean(self):
        return ptd_mean(self.sigma)

    def mode(self):
        return ptd_mode(self.sigma)

    def median(self):
        return ptd_median(self.sigma)

    def var(self):
        return ptd_var(self.sigma)

    def std(self):
        return ptd_std(self.sigma)

    def skewness(self):
        return ptd_skewness(self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        self._update(sigma / (1 - 2 / np.pi) ** 0.5)

    def _fit_mle(self, sample):
        self._update(np.mean(sample**2) ** 0.5)


@pytensor_jit
def ptd_pdf(x, sigma):
    return ptd_halfnormal.pdf(x, sigma)


@pytensor_jit
def ptd_cdf(x, sigma):
    return ptd_halfnormal.cdf(x, sigma)


@pytensor_jit
def ptd_ppf(q, sigma):
    return ptd_halfnormal.ppf(q, sigma)


@pytensor_jit
def ptd_logpdf(x, sigma):
    return ptd_halfnormal.logpdf(x, sigma)


@pytensor_jit
def ptd_entropy(sigma):
    return ptd_halfnormal.entropy(sigma)


@pytensor_jit
def ptd_mean(sigma):
    return ptd_halfnormal.mean(sigma)


@pytensor_jit
def ptd_mode(sigma):
    return ptd_halfnormal.mode(sigma)


@pytensor_jit
def ptd_median(sigma):
    return ptd_halfnormal.median(sigma)


@pytensor_jit
def ptd_var(sigma):
    return ptd_halfnormal.var(sigma)


@pytensor_jit
def ptd_std(sigma):
    return ptd_halfnormal.std(sigma)


@pytensor_jit
def ptd_skewness(sigma):
    return ptd_halfnormal.skewness(sigma)


@pytensor_jit
def ptd_kurtosis(sigma):
    return ptd_halfnormal.kurtosis(sigma)


@pytensor_rng_jit
def ptd_rvs(sigma, size, rng):
    return ptd_halfnormal.rvs(sigma, size=size, random_state=rng)
