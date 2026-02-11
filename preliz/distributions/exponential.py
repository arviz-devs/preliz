import numpy as np
from pytensor_distributions import exponential as ptd_exponential

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit


class Exponential(Continuous):
    r"""
    Exponential Distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::
        :context: close-figs


        from preliz import Exponential, style
        style.use('preliz-doc')
        for lam in [0.5,  2.]:
            Exponential(lam).plot_pdf(support=(0,5))


    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Exponential distribution has 2 alternative parametrizations. In terms of lambda (rate)
    or in terms of scale.

    The link between the two alternatives is given by:

    .. math::

        scale = \dfrac{1}{\lambda}

    Parameters
    ----------
    lam : float
        Rate or inverse scale (lam > 0).
    scale : float
        Scale (scale > 0).
    """

    def __init__(self, lam=None, scale=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(lam, scale)

    def _parametrization(self, lam=None, scale=None):
        if all_not_none(lam, scale):
            raise ValueError("Incompatible parametrization. Either use 'lam' or 'scale'.")

        self.param_names = ("lam",)
        self.params_support = ((eps, np.inf),)

        if scale is not None:
            lam = 1 / scale
            self.param_names = ("scale",)

        self.lam = lam
        self.scale = scale
        if self.lam is not None:
            self._update(self.lam)

    def _update(self, lam):
        self.lam = np.float64(lam)
        self.scale = 1 / self.lam

        if self.param_names[0] == "lam":
            self.params = (self.lam,)
        elif self.param_names[0] == "scale":
            self.params = (self.scale,)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.lam)

    def cdf(self, x):
        return ptd_cdf(x, self.lam)

    def ppf(self, q):
        return ptd_ppf(q, self.lam)

    def logpdf(self, x):
        return ptd_logpdf(x, self.lam)

    def entropy(self):
        return ptd_entropy(self.lam)

    def median(self):
        return ptd_median(self.lam)

    def mean(self):
        return ptd_mean(self.lam)

    def mode(self):
        return ptd_mode(self.lam)

    def std(self):
        return ptd_std(self.lam)

    def var(self):
        return ptd_var(self.lam)

    def skewness(self):
        return ptd_skewness(self.lam)

    def kurtosis(self):
        return ptd_kurtosis(self.lam)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.lam, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma=None):
        lam = 1 / mean
        self._update(lam)

    def _fit_mle(self, sample):
        mean = np.mean(sample)
        self._update(1 / mean)


@pytensor_jit
def ptd_pdf(x, lam):
    return ptd_exponential.pdf(x, lam)


@pytensor_jit
def ptd_cdf(x, lam):
    return ptd_exponential.cdf(x, lam)


@pytensor_jit
def ptd_ppf(q, lam):
    return ptd_exponential.ppf(q, lam)


@pytensor_jit
def ptd_logpdf(x, lam):
    return ptd_exponential.logpdf(x, lam)


@pytensor_jit
def ptd_entropy(lam):
    return ptd_exponential.entropy(lam)


@pytensor_jit
def ptd_mean(lam):
    return ptd_exponential.mean(lam)


@pytensor_jit
def ptd_mode(lam):
    return ptd_exponential.mode(lam)


@pytensor_jit
def ptd_median(lam):
    return ptd_exponential.median(lam)


@pytensor_jit
def ptd_var(lam):
    return ptd_exponential.var(lam)


@pytensor_jit
def ptd_std(lam):
    return ptd_exponential.std(lam)


@pytensor_jit
def ptd_skewness(lam):
    return ptd_exponential.skewness(lam)


@pytensor_jit
def ptd_kurtosis(lam):
    return ptd_exponential.kurtosis(lam)


@pytensor_rng_jit
def ptd_rvs(lam, size, rng):
    return ptd_exponential.rvs(lam, size=size, random_state=rng)
