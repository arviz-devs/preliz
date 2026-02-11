import numpy as np
from pytensor_distributions import kumaraswamy as ptd_kumaraswamy

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class Kumaraswamy(Continuous):
    r"""
    Kumaraswamy distribution.

    The pdf of this distribution is

    .. math::

         f(x \mid a, b) = a b x^{a - 1} (1 - x^a)^{b - 1}

    .. plot::
        :context: close-figs


        from preliz import Kumaraswamy, style
        style.use('preliz-doc')
        a_s = [.5, 5., 1., 2., 2.]
        b_s = [.5, 1., 3., 2., 5.]
        for a, b in zip(a_s, b_s):
            ax = Kumaraswamy(a, b).plot_pdf()
            ax.set_ylim(0, 3.)

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
    Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
    ========  ==============================================================

    Parameters
    ----------
    a : float
        a > 0.
    b : float
        b > 0.
    """

    def __init__(self, a=None, b=None):
        super().__init__()
        self.support = (0, 1)
        self._parametrization(a, b)

    def _parametrization(self, a=None, b=None):
        self.a = a
        self.b = b
        self.params = (self.a, self.b)
        self.param_names = ("a", "b")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if (a and b) is not None:
            self._update(a, b)

    def _update(self, a, b):
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.params = (self.a, self.b)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.a, self.b)

    def cdf(self, x):
        return ptd_cdf(x, self.a, self.b)

    def ppf(self, q):
        return ptd_ppf(q, self.a, self.b)

    def logpdf(self, x):
        return ptd_logpdf(x, self.a, self.b)

    def entropy(self):
        return ptd_entropy(self.a, self.b)

    def mean(self):
        return ptd_mean(self.a, self.b)

    def mode(self):
        return ptd_mode(self.a, self.b)

    def median(self):
        return ptd_median(self.a, self.b)

    def var(self):
        return ptd_var(self.a, self.b)

    def std(self):
        return ptd_std(self.a, self.b)

    def skewness(self):
        return ptd_skewness(self.a, self.b)

    def kurtosis(self):
        return ptd_kurtosis(self.a, self.b)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.a, self.b, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample, **kwargs)


@pytensor_jit
def ptd_pdf(x, a, b):
    return ptd_kumaraswamy.pdf(x, a, b)


@pytensor_jit
def ptd_cdf(x, a, b):
    return ptd_kumaraswamy.cdf(x, a, b)


@pytensor_jit
def ptd_ppf(q, a, b):
    return ptd_kumaraswamy.ppf(q, a, b)


@pytensor_jit
def ptd_logpdf(x, a, b):
    return ptd_kumaraswamy.logpdf(x, a, b)


@pytensor_jit
def ptd_entropy(a, b):
    return ptd_kumaraswamy.entropy(a, b)


@pytensor_jit
def ptd_mean(a, b):
    return ptd_kumaraswamy.mean(a, b)


@pytensor_jit
def ptd_mode(a, b):
    return ptd_kumaraswamy.mode(a, b)


@pytensor_jit
def ptd_median(a, b):
    return ptd_kumaraswamy.median(a, b)


@pytensor_jit
def ptd_var(a, b):
    return ptd_kumaraswamy.var(a, b)


@pytensor_jit
def ptd_std(a, b):
    return ptd_kumaraswamy.std(a, b)


@pytensor_jit
def ptd_skewness(a, b):
    return ptd_kumaraswamy.skewness(a, b)


@pytensor_jit
def ptd_kurtosis(a, b):
    return ptd_kumaraswamy.kurtosis(a, b)


@pytensor_rng_jit
def ptd_rvs(a, b, size, rng):
    return ptd_kumaraswamy.rvs(a, b, size=size, random_state=rng)
