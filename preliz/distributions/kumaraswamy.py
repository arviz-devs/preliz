# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.special import (
    beta,
    digamma,
    cdf_bounds,
    ppf_bounds_cont,
)


class Kumaraswamy(Continuous):
    r"""
    Kumaraswamy distribution.

    The pdf of this distribution is

    .. math::

         f(x \mid a, b) = a b x^{a - 1} (1 - x^a)^{b - 1}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Kumaraswamy
        az.style.use('arviz-doc')
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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.a, self.b)
        return frozen

    def _update(self, a, b):
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.params = (self.a, self.b)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.a, self.b))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.a, self.b)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.a, self.b)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.a, self.b)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.a, self.b)

    def entropy(self):
        return nb_entropy(self.a, self.b)

    def mean(self):
        return _mom(self.a, self.b, 1)

    def median(self):
        return (1 - 2 ** -(1 / self.b)) ** (1 / self.a)

    def var(self):
        m_1 = _mom(self.a, self.b, 1)
        m_2 = _mom(self.a, self.b, 2)
        return m_2 - m_1**2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mean = self.mean()
        var = self.var()
        m_3 = _mom(self.a, self.b, 3)
        return (m_3 - 3 * mean * var - mean**3) / var**1.5

    def kurtosis(self):
        mean = self.mean()
        var = self.var()
        m_2 = _mom(self.a, self.b, 2)
        m_3 = _mom(self.a, self.b, 3)
        m_4 = _mom(self.a, self.b, 4)
        return (m_4 + mean * (-4 * m_3 + mean * (6 * m_2 - 3 * mean**2))) / var**2 - 3

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.random(size))

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample, **kwargs)


@nb.njit(cache=True)
def nb_cdf(x, a, b):
    prob = 1 - (1 - x**a) ** b
    return cdf_bounds(prob, x, 0, 1)


@nb.njit(cache=True)
def nb_ppf(q, a, b):
    x_val = (1 - (1 - q) ** (1 / b)) ** (1 / a)
    return ppf_bounds_cont(x_val, q, 0, 1)


@nb.njit(cache=True)
def nb_entropy(a, b):
    h_b = digamma(b + 1) + np.euler_gamma
    return (1 - 1 / b) + (1 - 1 / a) * h_b - np.log(a) - np.log(b)


@nb.njit(cache=True)
def nb_logpdf(x, a, b):
    return np.log(a * b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)


@nb.njit(cache=True)
def nb_neg_logpdf(x, a, b):
    return -(nb_logpdf(x, a, b)).sum()


def _mom(a, b, n):
    return b * beta(1 + n / a, b)
