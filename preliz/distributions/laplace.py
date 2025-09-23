import numba as nb
import numpy as np
from pytensor_distributions import laplace as ptd_laplace

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit


class Laplace(Continuous):
    r"""
    Laplace distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::
        :context: close-figs


        from preliz import Laplace, style
        style.use('preliz-doc')
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            Laplace(mu, b).plot_pdf(support=(-10,10))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    """

    def __init__(self, mu=None, b=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, b)

    def _parametrization(self, mu=None, b=None):
        self.mu = mu
        self.b = b
        self.params = (self.mu, self.b)
        self.param_names = ("mu", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, b):
            self._update(mu, b)

    def _update(self, mu, b):
        self.mu = np.float64(mu)
        self.b = np.float64(b)
        self.params = (self.mu, self.b)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.b)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.b)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.b)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.b)

    def entropy(self):
        return ptd_entropy(self.mu, self.b)

    def median(self):
        return ptd_median(self.mu, self.b)

    def mean(self):
        return ptd_mean(self.mu, self.b)

    def mode(self):
        return ptd_mode(self.mu, self.b)

    def std(self):
        return ptd_std(self.mu, self.b)

    def var(self):
        return ptd_var(self.mu, self.b)

    def skewness(self):
        return ptd_skewness(self.mu, self.b)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.b)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.b, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        b = (sigma / 2) * (2**0.5)
        self._update(mean, b)

    def _fit_mle(self, sample):
        mu, b = nb_fit_mle(sample)
        self._update(mu, b)


@pytensor_jit
def ptd_pdf(x, mu, b):
    return ptd_laplace.pdf(x, mu, b)


@pytensor_jit
def ptd_cdf(x, mu, b):
    return ptd_laplace.cdf(x, mu, b)


@pytensor_jit
def ptd_ppf(q, mu, b):
    return ptd_laplace.ppf(q, mu, b)


@pytensor_jit
def ptd_logpdf(x, mu, b):
    return ptd_laplace.logpdf(x, mu, b)


@pytensor_jit
def ptd_entropy(mu, b):
    return ptd_laplace.entropy(mu, b)


@pytensor_jit
def ptd_mean(mu, b):
    return ptd_laplace.mean(mu, b)


@pytensor_jit
def ptd_mode(mu, b):
    return ptd_laplace.mode(mu, b)


@pytensor_jit
def ptd_median(mu, b):
    return ptd_laplace.median(mu, b)


@pytensor_jit
def ptd_var(mu, b):
    return ptd_laplace.var(mu, b)


@pytensor_jit
def ptd_std(mu, b):
    return ptd_laplace.std(mu, b)


@pytensor_jit
def ptd_skewness(mu, b):
    return ptd_laplace.skewness(mu, b)


@pytensor_jit
def ptd_kurtosis(mu, b):
    return ptd_laplace.kurtosis(mu, b)


@pytensor_rng_jit
def ptd_rvs(mu, b, size, rng):
    return ptd_laplace.rvs(mu, b, size=size, random_state=rng)


@nb.njit(cache=True)
def nb_fit_mle(sample):
    median = np.median(sample)
    scale = np.sum(np.abs(sample - median)) / len(sample)
    return median, scale
