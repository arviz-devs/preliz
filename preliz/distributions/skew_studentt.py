import numpy as np
from pytensor_distributions import skew_studentt as ptd_skew_studentt

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class SkewStudentT(Continuous):
    r"""
    Jones and Faddy's Skewed Student-t distribution.

    The pdf of this distribution is

    .. math::

        f(t)=f(t ; a, b)=C_{a, b}^{-1}\left\{1+\frac{t}{\left(a+b+t^2\right)^{1 / 2}}
        \right\}^{a+1 / 2}\left\{1-\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{b+1 / 2}

    where

    .. math::

        C_{a, b}=2^{a+b-1} B(a, b)(a+b)^{1 / 2}

    .. plot::
        :context: close-figs


        from preliz import SkewStudentT, style
        style.use('preliz-doc')
        mus = [2., 2., 4.]
        sigmas = [1., 2., 2.]
        a_s = [1., 1., 2.]
        b_s = [2., 3., 3.]
        for mu, sigma, a, b in zip(mus, sigmas, a_s, b_s):
            SkewStudentT(mu, sigma, a, b).plot_pdf(support=(-10,6))

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      .. math::
                  \mu + \sigma\frac{(a-b) \sqrt{(a+b)}}{2}\frac{\Gamma\left(a-\frac{1}{2}\right)
                  \Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a) \Gamma(b)}
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location Parameter
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as a and b approach close
    a : float
        First Shape Parameter (a > 0)
    b : float
        Second Shape Parameter (b > 0)
    lam : float
        Scale Parameter (lam > 0). Converges to the precision as a and b approach close

    Notes
    -----
    If a > b, the distribution is positively skewed, and if a < b, the distribution
    is negatively skewed. If a = b, the distribution is StudentT with
    nu (degrees of freedom) = 2a.

    """

    def __init__(self, mu=None, sigma=None, a=None, b=None, lam=None):
        super().__init__()
        self._parametrization(mu, sigma, a, b, lam)

    def _parametrization(self, mu=None, sigma=None, a=None, b=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError("Incompatible parametrization. Either use sigma or lam.")
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.support = (-np.inf, np.inf)
        self.param_names = ("mu", "sigma", "a", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (eps, np.inf), (eps, np.inf))
        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("mu", "lam", "a", "b")
        if all_not_none(mu, sigma, a, b):
            self._update(mu, sigma, a, b)

    def _update(self, mu, sigma, a, b):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(sigma)
        self.a = np.float64(a)
        self.b = np.float64(b)
        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma, self.a, self.b)
        elif self.param_names[1] == "lam":
            self.params = (self.mu, self.lam, self.a, self.b)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.a, self.b, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.a, self.b, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.a, self.b, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.a, self.b, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.a, self.b, self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.a, self.b, self.mu, self.sigma)

    def median(self):
        return ptd_median(self.a, self.b, self.mu, self.sigma)

    def var(self):
        return ptd_var(self.a, self.b, self.mu, self.sigma)

    def std(self):
        return ptd_std(self.a, self.b, self.mu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.a, self.b, self.mu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.a, self.b, self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.a, self.b, self.mu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, a, b, mu, sigma):
    return ptd_skew_studentt.pdf(x, a, b, mu, sigma)


@pytensor_jit
def ptd_cdf(x, a, b, mu, sigma):
    return ptd_skew_studentt.cdf(x, a, b, mu, sigma)


@pytensor_jit
def ptd_ppf(q, a, b, mu, sigma):
    return ptd_skew_studentt.ppf(q, a, b, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, a, b, mu, sigma):
    return ptd_skew_studentt.logpdf(x, a, b, mu, sigma)


@pytensor_jit
def ptd_entropy(a, b, mu, sigma):
    return ptd_skew_studentt.entropy(a, b, mu, sigma)


@pytensor_jit
def ptd_mean(a, b, mu, sigma):
    return ptd_skew_studentt.mean(a, b, mu, sigma)


@pytensor_jit
def ptd_mode(a, b, mu, sigma):
    return ptd_skew_studentt.mode(a, b, mu, sigma)


@pytensor_jit
def ptd_median(a, b, mu, sigma):
    return ptd_skew_studentt.median(a, b, mu, sigma)


@pytensor_jit
def ptd_var(a, b, mu, sigma):
    return ptd_skew_studentt.var(a, b, mu, sigma)


@pytensor_jit
def ptd_std(a, b, mu, sigma):
    return ptd_skew_studentt.std(a, b, mu, sigma)


@pytensor_jit
def ptd_skewness(a, b, mu, sigma):
    return ptd_skew_studentt.skewness(a, b, mu, sigma)


@pytensor_jit
def ptd_kurtosis(a, b, mu, sigma):
    return ptd_skew_studentt.kurtosis(a, b, mu, sigma)


@pytensor_rng_jit
def ptd_rvs(a, b, mu, sigma, size, rng):
    return ptd_skew_studentt.rvs(a, b, mu, sigma, size=size, random_state=rng)
