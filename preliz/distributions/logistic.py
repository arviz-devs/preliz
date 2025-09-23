import numpy as np
from pytensor_distributions import logistic as ptd_logistic

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Logistic(Continuous):
    r"""
    Logistic distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, s) =
            \frac{ \exp ( - \frac{x - \mu}{s})}{s(1 + \exp ( - \frac{x - \mu}{s}))^2}

    .. plot::
        :context: close-figs


        from preliz import Logistic, style
        style.use('preliz-doc')
        mus = [0., 0., -2.]
        ss = [1., 2., .4]
        for mu, s in zip(mus, ss):
            Logistic(mu, s).plot_pdf(support=(-5,5))

    =========  ==========================================
    Support    :math:`x \in \mathbb{R}`
    Mean       :math:`\mu`
    Variance   :math:`\frac{s^2 \pi^2}{3}`
    =========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    s : float
        Scale (s > 0).
    """

    def __init__(self, mu=None, s=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, s)

    def _parametrization(self, mu=None, s=None):
        self.mu = mu
        self.s = s
        self.params = (self.mu, self.s)
        self.param_names = ("mu", "s")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(self.mu, self.s):
            self._update(self.mu, self.s)

    def _update(self, mu, s):
        self.mu = np.float64(mu)
        self.s = np.float64(s)
        self.params = (self.mu, self.s)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.s)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.s)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.s)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.s)

    def entropy(self):
        return ptd_entropy(self.mu, self.s)

    def mean(self):
        return ptd_mean(self.mu, self.s)

    def mode(self):
        return ptd_mode(self.mu, self.s)

    def median(self):
        return ptd_median(self.mu, self.s)

    def var(self):
        return ptd_var(self.mu, self.s)

    def std(self):
        return ptd_std(self.mu, self.s)

    def skewness(self):
        return ptd_skewness(self.mu, self.s)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.s)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.s, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        s = (3 * sigma**2 / np.pi**2) ** 0.5
        self._update(mean, s)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, s):
    return ptd_logistic.pdf(x, mu, s)


@pytensor_jit
def ptd_cdf(x, mu, s):
    return ptd_logistic.cdf(x, mu, s)


@pytensor_jit
def ptd_ppf(q, mu, s):
    return ptd_logistic.ppf(q, mu, s)


@pytensor_jit
def ptd_logpdf(x, mu, s):
    return ptd_logistic.logpdf(x, mu, s)


@pytensor_jit
def ptd_entropy(mu, s):
    return ptd_logistic.entropy(mu, s)


@pytensor_jit
def ptd_mean(mu, s):
    return ptd_logistic.mean(mu, s)


@pytensor_jit
def ptd_mode(mu, s):
    return ptd_logistic.mode(mu, s)


@pytensor_jit
def ptd_median(mu, s):
    return ptd_logistic.median(mu, s)


@pytensor_jit
def ptd_var(mu, s):
    return ptd_logistic.var(mu, s)


@pytensor_jit
def ptd_std(mu, s):
    return ptd_logistic.std(mu, s)


@pytensor_jit
def ptd_skewness(mu, s):
    return ptd_logistic.skewness(mu, s)


@pytensor_jit
def ptd_kurtosis(mu, s):
    return ptd_logistic.kurtosis(mu, s)


@pytensor_rng_jit
def ptd_rvs(mu, s, size, rng):
    return ptd_logistic.rvs(mu, s, size=size, random_state=rng)
