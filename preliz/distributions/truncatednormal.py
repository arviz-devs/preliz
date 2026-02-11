import numpy as np
from pytensor_distributions import truncatednormal as ptd_truncatednormal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class TruncatedNormal(Continuous):
    r"""
    TruncatedNormal distribution.

    The pdf of this distribution is

    .. math::

       f(x;\mu ,\sigma ,a,b)={\frac {\phi ({\frac {x-\mu }{\sigma }})}{
            \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)}}

    .. plot::
        :context: close-figs


        from preliz import TruncatedNormal, style
        style.use('preliz-doc')
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        lowers = [-3, -5, -5]
        uppers = [7, 5, 4]
        for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
            TruncatedNormal(mu, sigma, lower, upper).plot_pdf(support=(-10,10))

    ========  ==========================================
    Support   :math:`x \in [a, b]`
    Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
    Variance  .. math::
                  \sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-
                  \left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0)
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, mu=None, sigma=None, lower=None, upper=None):
        super().__init__()
        self._parametrization(mu, sigma, lower, upper)

    def _parametrization(self, mu=None, sigma=None, lower=None, upper=None):
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.param_names = ("mu", "sigma", "lower", "upper")
        self.params_support = (
            (-np.inf, np.inf),
            (eps, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
        )
        if lower is None:
            self.lower = -np.inf
        if upper is None:
            self.upper = np.inf
        self.support = (self.lower, self.upper)
        if all_not_none(mu, sigma, lower, upper):
            self._update(mu, sigma, lower, upper)

    def _update(self, mu, sigma, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma, self.lower, self.upper)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma, self.lower, self.upper)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma, self.lower, self.upper)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma, self.lower, self.upper)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma, self.lower, self.upper)

    def mean(self):
        return ptd_mean(self.mu, self.sigma, self.lower, self.upper)

    def mode(self):
        return ptd_mode(self.mu, self.sigma, self.lower, self.upper)

    def median(self):
        return ptd_median(self.mu, self.sigma, self.lower, self.upper)

    def var(self):
        return ptd_var(self.mu, self.sigma, self.lower, self.upper)

    def std(self):
        return ptd_std(self.mu, self.sigma, self.lower, self.upper)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma, self.lower, self.upper)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma, self.lower, self.upper)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, self.lower, self.upper, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        # Assume gaussian
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        self._update(None, None, np.min(sample), np.max(sample))
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, sigma, lower, upper):
    return ptd_truncatednormal.pdf(x, mu, sigma, lower, upper)


@pytensor_jit
def ptd_cdf(x, mu, sigma, lower, upper):
    return ptd_truncatednormal.cdf(x, mu, sigma, lower, upper)


@pytensor_jit
def ptd_ppf(q, mu, sigma, lower, upper):
    return ptd_truncatednormal.ppf(q, mu, sigma, lower, upper)


@pytensor_jit
def ptd_logpdf(x, mu, sigma, lower, upper):
    return ptd_truncatednormal.logpdf(x, mu, sigma, lower, upper)


@pytensor_jit
def ptd_entropy(mu, sigma, lower, upper):
    return ptd_truncatednormal.entropy(mu, sigma, lower, upper)


@pytensor_jit
def ptd_mean(mu, sigma, lower, upper):
    return ptd_truncatednormal.mean(mu, sigma, lower, upper)


@pytensor_jit
def ptd_mode(mu, sigma, lower, upper):
    return ptd_truncatednormal.mode(mu, sigma, lower, upper)


@pytensor_jit
def ptd_median(mu, sigma, lower, upper):
    return ptd_truncatednormal.median(mu, sigma, lower, upper)


@pytensor_jit
def ptd_var(mu, sigma, lower, upper):
    return ptd_truncatednormal.var(mu, sigma, lower, upper)


@pytensor_jit
def ptd_std(mu, sigma, lower, upper):
    return ptd_truncatednormal.std(mu, sigma, lower, upper)


@pytensor_jit
def ptd_skewness(mu, sigma, lower, upper):
    return ptd_truncatednormal.skewness(mu, sigma, lower, upper)


@pytensor_jit
def ptd_kurtosis(mu, sigma, lower, upper):
    return ptd_truncatednormal.kurtosis(mu, sigma, lower, upper)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, lower, upper, size, rng):
    return ptd_truncatednormal.rvs(mu, sigma, lower, upper, size=size, random_state=rng)
