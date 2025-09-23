import numpy as np
from pytensor_distributions import skewnormal as ptd_skewnormal
from scipy.stats import skew

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


class SkewNormal(Continuous):
    r"""
    SkewNormal distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \tau, \alpha) =
        2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::
        :context: close-figs


        from preliz import SkewNormal, style
        style.use('preliz-doc')
        for alpha in [-6, 0, 6]:
            SkewNormal(mu=0, sigma=1, alpha=alpha).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac {\alpha }{{\sqrt {1+\alpha ^{2}}}}`
    Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
    ========  ==========================================

    SkewNormal distribution has 2 alternative parameterizations. In terms of mu, sigma (standard
    deviation) and alpha, or mu, tau (precision) and alpha.

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    alpha : float
        Skewness parameter.
    tau : float
        Precision (tau > 0).

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    and sigma the standard deviation. In the limit of alpha approaching
    plus/minus infinite we get a half-normal distribution.
    """

    def __init__(self, mu=None, sigma=None, alpha=None, tau=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, alpha, tau)

    def _parametrization(self, mu=None, sigma=None, alpha=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu, sigma and alpha,"
                " or mu, tau and alpha."
            )

        self.param_names = ("mu", "sigma", "alpha")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (-np.inf, np.inf))

        if tau is not None:
            self.tau = tau
            sigma = from_precision(tau)
            self.param_names = ("mu", "tau", "alpha")

        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        if all_not_none(self.mu, self.sigma, self.alpha):
            self._update(self.mu, self.sigma, self.alpha)

    def _update(self, mu, sigma, alpha):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.alpha = np.float64(alpha)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma, self.alpha)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau, self.alpha)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma, self.alpha)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma, self.alpha)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma, self.alpha)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma, self.alpha)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma, self.alpha)

    def mean(self):
        return ptd_mean(self.mu, self.sigma, self.alpha)

    def median(self):
        return ptd_median(self.mu, self.sigma, self.alpha)

    def var(self):
        return ptd_var(self.mu, self.sigma, self.alpha)

    def std(self):
        return ptd_std(self.mu, self.sigma, self.alpha)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma, self.alpha)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma, self.alpha)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, self.alpha, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        if self.alpha is None:
            self.alpha = 0
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        skewness = skew(sample)
        self.alpha = skewness / (1 - skewness**2) ** 0.5
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, sigma, alpha):
    return ptd_skewnormal.pdf(x, mu, sigma, alpha)


@pytensor_jit
def ptd_cdf(x, mu, sigma, alpha):
    return ptd_skewnormal.cdf(x, mu, sigma, alpha)


@pytensor_jit
def ptd_ppf(q, mu, sigma, alpha):
    return ptd_skewnormal.ppf(q, mu, sigma, alpha)


@pytensor_jit
def ptd_logpdf(x, mu, sigma, alpha):
    return ptd_skewnormal.logpdf(x, mu, sigma, alpha)


@pytensor_jit
def ptd_entropy(mu, sigma, alpha):
    return ptd_skewnormal.entropy(mu, sigma, alpha)


@pytensor_jit
def ptd_mean(mu, sigma, alpha):
    return ptd_skewnormal.mean(mu, sigma, alpha)


@pytensor_jit
def ptd_mode(mu, sigma, alpha):
    return ptd_skewnormal.mode(mu, sigma, alpha)


@pytensor_jit
def ptd_median(mu, sigma, alpha):
    return ptd_skewnormal.median(mu, sigma, alpha)


@pytensor_jit
def ptd_var(mu, sigma, alpha):
    return ptd_skewnormal.var(mu, sigma, alpha)


@pytensor_jit
def ptd_std(mu, sigma, alpha):
    return ptd_skewnormal.std(mu, sigma, alpha)


@pytensor_jit
def ptd_skewness(mu, sigma, alpha):
    return ptd_skewnormal.skewness(mu, sigma, alpha)


@pytensor_jit
def ptd_kurtosis(mu, sigma, alpha):
    return ptd_skewnormal.kurtosis(mu, sigma, alpha)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, alpha, size, rng):
    return ptd_skewnormal.rvs(mu, sigma, alpha, size=size, random_state=rng)
