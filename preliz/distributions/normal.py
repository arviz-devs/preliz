import numpy as np
import pytensor.tensor as pt
from pytensor_distributions import normal as ptd_normal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)
from preliz.internal.special import mean_and_std


class Normal(Continuous):
    r"""
    Normal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma) =
           \frac{1}{\sigma \sqrt{2\pi}}
           \exp\left\{ -\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2 \right\}

    .. plot::
        :context: close-figs

        from preliz import Normal, style
        style.use('preliz-doc')
        mus = [0., 0., -2.]
        sigmas = [1, 0.5, 1]
        for mu, sigma in zip(mus, sigmas):
            Normal(mu, sigma).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\sigma^2`
    ========  ==========================================

    Normal distribution has 2 alternative parameterizations. In terms of mean and
    sigma (standard deviation), or mean and tau (precision).

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0).
    tau : float
        Precision (tau > 0).
    """

    def __init__(self, mu=None, sigma=None, tau=None):
        super().__init__()
        self.support = (-pt.inf, pt.inf)
        self._parametrization(mu, sigma, tau)

    def _parametrization(self, mu=None, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu and sigma, or mu and tau."
            )

        names = ("mu", "sigma")
        self.params_support = ((-pt.inf, pt.inf), (eps, pt.inf))

        if tau is not None:
            self.tau = tau
            sigma = from_precision(tau)
            names = ("mu", "tau")

        self.mu = mu
        self.sigma = sigma
        self.param_names = names
        if all_not_none(mu, sigma):
            self._update(mu, sigma)

    def _update(self, mu, sigma):
        self.mu = mu  # np.float64(mu)
        self.sigma = sigma  # np.float64(sigma)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        self._update(*mean_and_std(sample))

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma)

    def sf(self, x):
        return ptd_sf(x, self.mu, self.sigma)

    def isf(self, q):
        return ptd_isf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma)

    def logcdf(self, x):
        return ptd_logcdf(x, self.mu, self.sigma)

    def logsf(self, x):
        return ptd_logsf(x, self.mu, self.sigma)

    def logisf(self, q):
        return ptd_logisf(q, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.mu, self.sigma)

    def mode(self):
        return ptd_mode(self.mu, self.sigma)

    def median(self):
        return ptd_median(self.mu, self.sigma)

    def var(self):
        return ptd_var(self.mu, self.sigma)

    def std(self):
        return ptd_std(self.mu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.mu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, size=size, rng=random_state)


@pytensor_jit
def ptd_pdf(x, mu, sigma):
    return ptd_normal.pdf(x, mu, sigma)


@pytensor_jit
def ptd_cdf(x, mu, sigma):
    return ptd_normal.cdf(x, mu, sigma)


@pytensor_jit
def ptd_ppf(q, mu, sigma):
    return ptd_normal.ppf(q, mu, sigma)


@pytensor_jit
def ptd_sf(x, mu, sigma):
    return ptd_normal.sf(x, mu, sigma)


@pytensor_jit
def ptd_isf(q, mu, sigma):
    return ptd_normal.isf(q, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, mu, sigma):
    return ptd_normal.logpdf(x, mu, sigma)


@pytensor_jit
def ptd_logcdf(x, mu, sigma):
    return ptd_normal.logcdf(x, mu, sigma)


@pytensor_jit
def ptd_logsf(x, mu, sigma):
    return ptd_normal.logsf(x, mu, sigma)


@pytensor_jit
def ptd_logisf(q, mu, sigma):
    return ptd_normal.logisf(q, mu, sigma)


@pytensor_jit
def ptd_entropy(mu, sigma):
    return ptd_normal.entropy(mu, sigma)


@pytensor_jit
def ptd_mean(mu, sigma):
    return ptd_normal.mean(mu, sigma)


@pytensor_jit
def ptd_mode(mu, sigma):
    return ptd_normal.mode(mu, sigma)


@pytensor_jit
def ptd_median(mu, sigma):
    return ptd_normal.median(mu, sigma)


@pytensor_jit
def ptd_var(mu, sigma):
    return ptd_normal.var(mu, sigma)


@pytensor_jit
def ptd_std(mu, sigma):
    return ptd_normal.std(mu, sigma)


@pytensor_jit
def ptd_skewness(mu, sigma):
    return ptd_normal.skewness(mu, sigma)


@pytensor_jit
def ptd_kurtosis(mu, sigma):
    return ptd_normal.kurtosis(mu, sigma)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, size, rng):
    return ptd_normal.rvs(mu, sigma, size=size, random_state=rng)
