import numpy as np
from pytensor_distributions import logitnormal as ptd_logitnormal

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    eps,
    from_precision,
    pytensor_jit,
    pytensor_rng_jit,
    to_precision,
)
from preliz.internal.special import (
    logit,
    mean_and_std,
)


class LogitNormal(Continuous):
    r"""
    Logit-Normal distribution.

    The pdf of this distribution is

    .. math::
       f(x \mid \mu, \tau) =
           \frac{1}{x(1-x)} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (logit(x)-\mu)^2 \right\}


    .. plot::
        :context: close-figs


        from preliz import LogitNormal, style
        style.use('preliz-doc')
        mus = [0., 0., 0., 1.]
        sigmas = [0.3, 1., 2., 1.]
        for mu, sigma in zip(mus, sigmas):
            LogitNormal(mu, sigma).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in (0, 1)`
    Mean      no analytical solution
    Variance  no analytical solution
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    tau : float
        Scale parameter (tau > 0).
    """

    def __init__(self, mu=None, sigma=None, tau=None):
        super().__init__()
        self.support = (0, 1)
        self._parametrization(mu, sigma, tau)

    def _parametrization(self, mu=None, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu and sigma, or mu and tau."
            )

        names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))

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
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.mu, self.sigma)

    def mean(self):
        return ptd_mean(self.mu, self.sigma)

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

    def mode(self):
        return ptd_mode(self.mu, self.sigma)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        mu = logit(mean)
        sigma = np.diff((mean - sigma * 3, mean + sigma * 3))
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        mu, sigma = mean_and_std(logit(sample))
        self._update(mu, sigma)


@pytensor_jit
def ptd_pdf(x, mu, sigma):
    return ptd_logitnormal.pdf(x, mu, sigma)


@pytensor_jit
def ptd_cdf(x, mu, sigma):
    return ptd_logitnormal.cdf(x, mu, sigma)


@pytensor_jit
def ptd_ppf(q, mu, sigma):
    return ptd_logitnormal.ppf(q, mu, sigma)


@pytensor_jit
def ptd_logpdf(x, mu, sigma):
    return ptd_logitnormal.logpdf(x, mu, sigma)


@pytensor_jit
def ptd_entropy(mu, sigma):
    return ptd_logitnormal.entropy(mu, sigma)


@pytensor_jit
def ptd_mean(mu, sigma):
    return ptd_logitnormal.mean(mu, sigma)


@pytensor_jit
def ptd_mode(mu, sigma):
    return ptd_logitnormal.mode(mu, sigma)


@pytensor_jit
def ptd_median(mu, sigma):
    return ptd_logitnormal.median(mu, sigma)


@pytensor_jit
def ptd_var(mu, sigma):
    return ptd_logitnormal.var(mu, sigma)


@pytensor_jit
def ptd_std(mu, sigma):
    return ptd_logitnormal.std(mu, sigma)


@pytensor_jit
def ptd_skewness(mu, sigma):
    return ptd_logitnormal.skewness(mu, sigma)


@pytensor_jit
def ptd_kurtosis(mu, sigma):
    return ptd_logitnormal.kurtosis(mu, sigma)


@pytensor_rng_jit
def ptd_rvs(mu, sigma, size, rng):
    return ptd_logitnormal.rvs(mu, sigma, size=size, random_state=rng)
