import numpy as np
from pytensor_distributions import vonmises as ptd_vonmises
from scipy.stats import circmean

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import find_kappa, optimize_mean_sigma


class VonMises(Continuous):
    r"""
    Univariate VonMises distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::
        :context: close-figs


        from preliz import VonMises, style
        style.use('preliz-doc')
        mus = [0., 0., 0.,  -2.5]
        kappas = [.01, 0.5, 4., 2.]
        for mu, kappa in zip(mus, kappas):
            VonMises(mu, kappa).plot_pdf(support=(-np.pi,np.pi))

    ========  ==========================================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    Variance  :math:`1-\frac{I_1(\kappa)}{I_0(\kappa)}`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    kappa : float
        Concentration (:math:`\frac{1}{\kappa}` is analogous to :math:`\kappa^2`).
    """

    def __init__(self, mu=None, kappa=None):
        super().__init__()
        self._parametrization(mu, kappa)

    def _parametrization(self, mu=None, kappa=None):
        self.mu = mu
        self.kappa = kappa
        self.param_names = ("mu", "kappa")
        self.params_support = ((-np.pi, np.pi), (eps, np.inf))
        self.support = (-np.pi, np.pi)
        if all_not_none(mu, kappa):
            self._update(mu, kappa)

    def _update(self, mu, kappa):
        self.mu = np.float64(mu)
        self.kappa = np.float64(kappa)
        self.params = (self.mu, self.kappa)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.kappa)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.kappa)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.kappa)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.kappa)

    def entropy(self):
        return ptd_entropy(self.mu, self.kappa)

    def mean(self):
        return ptd_mean(self.mu, self.kappa)

    def mode(self):
        return ptd_mode(self.mu, self.kappa)

    def median(self):
        return ptd_median(self.mu, self.kappa)

    def var(self):
        return ptd_var(self.mu, self.kappa)

    def std(self):
        return ptd_std(self.mu, self.kappa)

    def skewness(self):
        return ptd_skewness(self.mu, self.kappa)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.kappa)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.kappa, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        params = mean, 1 / sigma**1.8
        optimize_mean_sigma(self, mean, sigma, params)

    def _fit_mle(self, sample):
        data = np.mod(sample, 2 * np.pi)
        mu = circmean(data)
        kappa = find_kappa(data, mu)
        mu = np.mod(mu + np.pi, 2 * np.pi) - np.pi
        self._update(mu, kappa)

    def eti(self, mass=None, fmt=".2f"):
        mean = self.mu
        self.mu = 0
        hdi_min, hdi_max = super().eti(mass=mass, fmt=fmt)
        self.mu = mean
        return _warp_interval(hdi_min, hdi_max, self.mu, fmt)

    def hdi(self, mass=None, fmt=".2f"):
        mean = self.mu
        self.mu = 0
        hdi_min, hdi_max = super().hdi(mass=mass, fmt=fmt)
        self.mu = mean
        return _warp_interval(hdi_min, hdi_max, self.mu, fmt)


def _warp_interval(hdi_min, hdi_max, mu, fmt):
    hdi_min = hdi_min + mu
    hdi_max = hdi_max + mu

    lower_tail = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
    upper_tail = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))
    if fmt != "none":
        lower_tail = float(f"{lower_tail:{fmt}}")
        upper_tail = float(f"{upper_tail:{fmt}}")
    return (lower_tail, upper_tail)


@pytensor_jit
def ptd_pdf(x, mu, kappa):
    return ptd_vonmises.pdf(x, mu, kappa)


@pytensor_jit
def ptd_cdf(x, mu, kappa):
    return ptd_vonmises.cdf(x, mu, kappa)


@pytensor_jit
def ptd_ppf(q, mu, kappa):
    return ptd_vonmises.ppf(q, mu, kappa)


@pytensor_jit
def ptd_logpdf(x, mu, kappa):
    return ptd_vonmises.logpdf(x, mu, kappa)


@pytensor_jit
def ptd_entropy(mu, kappa):
    return ptd_vonmises.entropy(mu, kappa)


@pytensor_jit
def ptd_mean(mu, kappa):
    return ptd_vonmises.mean(mu, kappa)


@pytensor_jit
def ptd_mode(mu, kappa):
    return ptd_vonmises.mode(mu, kappa)


@pytensor_jit
def ptd_median(mu, kappa):
    return ptd_vonmises.median(mu, kappa)


@pytensor_jit
def ptd_var(mu, kappa):
    return ptd_vonmises.var(mu, kappa)


@pytensor_jit
def ptd_std(mu, kappa):
    return ptd_vonmises.std(mu, kappa)


@pytensor_jit
def ptd_skewness(mu, kappa):
    return ptd_vonmises.skewness(mu, kappa)


@pytensor_jit
def ptd_kurtosis(mu, kappa):
    return ptd_vonmises.kurtosis(mu, kappa)


@pytensor_rng_jit
def ptd_rvs(mu, kappa, size, rng):
    return ptd_vonmises.rvs(mu, kappa, size=size, random_state=rng)
