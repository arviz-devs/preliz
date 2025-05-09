import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect
from scipy.special import i0e, i1e
from scipy.stats import circmean

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps
from preliz.internal.optimization import find_kappa, optimize_moments


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
        x = np.asarray(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        return nb_cdf(x, self.pdf)

    def ppf(self, q):
        return nb_ppf(q, self.pdf)

    def logpdf(self, x):
        return nb_logpdf(x, self.mu, self.kappa)

    def _neg_logpdf(self, x):
        return nb_neg_logpdf(x, self.mu, self.kappa)

    def entropy(self):
        return nb_entropy(self.kappa, self.var())

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def median(self):
        return self.mu

    def var(self):
        return 1 - i1e(self.kappa) / i0e(self.kappa)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 0

    def kurtosis(self):
        return 0

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.vonmises(self.mu, self.kappa, size)

    def _fit_moments(self, mean, sigma):
        params = mean, 1 / sigma**1.8
        optimize_moments(self, mean, sigma, params)

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


def nb_cdf(x, pdf):
    if isinstance(x, int | float):
        x = [x]
        scalar_input = True
    else:
        scalar_input = False

    cdf_values = np.array([quad(pdf, -np.pi, xi)[0] if xi <= np.pi else 1 for xi in x])

    return cdf_values[0] if scalar_input else cdf_values


def nb_ppf(q, pdf):
    def root_func(x, q):
        return nb_cdf(x, pdf) - q

    if isinstance(q, int | float):
        q = [q]
        scalar_input = True
    else:
        scalar_input = False

    ppf_values = []
    for q_i in q:
        if q_i < 0:
            val = np.nan
        elif q_i > 1:
            val = np.nan
        elif q_i == 0:
            val = -np.inf
        elif q_i == 1:
            val = np.inf
        else:
            val = bisect(root_func, -np.pi, np.pi, args=(q_i,))

        ppf_values.append(val)

    return ppf_values[0] if scalar_input else np.array(ppf_values)


def nb_entropy(kappa, var):
    return np.log(2 * np.pi * i0e(kappa)) + kappa * var


def nb_logpdf(x, mu, kappa):
    return kappa * (np.cos(x - mu) - 1) - np.log(2 * np.pi) - np.log(i0e(kappa))


def nb_neg_logpdf(x, mu, kappa):
    return -(nb_logpdf(x, mu, kappa)).sum()


def _warp_interval(hdi_min, hdi_max, mu, fmt):
    hdi_min = hdi_min + mu
    hdi_max = hdi_max + mu

    lower_tail = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
    upper_tail = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))
    if fmt != "none":
        lower_tail = float(f"{lower_tail:{fmt}}")
        upper_tail = float(f"{upper_tail:{fmt}}")
    return (lower_tail, upper_tail)
