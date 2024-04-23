# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-import
"""
Continuous probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats

from ..internal.distribution_helper import all_not_none
from .distributions import Continuous
from .asymmetric_laplace import AsymmetricLaplace
from .beta import Beta
from .cauchy import Cauchy
from .chi_squared import ChiSquared
from .exponential import Exponential
from .gamma import Gamma
from .gumbel import Gumbel
from .halfcauchy import HalfCauchy
from .halfnormal import HalfNormal
from .halfstudentt import HalfStudentT
from .inversegamma import InverseGamma
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .logistic import Logistic
from .logitnormal import LogitNormal
from .lognormal import LogNormal
from .moyal import Moyal
from .normal import Normal
from .pareto import Pareto
from .skewnormal import SkewNormal
from .studentt import StudentT
from .rice import Rice
from .triangular import Triangular
from .truncatednormal import TruncatedNormal
from .uniform import Uniform
from .vonmises import VonMises
from .wald import Wald
from .weibull import Weibull


eps = np.finfo(float).eps


def from_precision(precision):
    sigma = 1 / precision**0.5
    return sigma


def to_precision(sigma):
    precision = 1 / sigma**2
    return precision


class BetaScaled(Continuous):
    r"""
    Scaled Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{(x-\text{lower})^{\alpha - 1} (\text{upper} - x)^{\beta - 1}}
           {(\text{upper}-\text{lower})^{\alpha+\beta-1} B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import BetaScaled
        az.style.use('arviz-doc')
        alphas = [2, 2]
        betas = [2, 5]
        lowers = [-0.5, -1]
        uppers = [1.5, 2]
        for alpha, beta, lower, upper in zip(alphas, betas, lowers, uppers):
            BetaScaled(alpha, beta, lower, upper).plot_pdf()

    ========  ==============================================================
    Support   :math:`x \in (lower, upper)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta} (upper-lower) + lower`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)} (upper-lower)`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, alpha=None, beta=None, lower=0, upper=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.dist = copy(stats.beta)
        self.support = (lower, upper)
        self._parametrization(self.alpha, self.beta, self.lower, self.upper)

    def _parametrization(self, alpha=None, beta=None, lower=None, upper=None):
        self.param_names = ("alpha", "beta", "lower", "upper")
        self.params_support = ((eps, np.inf), (eps, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta, lower, upper)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.alpha, self.beta, loc=self.lower, scale=self.upper - self.lower)
        return frozen

    def _update(self, alpha, beta, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta, self.lower, self.upper)
        self.support = self.lower, self.upper
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mean = (mean - self.lower) / (self.upper - self.lower)
        sigma = sigma / (self.upper - self.lower)
        kappa = mean * (1 - mean) / sigma**2 - 1
        alpha = max(0.5, kappa * mean)
        beta = max(0.5, kappa * (1 - mean))
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, beta, lower, scale = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta, lower, lower + scale)


class ExGaussian(Continuous):
    r"""
    Exponentially modified Gaussian (EMG) Distribution

    Results from the convolution of a normal distribution with an exponential
    distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \sigma, \tau) =
            \frac{1}{\nu}\;
            \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
            \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ExGaussian
        az.style.use('arviz-doc')
        mus = [0., 0., -3.]
        sigmas = [1., 3., 1.]
        nus = [1., 1., 4.]
        for mu, sigma, nu in zip(mus, sigmas, nus):
            ExGaussian(mu, sigma, nu).plot_pdf(support=(-6,9))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \nu`
    Variance  :math:`\sigma^2 + \nu^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution (sigma > 0).
    nu : float
        Mean of the exponential distribution (nu > 0).
    """

    def __init__(self, mu=None, sigma=None, nu=None):
        super().__init__()
        self.dist = copy(stats.exponnorm)
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, nu)

    def _parametrization(self, mu=None, sigma=None, nu=None):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.param_names = ("mu", "sigma", "nu")
        self.params = (mu, sigma, nu)
        #  if nu is too small we get a non-smooth distribution
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (1e-4, np.inf))
        if all_not_none(mu, sigma, nu):
            self._update(mu, sigma, nu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.nu, self.sigma):
            frozen = self.dist(K=self.nu / self.sigma, loc=self.mu, scale=self.sigma)
        return frozen

    def _update(self, mu, sigma, nu):
        self.nu = np.float64(nu)
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma, self.nu)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Just assume this is a approximately Gaussian
        self._update(mean, sigma, 1e-4)

    def _fit_mle(self, sample, **kwargs):
        K, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, K * sigma)
