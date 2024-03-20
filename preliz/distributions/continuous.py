# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init
"""
Continuous probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats
from scipy.special import gamma as gammaf
from scipy.special import beta as betaf  # pylint: disable=no-name-in-module
from scipy.special import logit, expit  # pylint: disable=no-name-in-module

from ..internal.optimization import optimize_ml, optimize_moments, optimize_moments_rice
from ..internal.distribution_helper import all_not_none, any_not_none
from .distributions import Continuous
from .asymmetric_laplace import AsymmetricLaplace  # pylint: disable=unused-import
from .beta import Beta  # pylint: disable=unused-import
from .exponential import Exponential  # pylint: disable=unused-import
from .normal import Normal  # pylint: disable=unused-import
from .halfnormal import HalfNormal  # pylint: disable=unused-import
from .laplace import Laplace  # pylint: disable=unused-import
from .weibull import Weibull  # pylint: disable=unused-import


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


class Cauchy(Continuous):
    r"""
    Cauchy Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \alpha, \beta) =
            \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Cauchy
        az.style.use('arviz-doc')
        alphas = [0., 0., -2.]
        betas = [.5, 1., 1.]
        for alpha, beta in zip(alphas, betas):
            Cauchy(alpha, beta).plot_pdf(support=(-5,5))

    ========  ==============================================================
    Support   :math:`x \in \mathbb{R}`
    Mean      undefined
    Variance  undefined
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        Location parameter.
    beta : float
        Scale parameter > 0.
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.dist = copy(stats.cauchy)
        self.support = (-np.inf, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ("alpha", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.params = (self.alpha, self.beta)
        if all_not_none(alpha, beta):
            self._update(alpha, beta)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.alpha, self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = mean
        beta = sigma
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class ChiSquared(Continuous):
    r"""
    Chi squared  distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) =
                \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ChiSquared
        az.style.use('arviz-doc')
        nus = [1., 3., 9.]
        for nu in nus:
                ax = ChiSquared(nu).plot_pdf(support=(0,20))
                ax.set_ylim(0, 0.6)

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu`
    Variance  :math:`2 \nu`
    ========  ===============================

    Parameters
    ----------
    nu : float
        Degrees of freedom (nu > 0).
    """

    def __init__(self, nu=None):
        super().__init__()
        self.nu = nu
        self.dist = copy(stats.chi2)
        self.support = (0, np.inf)
        self._parametrization(nu)

    def _parametrization(self, nu=None):
        self.nu = nu
        self.param_names = ("nu",)
        self.params_support = ((eps, np.inf),)
        self.params = (self.nu,)
        if self.nu is not None:
            self._update(self.nu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.nu)
        return frozen

    def _update(self, nu):
        self.nu = np.float64(nu)
        self.params = (self.nu,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        nu = mean
        self._update(nu)

    def _fit_mle(self, sample, **kwargs):
        nu, _, _ = self.dist.fit(sample, **kwargs)
        self._update(nu)


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


class Gamma(Continuous):
    r"""
    Gamma distribution.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has rate beta.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gamma
        az.style.use('arviz-doc')
        alphas = [1., 3., 7.5]
        betas = [.5, 1., 1.]
        for alpha, beta in zip(alphas, betas):
            Gamma(alpha, beta).plot_pdf()

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Gamma distribution has 2 alternative parameterizations. In terms of alpha and
    beta or mu (mean) and sigma (standard deviation).

    The link between the 2 alternatives is given by

    .. math::

       \alpha &= \frac{\mu^2}{\sigma^2} \\
       \beta  &= \frac{\mu}{\sigma^2}

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).
    mu : float
        Mean (mu > 0).
    sigma : float
        Standard deviation (sigma > 0)
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None):
        super().__init__()
        self.dist = copy(stats.gamma)
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, sigma)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None):
        if any_not_none(alpha, beta) and any_not_none(mu, sigma):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            if all_not_none(mu, sigma):
                alpha, beta = self._from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _from_mu_sigma(self, mu, sigma):
        alpha = mu**2 / sigma**2
        beta = mu / sigma**2
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        mu = alpha / beta
        sigma = alpha**0.5 / beta
        return mu, sigma

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(a=self.alpha, scale=1 / self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, 1 / beta)


class Gumbel(Continuous):
    r"""
    Gumbel distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gumbel
        az.style.use('arviz-doc')
        mus = [0., 4., -1.]
        betas = [1., 2., 4.]
        for mu, beta in zip(mus, betas):
            Gumbel(mu, beta).plot_pdf(support=(-10,20))

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \beta\gamma`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, mu=None, beta=None):
        super().__init__()
        self.dist = copy(stats.gumbel_r)
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, beta)

    def _parametrization(self, mu=None, beta=None):
        self.mu = mu
        self.beta = beta
        self.params = (self.mu, self.beta)
        self.param_names = ("mu", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(self.mu, self.beta):
            self._update(self.mu, self.beta)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(loc=self.mu, scale=self.beta)
        return frozen

    def _update(self, mu, beta):
        self.mu = np.float64(mu)
        self.beta = np.float64(beta)
        self.params = (self.mu, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        beta = sigma / np.pi * 6**0.5
        mu = mean - beta * np.euler_gamma
        self._update(mu, beta)

    def _fit_mle(self, sample, **kwargs):
        mu, beta = self.dist.fit(sample, **kwargs)
        self._update(mu, beta)


class HalfCauchy(Continuous):
    r"""
    HalfCauchy Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \beta) =
            \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfCauchy
        az.style.use('arviz-doc')
        for beta in [.5, 1., 2.]:
            HalfCauchy(beta).plot_pdf(support=(0,5))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      undefined
    Variance  undefined
    ========  ==========================================

    Parameters
    ----------
    beta : float
        Scale parameter :math:`\beta` (``beta`` > 0)
    """

    def __init__(self, beta=None):
        super().__init__()
        self.dist = copy(stats.halfcauchy)
        self.support = (0, np.inf)
        self._parametrization(beta)

    def _parametrization(self, beta=None):
        self.beta = beta
        self.params = (self.beta,)
        self.param_names = ("beta",)
        self.params_support = ((eps, np.inf),)
        if self.beta is not None:
            self._update(self.beta)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(scale=self.beta)
        return frozen

    def _update(self, beta):
        self.beta = np.float64(beta)
        self.params = (self.beta,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        beta = sigma
        self._update(beta)

    def _fit_mle(self, sample, **kwargs):
        _, beta = self.dist.fit(sample, **kwargs)
        self._update(beta)


class HalfStudentT(Continuous):
    r"""
    HalfStudentT Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfStudentT
        az.style.use('arviz-doc')
        sigmas = [1., 2., 2.]
        nus = [3, 3., 10.]
        for sigma, nu in zip(sigmas, nus):
            HalfStudentT(nu, sigma).plot_pdf(support=(0,10))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      .. math::
                  2\sigma\sqrt{\frac{\nu}{\pi}}\
                  \frac{\Gamma\left(\frac{\nu+1}{2}\right)}\
                  {\Gamma\left(\frac{\nu}{2}\right)(\nu-1)}\, \text{for } \nu > 2
    Variance  .. math::
                  \sigma^2\left(\frac{\nu}{\nu - 2}-\
                  \frac{4\nu}{\pi(\nu-1)^2}\left(\frac{\Gamma\left(\frac{\nu+1}{2}\right)}\
                  {\Gamma\left(\frac{\nu}{2}\right)}\right)^2\right) \text{for } \nu > 2\, \infty\
                  \text{for } 1 < \nu \le 2\, \text{otherwise undefined}
    ========  ==========================================

    HalfStudentT distribution has 2 alternative parameterizations. In terms of nu and
    sigma (standard deviation as nu increases) or nu lam (precision as nu increases).

    The link between the 2 alternatives is given by

    .. math::

        \lambda = \frac{1}{\sigma^2}

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu increases.
    """

    def __init__(self, nu=None, sigma=None, lam=None):
        super().__init__()
        self.dist = _HalfStudentT
        self.support = (0, np.inf)
        self._parametrization(nu, sigma, lam)

    def _parametrization(self, nu=None, sigma=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError(
                "Incompatible parametrization. Either use nu and sigma, or nu and lam."
            )

        self.param_names = ("nu", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("nu", "lam")

        self.nu = nu
        self.sigma = sigma
        if all_not_none(self.nu, self.sigma):
            self._update(self.nu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(nu=self.nu, sigma=self.sigma)
        return frozen

    def _update(self, nu, sigma):
        self.nu = np.float64(nu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.nu, self.sigma)
        elif self.param_names[1] == "lam":
            self.params = (self.nu, self.lam)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu is None:
            nu = 100
        elif nu <= 2:
            nu = 2.1

        gamma0 = gammaf((nu + 1) / 2)
        gamma1 = gammaf(nu / 2)
        if np.isfinite(gamma0) and np.isfinite(gamma1):
            sigma = (
                sigma**2
                / ((nu / (nu - 2)) - ((4 * nu) / (np.pi * (nu - 1) ** 2)) * (gamma0 / gamma1) ** 2)
            ) ** 0.5
        else:
            # we assume a Gaussian for large nu
            sigma = sigma / (1 - 2 / np.pi) ** 0.5
        self._update(nu, sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample)


class _HalfStudentT(stats.rv_continuous):
    def __init__(self, nu=None, sigma=None):
        super().__init__()
        self.nu = nu
        self.sigma = sigma
        self.dist = stats.t(loc=0, df=self.nu, scale=self.sigma)

    def support(self, *args, **kwd):  # pylint: disable=unused-argument
        return (0, np.inf)

    def cdf(self, x, *args, **kwds):
        return np.maximum(0, self.dist.cdf(x, *args, **kwds) * 2 - 1)

    def pdf(self, x, *args, **kwds):
        return np.where(x < 0, -np.inf, self.dist.pdf(x, *args, **kwds) * 2)

    def logpdf(self, x, *args, **kwds):
        return np.where(x < 0, -np.inf, self.dist.logpdf(x, *args, **kwds) + np.log(2))

    def ppf(self, q, *args, **kwds):
        x_vals = np.linspace(0, self.rvs(10000).max(), 1000)
        idx = np.searchsorted(self.cdf(x_vals[:-1], *args, **kwds), q)
        return x_vals[idx]

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        mean = np.nan
        var = np.nan
        skew = np.nan
        kurtosis = np.nan

        if self.nu > 1:
            gamma0 = gammaf((self.nu + 1) / 2)
            gamma1 = gammaf(self.nu / 2)
            if np.isfinite(gamma0) and np.isfinite(gamma1):
                mean = (
                    2 * self.sigma * (self.nu / np.pi) ** 0.5 * (gamma0 / (gamma1 * (self.nu - 1)))
                )
            else:
                # assume nu is large enough that the mean of the halfnormal is a good approximation
                mean = self.sigma * (2 / np.pi) ** 0.5
        if self.nu > 2:
            if np.isfinite(gamma0) and np.isfinite(gamma1):
                var = self.sigma**2 * (
                    (self.nu / (self.nu - 2))
                    - ((4 * self.nu) / (np.pi * (self.nu - 1) ** 2)) * (gamma0 / gamma1) ** 2
                )
            else:
                # assume nu is large enough that the std of the halfnormal is a good approximation
                var = self.sigma**2 * (1 - 2.0 / np.pi)

        return (mean, var, skew, kurtosis)

    def entropy(self):  # pylint: disable=arguments-differ
        return self.dist.entropy() - np.log(2)

    def rvs(self, size=1, random_state=None):  # pylint: disable=arguments-differ
        return np.abs(self.dist.rvs(size=size, random_state=random_state))


class InverseGamma(Continuous):
    r"""
    Inverse gamma distribution, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import InverseGamma
        az.style.use('arviz-doc')
        alphas = [1., 2., 3.]
        betas = [1., 1., .5]
        for alpha, beta in zip(alphas, betas):
            InverseGamma(alpha, beta).plot_pdf(support=(0, 3))

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}` for :math:`\alpha > 2`
    ========  ===============================

    Inverse gamma distribution has 2 alternative parameterizations. In terms of alpha and
    beta or mu (mean) and sigma (standard deviation).

    The link between the 2 alternatives is given by

    .. math::

       \alpha &= \frac{\mu^2}{\sigma^2} + 2 \\
       \beta  &= \frac{\mu^3}{\sigma^2} + \mu

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    mu : float
        Mean (mu > 0).
    sigma : float
        Standard deviation (sigma > 0)
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None):
        super().__init__()
        self.dist = copy(stats.invgamma)
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, sigma)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None):
        if any_not_none(alpha, beta) and any_not_none(mu, sigma):
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            if all_not_none(mu, sigma):
                alpha, beta = self._from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _from_mu_sigma(self, mu, sigma):
        alpha = mu**2 / sigma**2 + 2
        beta = mu**3 / sigma**2 + mu
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        if alpha > 1:
            mu = beta / (alpha - 1)
        else:
            mu = None
        if alpha > 2:
            sigma = beta / ((alpha - 1) * (alpha - 2) ** 0.5)
        else:
            sigma = None
        return mu, sigma

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(a=self.alpha, scale=self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class Kumaraswamy(Continuous):
    r"""
    Kumaraswamy distribution.

    The pdf of this distribution is

    .. math::

         f(x \mid a, b) = a b x^{a - 1} (1 - x^a)^{b - 1}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Kumaraswamy
        az.style.use('arviz-doc')
        a_s = [.5, 5., 1., 2., 2.]
        b_s = [.5, 1., 3., 2., 5.]
        for a, b in zip(a_s, b_s):
            ax = Kumaraswamy(a, b).plot_pdf()
            ax.set_ylim(0, 3.)

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
    Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
    ========  ==============================================================

    Parameters
    ----------
    a : float
        a > 0.
    b : float
        b > 0.
    """

    def __init__(self, a=None, b=None):
        super().__init__()
        self.dist = _Kumaraswamy
        self.support = (0, 1)
        self._parametrization(a, b)

    def _parametrization(self, a=None, b=None):
        self.a = a
        self.b = b
        self.params = (self.a, self.b)
        self.param_names = ("a", "b")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if (a and b) is not None:
            self._update(a, b)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.a, self.b)
        return frozen

    def _update(self, a, b):
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.params = (self.a, self.b)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample, **kwargs)


class _Kumaraswamy(stats.rv_continuous):
    def __init__(self, a=None, b=None):
        super().__init__()
        self.a = a
        self.b = b

    def support(self, *args, **kwds):  # pylint: disable=unused-argument
        return (0, 1)

    def cdf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        return 1 - (1 - x**self.a) ** self.b

    def pdf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        return (self.a * self.b * x ** (self.a - 1)) * ((1 - x**self.a) ** (self.b - 1))

    def logpdf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        return (
            np.log(self.a * self.b)
            + (self.a - 1) * np.log(x)
            + (self.b - 1) * np.log(1 - x**self.a)
        )

    def ppf(self, q, *args, **kwds):  # pylint: disable=unused-argument
        return (1 - (1 - q) ** (1 / self.b)) ** (1 / self.a)

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        mean = self.b * betaf(1 + 1 / self.a, self.b)
        var = self.b * betaf(1 + 2 / self.a, self.b) - self.b * betaf(1 + 2 / self.a, self.b) ** 2
        return (mean, var, np.nan, np.nan)

    def entropy(self, *args, **kwds):  # pylint: disable=unused-argument
        # https://www.ijicc.net/images/vol12/iss4/12449_Nassir_2020_E_R.pdf
        return (
            (1 - 1 / self.b)
            + (1 - 1 / self.a) * sum(1 / i for i in range(1, int(self.b) + 1))
            - np.log(self.a * self.b)
        )

    def rvs(self, size=1, random_state=None):  # pylint: disable=arguments-differ
        if random_state is None:
            q = np.random.rand(size)
        elif isinstance(random_state, int):
            q = np.random.default_rng(random_state).random(size)
        else:
            q = random_state.random(size)

        return self.ppf(q)


class Logistic(Continuous):
    r"""
    Logistic distribution.

    The pdf of this distribution is

    .. math::

    f(x \mid \mu, s) =
        \frac{\exp\left(-\frac{x - \mu}{s}\right)}
        {s \left(1 + \exp\left(-\frac{x - \mu}{s}\right)\right)^2}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Logistic
        az.style.use('arviz-doc')
        mus = [0., 0., -2.]
        ss = [1., 2., .4]
        for mu, s in zip(mus, ss):
            Logistic(mu, s).plot_pdf(support=(-5,5))

    =========  ==========================================
     Support   :math:`x \in \mathbb{R}`
     Mean      :math:`\mu`
     Variance  :math:`\frac{s^2 \pi^2}{3}`
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
        self.dist = copy(stats.logistic)
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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(loc=self.mu, scale=self.s)
        return frozen

    def _update(self, mu, s):
        self.mu = np.float64(mu)
        self.s = np.float64(s)
        self.params = (self.mu, self.s)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = mean
        s = (3 * sigma**2 / np.pi**2) ** 0.5
        self._update(mu, s)

    def _fit_mle(self, sample, **kwargs):
        mu, s = self.dist.fit(sample, **kwargs)
        self._update(mu, s)


class LogNormal(Continuous):
    r"""
    Log-normal distribution.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma) =
          \frac{1}{x \sigma \sqrt{2\pi}}
           \exp\left\{ -\frac{(\ln(x)-\mu)^2}{2\sigma^2} \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import LogNormal
        az.style.use('arviz-doc')
        mus = [ 0., 0.]
        sigmas = [.5, 1.]
        for mu, sigma in zip(mus, sigmas):
            LogNormal(mu, sigma).plot_pdf(support=(0,5))

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\exp\left(\mu+\frac{\sigma^2}{2}\right)`
    Variance  :math:`[\exp(\sigma^2)-1] \exp(2\mu+\sigma^2)`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Standard deviation. (sigma > 0)).
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.dist = copy(stats.lognorm)
        self.support = (0, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, sigma):
            self._update(mu, sigma)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.sigma, scale=np.exp(self.mu))
        return frozen

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def _fit_mle(self, sample, **kwargs):
        sigma, _, mu = self.dist.fit(sample, **kwargs)
        self._update(np.log(mu), sigma)


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

        import arviz as az
        from preliz import LogitNormal
        az.style.use('arviz-doc')
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
        self.dist = _LogitNormal
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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = logit(mean)
        sigma = np.diff((mean - sigma * 3, mean + sigma * 3))
        self._update(mu, sigma)

    def _fit_mle(self, sample, **kwargs):
        mu, sigma = stats.norm.fit(logit(sample), **kwargs)
        self._update(mu, sigma)


class _LogitNormal(stats.rv_continuous):
    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def support(self, *args, **kwds):  # pylint: disable=unused-argument
        return (0, 1)

    def cdf(self, x, *args, **kwds):
        return stats.norm(self.mu, self.sigma, *args, **kwds).cdf(logit(x))

    def pdf(self, x, *args, **kwds):
        x = np.asarray(x)
        mask = np.logical_or(x == 0, x == 1)
        result = np.zeros_like(x, dtype=float)
        result[~mask] = stats.norm(self.mu, self.sigma, *args, **kwds).pdf(logit(x[~mask])) / (
            x[~mask] * (1 - x[~mask])
        )
        return result

    def logpdf(self, x, *args, **kwds):
        x = np.asarray(x)
        mask = np.logical_or(x == 0, x == 1)
        result = np.full_like(x, -np.inf, dtype=float)
        result[~mask] = (
            stats.norm(self.mu, self.sigma, *args, **kwds).logpdf(logit(x[~mask]))
            - np.log(x[~mask])
            - np.log1p(-x[~mask])
        )
        return result

    def ppf(self, q, *args, **kwds):
        x_vals = np.linspace(0, 1, 1000)
        idx = np.searchsorted(self.cdf(x_vals[:-1], *args, **kwds), q)
        return x_vals[idx]

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        # https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
        norm = stats.norm(self.mu, self.sigma)
        logistic_inv = expit(norm.ppf(np.linspace(0, 1, 100000)))
        mean = np.mean(logistic_inv)
        var = np.var(logistic_inv)
        return (mean, var, np.nan, np.nan)

    def entropy(self):  # pylint: disable=arguments-differ
        moments = self._stats()
        return stats.norm(moments[0], moments[1] ** 0.5).entropy()

    def rvs(
        self, size=1, random_state=None
    ):  # pylint: disable=arguments-differ, disable=unused-argument
        return expit(np.random.normal(self.mu, self.sigma, size))


class Moyal(Continuous):
    r"""
    Moyal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu,\sigma) =
            \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(z + e^{-z}\right)},

    where

    .. math::

       z = \frac{x-\mu}{\sigma}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Moyal
        az.style.use('arviz-doc')
        mus = [-1., 0., 4.]
        sigmas = [2., 1., 4.]
        for mu, sigma in zip(mus, sigmas):
            Moyal(mu, sigma).plot_pdf(support=(-10,20))

    ========  ==============================================================
    Support   :math:`x \in (-\infty, \infty)`
    Mean      :math:`\mu + \sigma\left(\gamma + \log 2\right)`, where
              :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^{2}}{2}\sigma^{2}`
    ========  ==============================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.dist = copy(stats.moyal)
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if all_not_none(mu, sigma):
            self._update(self.mu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(loc=self.mu, scale=self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        sigma = sigma / np.pi * 2**0.5
        mu = mean - sigma * (np.euler_gamma + np.log(2))
        self._update(mu, sigma)

    def _fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


class Pareto(Continuous):
    r"""
    Pareto distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Pareto
        az.style.use('arviz-doc')
        alphas = [1., 5., 5.]
        ms = [1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            Pareto(alpha, m).plot_pdf(support=(0,4))

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m \alpha}{(\alpha - 1)^2 (\alpha - 2)}` for :math:`\alpha > 2`
    ========  =============================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    m : float
        Scale parameter (m > 0).
    """

    def __init__(self, alpha=None, m=None):
        super().__init__()
        self.dist = copy(stats.pareto)
        self.support = (0, np.inf)
        self._parametrization(alpha, m)

    def _parametrization(self, alpha=None, m=None):
        self.alpha = alpha
        self.m = m
        self.params = (self.alpha, self.m)
        self.param_names = ("alpha", "m")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, m):
            self._update(alpha, m)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.alpha, scale=self.m)
        return frozen

    def _update(self, alpha, m):
        self.alpha = np.float64(alpha)
        self.m = np.float64(m)
        self.support = (self.m, np.inf)
        self.params = (self.alpha, self.m)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = 1 + (1 + (mean / sigma) ** 2) ** (1 / 2)
        m = (alpha - 1) * mean / alpha
        self._update(alpha, m)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, m = self.dist.fit(sample, **kwargs)
        self._update(alpha, m)


class Rice(Continuous):
    r"""
    Rice distribution.

    The pdf of this distribution is

    .. math::

        f(x\mid \nu ,\sigma )=
            {\frac  {x}{\sigma ^{2}}}\exp
            \left({\frac  {-(x^{2}+\nu ^{2})}
            {2\sigma ^{2}}}\right)I_{0}\left({\frac  {x\nu }{\sigma ^{2}}}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Rice
        az.style.use('arviz-doc')
        nus = [0., 0., 4.]
        sigmas = [1., 2., 2.]
        for nu, sigma in  zip(nus, sigmas):
            Rice(nu, sigma).plot_pdf(support=(0,10))

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma {\sqrt  {\pi /2}}\,\,L_{{1/2}}(-\nu ^{2}/2\sigma ^{2})`
    Variance  :math:`2\sigma ^{2}+\nu ^{2}-{\frac  {\pi \sigma ^{2}}{2}}L_{{1/2}}^{2}
                        \left({\frac  {-\nu ^{2}}{2\sigma ^{2}}}\right)`
    ========  ==============================================================

    Rice distribution has 2 alternative parameterizations. In terms of nu and sigma
    or b and sigma.

    The link between the two parametrizations is given by

    .. math::

       b = \dfrac{\nu}{\sigma}

    Parameters
    ----------
    nu : float
        Noncentrality parameter.
    sigma : float
        Scale parameter.
    b : float
        Shape parameter.
    """

    def __init__(self, nu=None, sigma=None, b=None):
        super().__init__()
        self.name = "rice"
        self.dist = copy(stats.rice)
        self.support = (0, np.inf)
        self._parametrization(nu, sigma, b)

    def _parametrization(self, nu=None, sigma=None, b=None):
        if all_not_none(nu, b):
            raise ValueError(
                "Incompatible parametrization. Either use nu and sigma or b and sigma."
            )

        self.param_names = ("nu", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if b is not None:
            self.b = b
            self.sigma = sigma
            self.param_names = ("b", "sigma")
            if all_not_none(b, sigma):
                nu = self._from_b(b, sigma)

        self.nu = nu
        self.sigma = sigma
        if all_not_none(self.nu, self.sigma):
            self._update(self.nu, self.sigma)

    def _from_b(self, b, sigma):
        nu = b * sigma
        return nu

    def _to_b(self, nu, sigma):
        b = nu / sigma
        return b

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            b_ = self._to_b(self.nu, self.sigma)
            frozen = self.dist(b=b_, scale=self.sigma)
        return frozen

    def _update(self, nu, sigma):
        self.nu = np.float64(nu)
        self.sigma = np.float64(sigma)
        self.b = self._to_b(self.nu, self.sigma)

        if self.param_names[0] == "nu":
            self.params = (self.nu, self.sigma)
        elif self.param_names[0] == "b":
            self.params = (self.b, self.sigma)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        nu, sigma = optimize_moments_rice(mean, sigma)
        self._update(nu, sigma)

    def _fit_mle(self, sample, **kwargs):
        b, _, sigma = self.dist.fit(sample, **kwargs)
        nu = self._from_b(b, sigma)
        self._update(nu, sigma)


class SkewNormal(Continuous):
    r"""
    SkewNormal distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \tau, \alpha) =
        2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import SkewNormal
        az.style.use('arviz-doc')
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
        self.dist = copy(stats.skewnorm)
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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.alpha, self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma, alpha):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.alpha = np.float64(alpha)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma, self.alpha)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau, self.alpha)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Assume gaussian
        self._update(mean, sigma, 0)

    def _fit_mle(self, sample, **kwargs):
        alpha, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, alpha)


class StudentT(Continuous):
    r"""
    StudentT's distribution.

    Describes a normal variable whose precision is gamma distributed.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu, \mu, \sigma) =
           \frac{\Gamma \left(\frac{\nu+1}{2} \right)} {\sqrt{\nu\pi}\
           \Gamma \left(\frac{\nu}{2} \right)} \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import StudentT
        az.style.use('arviz-doc')
        nus = [2., 5., 5.]
        mus = [0., 0.,  -4.]
        sigmas = [1., 1., 2.]
        for nu, mu, sigma in zip(nus, mus, sigmas):
            StudentT(nu, mu, sigma).plot_pdf(support=(-10,6))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu` for :math:`\nu > 1`, otherwise undefined
    Variance  :math:`\frac{\nu}{\nu-2}` for :math:`\nu > 2`,
              :math:`\infty` for :math:`1 < \nu \le 2`, otherwise undefined
    ========  ========================

    StudentT distribution has 2 alternative parameterizations. In terms of nu, mu and
    sigma (standard deviation as nu increases) or nu, mu and lam (precision as nu increases).

    The link between the 2 alternatives is given by

    .. math::

        \lambda = \frac{1}{\sigma^2}

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu increases.
    """

    def __init__(self, nu=None, mu=None, sigma=None, lam=None):
        super().__init__()
        self.dist = copy(stats.t)
        self.support = (-np.inf, np.inf)
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))
        self._parametrization(nu, mu, sigma, lam)

    def _parametrization(self, nu=None, mu=None, sigma=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError(
                "Incompatible parametrization. Either use nu, mu and sigma, or nu, mu and lam."
            )

        self.param_names = ("nu", "mu", "sigma")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))

        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("nu", "mu", "lam")

        self.nu = nu
        self.mu = mu
        self.sigma = sigma

        if all_not_none(self.nu, self.mu, self.sigma):
            self._update(self.nu, self.mu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.nu, self.mu, self.sigma)
        return frozen

    def _update(self, nu, mu, sigma):
        self.nu = np.float64(nu)
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(sigma)

        if self.param_names[2] == "sigma":
            self.params = (self.nu, self.mu, self.sigma)
        elif self.param_names[2] == "lam":
            self.params = (self.nu, self.mu, self.lam)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu is None:
            nu = 100
        elif nu <= 2:
            nu = 2.1
        else:
            sigma = sigma / (nu / (nu - 2)) ** 0.5

        self._update(nu, mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        nu, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(nu, mu, sigma)


class Triangular(Continuous):
    r"""
    Triangular distribution

    The pdf of this distribution is

    .. math::

       \begin{cases}
         0 & \text{for } x < a, \\
         \frac{2(x-a)}{(b-a)(c-a)} & \text{for } a \le x < c, \\[4pt]
         \frac{2}{b-a}             & \text{for } x = c, \\[4pt]
         \frac{2(b-x)}{(b-a)(b-c)} & \text{for } c < x \le b, \\[4pt]
         0 & \text{for } b < x.
        \end{cases}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Triangular
        az.style.use('arviz-doc')
        lowers = [0., -1, 2]
        cs = [2., 0., 6.5]
        uppers = [4., 1, 8]
        for lower, c, upper in zip(lowers, cs, uppers):
            scale = upper - lower
            c_ = (c - lower) / scale
            Triangular(lower, c, upper).plot_pdf()

    ========  ============================================================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper + c}{3}`
    Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
    ========  ============================================================================

    Parameters
    ----------
    lower : float
        Lower limit.
    c : float
        Mode.
    upper : float
        Upper limit.
    """

    def __init__(self, lower=None, c=None, upper=None):
        super().__init__()
        self.dist = copy(stats.triang)
        self.support = (-np.inf, np.inf)
        self._parametrization(lower, c, upper)

    def _parametrization(self, lower=None, c=None, upper=None):
        self.lower = lower
        self.c = c
        self.upper = upper
        self.params = (self.lower, self.c, self.upper)
        self.param_names = ("lower", "c", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if all_not_none(lower, c, upper):
            self._update(lower, c, upper)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            scale = self.upper - self.lower
            c_ = (self.c - self.lower) / scale
            frozen = self.dist(c=c_, loc=self.lower, scale=scale)
        return frozen

    def _update(self, lower, c, upper):
        self.lower = np.float64(lower)
        self.c = np.float64(c)
        self.upper = np.float64(upper)
        self.support = (self.lower, self.upper)
        self.params = (self.lower, self.c, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Assume symmetry
        lower = mean - 6**0.5 * sigma
        upper = mean + 6**0.5 * sigma
        c = mean
        self._update(lower, c, upper)

    def _fit_mle(self, sample, **kwargs):
        c_, lower, scale = self.dist.fit(sample, **kwargs)
        upper = scale + lower
        c = c_ * scale + lower
        self._update(lower, c, upper)


class TruncatedNormal(Continuous):
    r"""
    TruncatedNormal distribution.

    The pdf of this distribution is

    .. math::

       f(x;\mu ,\sigma ,a,b)={\frac {\phi ({\frac {x-\mu }{\sigma }})}{
            \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import TruncatedNormal
        az.style.use('arviz-doc')
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        lowers = [-3, -5, -5]
        uppers = [7, 5, 4]
        for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
            TruncatedNormal(mu, sigma, lower, upper).plot_pdf(support=(-10,10))

    ========  ==========================================
    Support   :math:`x \in [a, b]`
    Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
    Variance  :math:`\sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-
                        \left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]`
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
        self.dist = copy(stats.truncnorm)
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

    def _get_frozen(self):
        frozen = None
        if any_not_none(self.mu, self.sigma):
            a, b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
            frozen = self.dist(a, b, self.mu, self.sigma)
            frozen.entropy = self._entropy
        return frozen

    def _update(self, mu, sigma, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Assume gaussian
        self._update(mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        a, b, mu, sigma = self.dist.fit(sample, **kwargs)
        lower, upper = a * sigma + mu, b * sigma + mu
        self._update(mu, sigma, lower, upper)

    def _entropy(self):
        "Override entropy to handle lower or upper infinite values"
        norm = stats.norm
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        zed = norm.cdf(beta) - norm.cdf(alpha)

        if np.isfinite(alpha):
            a_pdf = alpha * norm.pdf(alpha)
        else:
            a_pdf = 0

        if np.isfinite(beta):
            b_pdf = beta * norm.pdf(beta)
        else:
            b_pdf = 0

        return np.log(4.132731354122493 * zed * self.sigma) + (a_pdf - b_pdf) / (2 * zed)


class Uniform(Continuous):
    r"""
    Uniform distribution.

    The pdf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Uniform
        az.style.use('arviz-doc')
        ls = [1, -2]
        us = [6, 2]
        for l, u in zip(ls, us):
            ax = Uniform(l, u).plot_pdf()
        ax.set_ylim(0, 0.3)

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  =====================================

    Parameters
    ----------
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.dist = copy(stats.uniform)
        self._parametrization(lower, upper)

    def _parametrization(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self.params = (self.lower, self.upper)
        self.param_names = ("lower", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf))
        if lower is None:
            self.lower = -np.inf
        if upper is None:
            self.upper = np.inf
        self.support = (self.lower, self.upper)
        if all_not_none(lower, upper):
            self._update(lower, upper)
            self.dist.a = self.lower
            self.dist.b = self.upper
        else:
            self.lower = lower
            self.upper = upper

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.lower, self.upper - self.lower)
        return frozen

    def _update(self, lower, upper):
        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
        self.params = (self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        lower = mean - 1.73205 * sigma
        upper = mean + 1.73205 * sigma
        self._update(lower, upper)

    def _fit_mle(self, sample, **kwargs):
        lower = np.min(sample)
        upper = np.max(sample)
        self._update(lower, upper)


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

        import arviz as az
        from preliz import VonMises
        az.style.use('arviz-doc')
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
        Concentration (:math:`\frac{1}{\kappa}` is analogous to :math:`\sigma^2`).
    """

    def __init__(self, mu=None, kappa=None):
        super().__init__()
        self.dist = copy(stats.vonmises)
        self._parametrization(mu, kappa)

    def _parametrization(self, mu=None, kappa=None):
        self.mu = mu
        self.kappa = kappa
        self.param_names = ("mu", "kappa")
        self.params_support = ((-np.pi, np.pi), (eps, np.inf))
        self.support = (-np.pi, np.pi)
        if all_not_none(mu, kappa):
            self._update(mu, kappa)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(kappa=self.kappa, loc=self.mu)
        return frozen

    def _update(self, mu, kappa):
        self.mu = np.float64(mu)
        self.kappa = np.float64(kappa)
        self.params = (self.mu, self.kappa)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = mean
        kappa = 1 / sigma**2
        self._update(mu, kappa)

    def _fit_mle(self, sample, **kwargs):
        kappa, mu, _ = self.dist.fit(sample, **kwargs)
        self._update(mu, kappa)


class Wald(Continuous):
    r"""
    Wald distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Wald
        az.style.use('arviz-doc')
        mus = [1., 1.]
        lams = [1., 3.]
        for mu, lam in zip(mus, lams):
            Wald(mu, lam).plot_pdf(support=(0,4))

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Wald distribution has 3 alternative parametrizations. In terms of mu and lam,
    mu and phi or lam and phi.

    The link between the 3 alternatives is given by

    .. math::

       \phi = \dfrac{\lambda}{\mu}

    Parameters
    ----------
    mu : float
        Mean of the distribution (mu > 0).
    lam : float
        Relative precision (lam > 0).
    phi : float
        Shape parameter (phi > 0).
    """

    def __init__(self, mu=None, lam=None, phi=None):
        super().__init__()
        self.dist = copy(stats.invgauss)
        self.support = (0, np.inf)
        self._parametrization(mu, lam, phi)

    def _parametrization(self, mu=None, lam=None, phi=None):
        if (mu and lam and phi) is not None:
            raise ValueError(
                "Incompatible parametrization. Either use mu and lam or mu and phi or lam and phi."
            )

        self.param_names = ("mu", "lam")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if phi is not None:
            self.phi = phi
            if (mu and phi) is not None:
                lam = self._from_mu_phi(mu, phi)
                self.param_names = ("mu", "phi")

            elif (lam and phi) is not None:
                mu = self._from_lam_phi(lam, phi)
                self.param_names = ("lam", "phi")

        self.mu = mu
        self.lam = lam
        if all_not_none(self.mu, self.lam):
            self._update(self.mu, self.lam)

    def _from_mu_phi(self, mu, phi):
        lam = mu * phi
        return lam

    def _from_lam_phi(self, lam, phi):
        mu = lam / phi
        return mu

    def _to_phi(self, mu, lam):
        phi = lam / mu
        return phi

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.mu / self.lam, scale=self.lam)
        return frozen

    def _update(self, mu, lam):
        self.mu = np.float64(mu)
        self.lam = np.float64(lam)
        self.phi = self._to_phi(mu, lam)

        if self.param_names == ("mu", "lam"):
            self.params = (self.mu, self.lam)
        elif self.param_names == ("mu", "phi"):
            self.params = (self.mu, self.phi)
        elif self.param_names == ("lam", "phi"):
            self.params = (self.lam, self.phi)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        lam = mean**3 / sigma**2
        self._update(mean, lam)

    def _fit_mle(self, sample, **kwargs):
        mu, _, lam = self.dist.fit(sample, **kwargs)
        self._update(mu * lam, lam)
