# pylint: disable=too-many-lines
"""
Continuous probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats
from scipy.special import gamma as gammaf


from .distributions import Continuous
from ..utils.utils import garcia_approximation
from ..utils.optimization import optimize_ml

eps = np.finfo(float).eps


class Beta(Continuous):
    r"""
    Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Beta
        az.style.use('arviz-white')
        alphas = [.5, 5., 2.]
        betas = [.5, 5., 5.]
        for alpha, beta in zip(alphas, betas):
            Beta(alpha, beta).plot_pdf()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "beta"
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.beta
        self.support = (0, 1)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        kappa = mean * (1 - mean) / sigma**2 - 1
        alpha = max(0.5, kappa * mean)
        beta = max(0.5, kappa * (1 - mean))
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, beta, _, _ = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


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
        az.style.use('arviz-white')
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
        self.name = "betascaled"
        self.params = (self.alpha, self.beta, self.lower, self.upper)
        self.param_names = ("alpha", "beta", "lower", "upper")
        self.params_support = ((eps, np.inf), (eps, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        self.dist = copy(stats.beta)
        self.support = (lower, upper)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if self.alpha is not None or self.beta is not None:
            frozen = self.dist(self.alpha, self.beta, loc=self.lower, scale=self.upper - self.lower)
        return frozen

    def _update(self, alpha, beta, lower=None, upper=None):
        if lower is not None:
            self.lower = lower
        if upper is not None:
            self.upper = upper

        self.alpha = alpha
        self.beta = beta
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
        az.style.use('arviz-white')
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
        self.alpha = alpha
        self.beta = beta
        self.name = "cauchy"
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.cauchy
        self.support = (-np.inf, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = mean
        beta = sigma
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class Exponential(Continuous):
    r"""
    Exponential Distribution

    The pdf of this distribution is

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Exponential
        az.style.use('arviz-white')
        for lam in [0.5,  2.]:
            Exponential(lam).plot_pdf()


    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Parameters
    ----------
    lam : float
        Rate or inverse scale (lam > 0).
    """

    def __init__(self, lam=None):
        super().__init__()
        self.lam = lam
        self.name = "exponential"
        self.params = (self.lam,)
        self.param_names = ("lam",)
        self.params_support = ((eps, np.inf),)
        self.dist = stats.expon
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(scale=1 / self.lam)
        return frozen

    def _update(self, lam):
        self.lam = lam
        self.params = (self.lam,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        lam = 1 / mean
        self._update(lam)

    def _fit_mle(self, sample, **kwargs):
        _, lam = self.dist.fit(sample, **kwargs)
        self._update(1 / lam)


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
        az.style.use('arviz-white')
        alphas = [1., 3., 7.5]
        betas = [.5, 1., 1.]
        for alpha, beta in zip(alphas, betas):
            Gamma(alpha, beta).plot_pdf()

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "gamma"
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.gamma
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(a=self.alpha, scale=1 / self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = mean**2 / sigma**2
        beta = mean / sigma**2
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, 1 / beta)


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
        az.style.use('arviz-white')
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
        self.beta = beta
        self.name = "halfcauchy"
        self.params = (self.beta,)
        self.param_names = ("beta",)
        self.params_support = ((eps, np.inf),)
        self.dist = stats.halfcauchy
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(scale=self.beta)
        return frozen

    def _update(self, beta):
        self.beta = beta
        self.params = (self.beta,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        beta = sigma
        self._update(beta)

    def _fit_mle(self, sample, **kwargs):
        _, beta = self.dist.fit(sample, **kwargs)
        self._update(beta)


class HalfNormal(Continuous):
    r"""
    HalfNormal Distribution

    The pdf of this distribution is

    .. math::

       f(x \mid \sigma) =
           \sqrt{\frac{2}{\pi\sigma^2}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfNormal
        az.style.use('arviz-white')
        for sigma in [0.4,  2.]:
            HalfNormal(sigma).plot_pdf()


    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`\sigma` (``sigma`` > 0)
    """

    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma
        self.name = "halfnormal"
        self.params = (self.sigma,)
        self.param_names = ("sigma",)
        self.params_support = ((eps, np.inf),)
        self.dist = stats.halfnorm
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(scale=self.sigma)
        return frozen

    def _update(self, sigma):
        self.sigma = sigma
        self.params = (self.sigma,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        sigma = sigma / (1 - 2 / np.pi) ** 0.5
        self._update(sigma)

    def _fit_mle(self, sample, **kwargs):
        _, sigma = self.dist.fit(sample, **kwargs)
        self._update(sigma)


class HalfStudent(Continuous):
    r"""
    HalfStudent Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfStudent
        az.style.use('arviz-white')
        sigmas = [1., 2., 2.]
        nus = [3, 3., 10.]
        for sigma, nu in zip(sigmas, nus):
            HalfStudent(nu, sigma).plot_pdf()

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

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    """

    def __init__(self, nu=3, sigma=None):
        super().__init__()
        self.nu = nu
        self.sigma = sigma
        self.name = "halfstudent"
        self.params = (self.nu, self.sigma)
        self.param_names = ("nu", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = _HalfStudent
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(nu=self.nu, sigma=self.sigma)
        return frozen

    def _update(self, sigma, nu=None):
        if nu is not None:
            self.nu = nu
        self.sigma = sigma
        self.params = (self.nu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu <= 2:
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
        self._update(sigma)

    def _fit_mle(self, sample, **kwargs):
        optimize_ml(self, sample)


class _HalfStudent(stats._distn_infrastructure.rv_continuous):
    def __init__(self, nu=2, sigma=1):
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
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import InverseGamma
        az.style.use('arviz-white')
        alphas = [1., 2., 3.]
        betas = [1., 1., .5]
        for alpha, beta in zip(alphas, betas):
            InverseGamma(alpha, beta).plot_pdf(support=(0, 3))

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}` for :math:`\alpha > 2`
    ========  ===============================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "inversegamma"
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.invgamma
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(a=self.alpha, scale=self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = (mean**2 / sigma**2) + 2
        beta = (mean**3 / sigma**2) + mean
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class Laplace(Continuous):
    r"""
    Laplace distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Laplace
        az.style.use('arviz-white')
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            Laplace(mu, b).plot_pdf()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    """

    def __init__(self, mu=None, b=None):
        super().__init__()
        self.mu = mu
        self.b = b
        self.name = "laplace"
        self.params = (self.mu, self.b)
        self.param_names = ("mu", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.laplace
        self.support = (-np.inf, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(loc=self.mu, scale=self.b)
        return frozen

    def _update(self, mu, b):
        self.mu = mu
        self.b = b
        self.params = (self.mu, self.b)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = mean
        b = (sigma / 2) * (2**0.5)
        self._update(mu, b)

    def _fit_mle(self, sample, **kwargs):
        mu, b = self.dist.fit(sample, **kwargs)
        self._update(mu, b)


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
        az.style.use('arviz-white')
        mus = [ 0., 0.]
        sigmas = [.5, 1.]
        for mu, sigma in zip(mus, sigmas):
            LogNormal(mu, sigma).plot_pdf()

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
        self.mu = mu
        self.sigma = sigma
        self.name = "lognormal"
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.lognorm
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.sigma, scale=np.exp(self.mu))
        return frozen

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def _fit_mle(self, sample, **kwargs):
        sigma, _, mu = self.dist.fit(sample, **kwargs)
        self._update(np.log(mu), sigma)


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

        import arviz as az
        from preliz import Normal
        az.style.use('arviz-white')
        mus = [0., 0., -2.]
        sigmas = [1, 0.5, 1]
        for mu, sigma in zip(mus, sigmas):
            Normal(mu, sigma).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\sigma^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0).
    """

    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.name = "normal"
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.norm
        self.support = (-np.inf, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


class Pareto(Continuous):
    r"""
    Pareto log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Pareto
        az.style.use('arviz-white')
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
        self.alpha = alpha
        self.m = m  # pylint: disable=invalid-name
        self.name = "pareto"
        self.params = (self.alpha, self.m)
        self.param_names = ("alpha", "m")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.pareto
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, scale=self.m)
        return frozen

    def _update(self, alpha, m):  # pylint: disable=invalid-name
        self.alpha = alpha
        self.m = m
        self.params = (self.alpha, self.m)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = 1 + (1 + (mean / sigma) ** 2) ** (1 / 2)
        m = (alpha - 1) * mean / alpha  # pylint: disable=invalid-name
        self._update(alpha, m)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, m = self.dist.fit(sample, **kwargs)  # pylint: disable=invalid-name
        self._update(alpha, m)


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
            az.style.use('arviz-white')
            for alpha in [-6, 0, 6]:
                SkewNormal(mu=0, sigma=1, alpha=alpha).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac {\alpha }{{\sqrt {1+\alpha ^{2}}}}`
    Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    alpha : float
        Skewness parameter.

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    and sigma the standard deviation. In the limit of alpha approaching
    plus/minus infinite we get a half-normal distribution.
    """

    def __init__(self, mu=None, sigma=None, alpha=None):
        super().__init__()
        if alpha is None:
            alpha = 0
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.name = "skewnormal"
        self.params = (self.mu, self.sigma, self.alpha)
        self.param_names = ("mu", "sigma", "alpha")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (-np.inf, np.inf))
        self.dist = stats.skewnorm
        self.support = (-np.inf, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma, self.alpha)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Just assume this is a gaussian
        self._update(mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        alpha, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, alpha)


class Student(Continuous):
    r"""
    Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu, \mu, \sigma) =
           \frac{\Gamma \left(\frac{\nu+1}{2} \right)} {\sqrt{\nu\pi}\
           \Gamma \left(\frac{\nu}{2} \right)} \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Student
        az.style.use('arviz-white')
        nus = [2., 5., 5.]
        mus = [0., 0.,  -4.]
        sigmas = [1., 1., 2.]
        for nu, mu, sigma in zip(nus, mus, sigmas):
            Student(nu, mu, sigma).plot_pdf()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu` for :math:`\nu > 1`, otherwise undefined
    Variance  :math:`\frac{\nu}{\nu-2}` for :math:`\nu > 2`,
              :math:`\infty` for :math:`1 < \nu \le 2`, otherwise undefined
    ========  ========================

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases.
    """

    def __init__(self, nu=3, mu=None, sigma=None):
        super().__init__()
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.name = "student"
        self.params = (self.nu, self.mu, self.sigma)
        self.param_names = ("nu", "mu", "sigma")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.t
        self.support = (-np.inf, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.nu, self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma, nu=None):
        if nu is not None:
            self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.params = (self.nu, self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # if nu is smaller than 2 the variance is not defined,
        # so if that happens we use 2.1 as an approximation
        nu = self.nu
        if nu <= 2:
            nu = 2.1
        sigma = sigma / (nu / (nu - 2)) ** 0.5
        self._update(mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        nu, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, nu)


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
        az.style.use('arviz-white')
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        lowers = [-3, -5, -5]
        uppers = [7, 5, 4]
        for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
            TruncatedNormal(mu, sigma, lower, upper).plot_pdf()

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

    def __init__(self, mu=None, sigma=None, lower=-np.inf, upper=np.inf):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.name = "truncatednormal"
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.param_names = ("mu", "sigma", "lower", "upper")
        self.params_support = (
            (-np.inf, np.inf),
            (eps, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
        )
        self.dist = stats.truncnorm
        self.support = (self.lower, self.upper)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if self.mu is not None or self.sigma is not None:
            a, b = (self.lower - self.mu) / self.sigma, (self.upper - self.mu) / self.sigma
            frozen = self.dist(a, b, self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma, lower=None, upper=None):
        if lower is not None:
            self.lower = lower
        if upper is not None:
            self.upper = upper

        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # aproximated version
        self._update(mean, sigma)

    def _fit_mle(self, sample, **kwargs):
        a, b, mu, sigma = self.dist.fit(sample, **kwargs)
        lower, upper = a * sigma + mu, b * sigma + mu
        self._update(mu, sigma, lower, upper)


class Uniform(Continuous):
    r"""
    Uniform distribution.

    The pdf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Uniform
        az.style.use('arviz-white')
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

    def __init__(self, lower=-np.inf, upper=np.inf):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.name = "uniform"
        self.params = (self.lower, self.upper)
        self.param_names = ("lower", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf))
        self.dist = stats.uniform
        self.support = (lower, upper)
        self.dist.a = -np.inf
        self.dist.b = np.inf
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.lower, self.upper - self.lower)
        return frozen

    def _update(self, lower, upper):
        self.lower = lower
        self.upper = upper
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
        plt.style.use('arviz-white')
        mus = [1., 1.]
        lams = [1., 3.]
        for mu, lam in zip(mus, lams):
            Wald(mu, lam).plot_pdf()

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Parameters
    ----------
    mu : float
        Mean of the distribution (mu > 0).
    lam : float
        Relative precision (lam > 0).
    """

    def __init__(self, mu=None, lam=None):
        super().__init__()
        self.mu = mu
        self.lam = lam
        self.name = "wald"
        self.params = (self.mu, self.lam)
        self.param_names = ("mu", "lam")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.invgauss
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.mu / self.lam, scale=self.lam)
        return frozen

    def _update(self, mu, lam):
        self.mu = mu
        self.lam = lam
        self.params = (self.mu, self.lam)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        lam = mean**3 / sigma**2
        self._update(mean, lam)

    def _fit_mle(self, sample, **kwargs):
        mu, _, lam = self.dist.fit(sample, **kwargs)
        self._update(mu * lam, lam)


class Weibull(Continuous):
    r"""
    Weibull distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Weibull
        plt.style.use('arviz-white')
        alphas = [1., 2, 5.]
        betas = [1., 1., 2.]
        for a, b in zip(alphas, betas):
            Weibull(a, b).plot_pdf()

    ========  ====================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2/\beta^2)`
    ========  ====================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "weibull"
        self.params = (self.alpha, self.beta)
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        self.dist = stats.weibull_min
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, scale=self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha, beta = garcia_approximation(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)
