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

from ..utils.optimization import optimize_ml
from ..utils.utils import garcia_approximation
from .distributions import Continuous

eps = np.finfo(float).eps


def from_precision(precision):
    sigma = 1 / precision**0.5
    return sigma


def to_precision(sigma):
    precision = 1 / sigma**2
    return precision


class AsymmetricLaplace(Continuous):
    r"""
    Asymmetric-Laplace dstribution.

    The pdf of this distribution is

    .. math::
        {f(x|\\b,\kappa,\mu) =
            \left({\frac{\\b}{\kappa + 1/\kappa}}\right)\,e^{-(x-\mu)\\b\,s\kappa ^{s}}}

    where

    .. math::

        s = sgn(x-\mu)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import AsymmetricLaplace
        az.style.use('arviz-white')
        kappas = [1., 2., .5]
        mus = [0., 0., 3.]
        bs = [1., 1., 1.]
        for kappa, mu, b in zip(kappas, mus, bs):
            AsymmetricLaplace(kappa, mu, b).plot_pdf(support=(-10,10))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
    Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
    ========  ========================

    AsymmetricLaplace distribution has 2 alternative parametrizarions. In terms of kappa,
    mu and b or q, mu and b.

    The link between the 2 alternatives is given by

    .. math::

       \kappa = \sqrt(\frac{q}{1-q})

    Parameters
    ----------
    kappa : float
        Symmetry parameter (kappa > 0).
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    q : float
        Symmetry parameter (0 < q < 1).
    """

    def __init__(self, kappa=None, mu=None, b=None, q=None):
        super().__init__()
        self.name = "asymmetriclaplace"
        self.dist = stats.laplace_asymmetric
        self.support = (-np.inf, np.inf)
        self._parametrization(kappa, mu, b, q)

    def _parametrization(self, kappa=None, mu=None, b=None, q=None):
        if kappa is not None and q is not None:
            raise ValueError("Incompatible parametrization. Either use kappa or q.")

        self.param_names = ("kappa", "mu", "b")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))

        if q is not None:
            self.q = q
            kappa = self._from_q(q)
            self.param_names = ("q", "mu", "b")
            self.params_support = ((eps, 1 - eps), (-np.inf, np.inf), (eps, np.inf))

        self.kappa = kappa
        self.mu = mu
        self.b = b
        if (kappa and mu and b) is not None:
            self._update(kappa, mu, b)

    def _from_q(self, q):
        kappa = (q / (1 - q)) ** 0.5
        return kappa

    def _to_q(self, kappa):
        q = kappa**2 / (1 + kappa**2)
        return q

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(kappa=self.kappa, loc=self.mu, scale=self.b)
        return frozen

    def _update(self, kappa, mu, b):
        self.kappa = kappa
        self.mu = mu
        self.b = b
        self.q = self._to_q(self.kappa)

        if self.param_names[0] == "kappa":
            self.params = (self.kappa, self.mu, self.b)
        elif self.param_names[0] == "q":
            self.params = (self.q, self.mu, self.b)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Assume symmetry
        mu = mean
        b = (sigma / 2) * (2**0.5)
        self._update(1, mu, b)

    def _fit_mle(self, sample, **kwargs):
        kappa, mu, b = self.dist.fit(sample, **kwargs)
        self._update(kappa, mu, b)


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

    Beta distribution has 3 alternative parameterizations. In terms of alpha and
    beta, mean and sigma (standard deviation) or mean and kappa (concentration).

    The link between the 3 alternatives is given by

    .. math::

       \alpha &= \mu \kappa \\
       \beta  &= (1 - \mu) \kappa

       \text{where } \kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1


    Parameters
    ----------
    alpha : float
        alpha  > 0
    beta : float
        beta  > 0
    mu : float
        mean (0 < ``mu`` < 1).
    sigma : float
        standard deviation (``sigma`` < sqrt(``mu`` * (1 - ``mu``))).
    kappa : float
        concentration > 0
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None, kappa=None):
        super().__init__()
        self.name = "beta"
        self.dist = stats.beta
        self.support = (0, 1)
        self._parametrization(alpha, beta, mu, sigma, kappa)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None, kappa=None):
        if (
            (alpha or beta) is not None
            and (mu or sigma or kappa) is not None
            or (sigma and kappa) is not None
        ):
            raise ValueError(
                "Incompatible parametrization. Either use alpha " "and beta, or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if (mu or sigma) is not None:
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            self.params_support = ((eps, 1 - eps), (eps, 1 - eps))
            if (mu and sigma) is not None:
                alpha, beta = self._from_mu_sigma(mu, sigma)

        if (mu or kappa) is not None:
            self.mu = mu
            self.kappa = kappa
            self.param_names = ("mu", "kappa")
            self.params_support = ((eps, 1 - eps), (eps, np.inf))
            if (mu and kappa) is not None:
                alpha, beta = self._from_mu_kappa(mu, kappa)

        self.alpha = alpha
        self.beta = beta
        if (self.alpha and self.beta) is not None:
            self._update(self.alpha, self.beta)

    def _from_mu_sigma(self, mu, sigma):
        kappa = mu * (1 - mu) / sigma**2 - 1
        alpha = mu * kappa
        beta = (1 - mu) * kappa
        return alpha, beta

    def _from_mu_kappa(self, mu, kappa):
        alpha = mu * kappa
        beta = (1 - mu) * kappa
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        alpha_plus_beta = alpha + beta
        mu = alpha / alpha_plus_beta
        sigma = (alpha * beta) ** 0.5 / alpha_plus_beta / (alpha_plus_beta + 1) ** 0.5
        return mu, sigma

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.mu, self.sigma = self._to_mu_sigma(self.alpha, self.beta)
        self.kappa = self.mu * (1 - self.mu) / self.sigma**2 - 1

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "kappa":
            self.params = (self.mu, self.kappa)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha, beta = self._from_mu_sigma(mean, sigma)
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
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
        self.dist = copy(stats.beta)
        self.support = (lower, upper)
        self._parametrization(self.alpha, self.beta, self.lower, self.upper)

    def _parametrization(self, alpha=None, beta=None, lower=None, upper=None):
        self.param_names = ("alpha", "beta", "lower", "upper")
        self.params_support = ((eps, np.inf), (eps, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if (alpha and beta) is not None:
            self._update(alpha, beta, lower, upper)

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
        self.name = "cauchy"
        self.dist = stats.cauchy
        self.support = (-np.inf, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ("alpha", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.params = (self.alpha, self.beta)
        if (alpha and beta) is not None:
            self._update(alpha, beta)

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


class ChiSquared(Continuous):
    r"""
    Chi squared  log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) =
                \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ChiSquared
        az.style.use('arviz-white')
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
        self.name = "chisquared"
        self.dist = stats.chi2
        self.support = (0, np.inf)
        self._parametrization(nu)

    def _parametrization(self, nu=None):
        self.nu = nu
        self.param_names = ("nu",)
        self.params_support = ((eps, np.inf),)
        self.params = (self.nu,)
        self._update(self.nu)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.nu)
        return frozen

    def _update(self, nu):
        self.nu = nu
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
        az.style.use('arviz-white')
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
        self.name = "exgaussian"
        self.dist = stats.exponnorm
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, nu)

    def _parametrization(self, mu=None, sigma=None, nu=None):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.param_names = ("mu", "sigma", "nu")
        self.params = (mu, sigma, nu)
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (eps, np.inf))
        if (mu and sigma and nu) is not None:
            self._update(mu, sigma, nu)

    def _get_frozen(self):
        frozen = None
        if self.nu is not None and self.sigma is not None:
            frozen = self.dist(K=self.nu / self.sigma, loc=self.mu, scale=self.sigma)
        return frozen

    def _update(self, mu, sigma, nu):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma, self.nu)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Just assume this is a approximately Gaussian
        self._update(mean, sigma, 1e-6)

    def _fit_mle(self, sample, **kwargs):
        K, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, K * sigma)


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
            Exponential(lam).plot_pdf(support=(0,5))


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
        self.dist = stats.expon
        self.support = (0, np.inf)
        self._parametrization(lam)

    def _parametrization(self, lam=None):
        self.lam = lam
        self.params = (self.lam,)
        self.param_names = ("lam",)
        self.params_support = ((eps, np.inf),)
        self._update(self.lam)

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
        self.name = "gamma"
        self.dist = stats.gamma
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, sigma)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha or beta) is not None and (mu or sigma) is not None:
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if (mu or sigma) is not None:
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            if (mu and sigma) is not None:
                alpha, beta = self._from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if self.alpha is not None and self.beta is not None:
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
        if any(self.params):
            frozen = self.dist(a=self.alpha, scale=1 / self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
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
        az.style.use('arviz-white')
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
        self.name = "gumbel"
        self.dist = stats.gumbel_r
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, beta)

    def _parametrization(self, mu=None, beta=None):
        self.mu = mu
        self.beta = beta
        self.params = (self.mu, self.beta)
        self.param_names = ("mu", "beta")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self._update(self.mu, self.beta)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(loc=self.mu, scale=self.beta)
        return frozen

    def _update(self, mu, beta):
        self.mu = mu
        self.beta = beta
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
        self.name = "halfcauchy"
        self.dist = stats.halfcauchy
        self.support = (0, np.inf)
        self._parametrization(beta)

    def _parametrization(self, beta=None):
        self.beta = beta
        self.params = (self.beta,)
        self.param_names = ("beta",)
        self.params_support = ((eps, np.inf),)
        self._update(self.beta)

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
            HalfNormal(sigma).plot_pdf(support=(0,5))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    HalfNormal distribution has 2 alternative parameterizations. In terms of sigma (standard
    deviation) or tau (precision).

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`\sigma` (``sigma`` > 0).
    tau : float
        Precision :math:`\tau` (``tau`` > 0).
    """

    def __init__(self, sigma=None, tau=None):
        super().__init__()
        self.name = "halfnormal"
        self.dist = stats.halfnorm
        self.support = (0, np.inf)
        self._parametrization(sigma, tau)

    def _parametrization(self, sigma=None, tau=None):
        if sigma is not None and tau is not None:
            raise ValueError("Incompatible parametrization. Either use sigma or tau.")

        self.param_names = ("sigma",)
        self.params_support = ((eps, np.inf),)

        if tau is not None:
            sigma = from_precision(tau)
            self.param_names = ("tau",)

        self.sigma = sigma
        self.tau = tau
        if self.sigma is not None:
            self._update(self.sigma)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(scale=self.sigma)
        return frozen

    def _update(self, sigma):
        self.sigma = sigma
        self.tau = to_precision(sigma)

        if self.param_names[0] == "sigma":
            self.params = (self.sigma,)
        elif self.param_names[0] == "tau":
            self.params = (self.tau,)

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
            HalfStudent(nu, sigma).plot_pdf(support=(0,10))

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

    HalfStudent distribution has 2 alternative parameterizations. In terms of nu and
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
        self.name = "halfstudent"
        self.dist = _HalfStudent
        self.support = (0, np.inf)
        self._parametrization(nu, sigma, lam)

    def _parametrization(self, nu=None, sigma=None, lam=None):
        if sigma is not None and lam is not None:
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
        if self.nu is not None and self.sigma is not None:
            self._update(self.nu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if all(self.params):
            frozen = self.dist(nu=self.nu, sigma=self.sigma)
        return frozen

    def _update(self, nu, sigma):
        self.nu = nu
        self.sigma = sigma
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


class _HalfStudent(stats._distn_infrastructure.rv_continuous):
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
        self.name = "inversegamma"
        self.dist = stats.invgamma
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, mu, sigma)

    def _parametrization(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha or beta) is not None and (mu or sigma) is not None:
            raise ValueError(
                "Incompatible parametrization. Either use alpha and beta or mu and sigma."
            )

        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if (mu or sigma) is not None:
            self.mu = mu
            self.sigma = sigma
            self.param_names = ("mu", "sigma")
            if (mu and sigma) is not None:
                alpha, beta = self._from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if self.alpha is not None and self.beta is not None:
            self._update(self.alpha, self.beta)

    def _from_mu_sigma(self, mu, sigma):
        alpha = mu**2 / sigma**2 + 2
        beta = mu**3 / sigma**2 + mu
        return alpha, beta

    def _to_mu_sigma(self, alpha, beta):
        mu = beta / (alpha - 1)
        sigma = beta / ((alpha - 1) * (alpha - 2) ** 0.5)
        return mu, sigma

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(a=self.alpha, scale=self.beta)
        return frozen

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
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
            Laplace(mu, b).plot_pdf(support=(-10,10))

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
        self.name = "laplace"
        self.dist = stats.laplace
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, b)

    def _parametrization(self, mu=None, b=None):
        self.mu = mu
        self.b = b
        self.params = (self.mu, self.b)
        self.param_names = ("mu", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if (mu and b) is not None:
            self._update(mu, b)

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
         az.style.use('arviz-white')
         mus = [0., 0., -2.]
         ss = [1., 2., .4]
         for mu, s in zip(mus, ss):
             Logistic(mu, s).plot_pdf(support=(-5,5))

    ========  ==========================================
     Support   :math:`x \in \mathbb{R}`
     Mean      :math:`\mu`
     Variance  :math:`\frac{s^2 \pi^2}{3}`
     ========  ==========================================

     Parameters
     ----------
     mu : float
        Mean.
     s : float
        Scale (s > 0).
    """

    def __init__(self, mu=None, s=None):
        super().__init__()
        self.name = "logistic"

        self.dist = stats.logistic
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, s)

    def _parametrization(self, mu=None, s=None):
        self.mu = mu
        self.s = s
        self.params = (self.mu, self.s)
        self.param_names = ("mu", "s")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self._update(self.mu, self.s)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(loc=self.mu, scale=self.s)
        return frozen

    def _update(self, mu, s):
        self.mu = mu
        self.s = s
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
        az.style.use('arviz-white')
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
        self.name = "lognormal"
        self.dist = stats.lognorm
        self.support = (0, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if (mu and sigma) is not None:
            self._update(mu, sigma)

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
        az.style.use('arviz-white')
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
        self.name = "moyal"
        self.dist = stats.moyal
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma)

    def _parametrization(self, mu=None, sigma=None):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self.param_names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        if (mu and sigma) is not None:
            self._update(self.mu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(loc=self.mu, scale=self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        sigma = sigma / np.pi * 2**0.5
        mu = mean - sigma * (np.euler_gamma + np.log(2))
        self._update(mu, sigma)

    def _fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


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
        self.name = "normal"
        self.dist = stats.norm
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, tau)

    def _parametrization(self, mu=None, sigma=None, tau=None):
        if sigma is not None and tau is not None:
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
        if mu is not None and sigma is not None:
            self._update(mu, sigma)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

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
        self.name = "pareto"
        self.dist = stats.pareto
        self.support = (0, np.inf)
        self._parametrization(alpha, m)

    def _parametrization(self, alpha=None, m=None):
        self.alpha = alpha
        self.m = m
        self.params = (self.alpha, self.m)
        self.param_names = ("alpha", "m")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if (alpha and m) is not None:
            self._update(alpha, m)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, scale=self.m)
        return frozen

    def _update(self, alpha, m):
        self.alpha = alpha
        self.m = m
        self.support = (m, np.inf)
        self.params = (self.alpha, self.m)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = 1 + (1 + (mean / sigma) ** 2) ** (1 / 2)
        m = (alpha - 1) * mean / alpha
        self._update(alpha, m)

    def _fit_mle(self, sample, **kwargs):
        alpha, _, m = self.dist.fit(sample, **kwargs)
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
        self.name = "skewnormal"
        self.dist = stats.skewnorm
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, sigma, alpha, tau)

    def _parametrization(self, mu=None, sigma=None, alpha=None, tau=None):
        if sigma is not None and tau is not None:
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
        if self.mu is not None and self.sigma is not None and self.alpha is not None:
            self._update(self.mu, self.sigma, self.alpha)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.alpha, self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma, alpha):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
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
            Student(nu, mu, sigma).plot_pdf(support=(-10,6))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu` for :math:`\nu > 1`, otherwise undefined
    Variance  :math:`\frac{\nu}{\nu-2}` for :math:`\nu > 2`,
              :math:`\infty` for :math:`1 < \nu \le 2`, otherwise undefined
    ========  ========================

    Student's T distribution has 2 alternative parameterizations. In terms of nu, mu and
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
        self.name = "student"
        self.dist = stats.t
        self.support = (-np.inf, np.inf)
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))
        self._parametrization(nu, mu, sigma, lam)

    def _parametrization(self, nu=None, mu=None, sigma=None, lam=None):
        if sigma is not None and lam is not None:
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

        if self.nu is not None and self.mu is not None and self.sigma is not None:
            self._update(self.nu, self.mu, self.sigma)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.nu, self.mu, self.sigma)
        return frozen

    def _update(self, nu, mu, sigma):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
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
        az.style.use('arviz-white')
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
        self.name = "triangular"
        self.dist = stats.triang
        self.support = (-np.inf, np.inf)
        self._parametrization(lower, c, upper)

    def _parametrization(self, lower=None, c=None, upper=None):
        self.lower = lower
        self.c = c
        self.upper = upper
        self.params = (self.lower, self.c, self.upper)
        self.param_names = ("lower", "c", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if (lower and c and upper) is not None:
            self._update(lower, c, upper)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            scale = self.upper - self.lower
            c_ = (self.c - self.lower) / scale
            frozen = self.dist(c=c_, loc=self.lower, scale=scale)
        return frozen

    def _update(self, lower, c, upper):
        self.lower = lower
        self.c = c
        self.upper = upper
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
        az.style.use('arviz-white')
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
        self.name = "truncatednormal"
        self.dist = stats.truncnorm
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
        if (mu and sigma and lower and upper) is not None:
            self._update(mu, sigma, lower, upper)

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
        # Assume gaussian
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

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.name = "uniform"
        self.dist = stats.uniform
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
        self.dist.a = self.lower
        self.dist.b = self.upper
        if (lower and upper) is not None:
            self._update(lower, upper)
        else:
            self.lower = lower
            self.upper = upper

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


class VonMises(Continuous):
    r"""
    Univariate VonMises log-likelihood.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import VonMises
        az.style.use('arviz-white')
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
        self.name = "vonmises"
        self.dist = stats.vonmises
        self._parametrization(mu, kappa)

    def _parametrization(self, mu=None, kappa=None):
        self.mu = mu
        self.kappa = kappa
        self.param_names = ("mu", "kappa")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))
        self.support = (-np.pi, np.pi)
        if (mu and kappa) is not None:
            self._update(mu, kappa)

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(kappa=self.kappa, loc=self.mu)
        return frozen

    def _update(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa
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
        az.style.use('arviz-white')
        mus = [1., 1.]
        lams = [1., 3.]
        for mu, lam in zip(mus, lams):
            Wald(mu, lam).plot_pdf(support=(0,4))

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
        self.name = "wald"
        self.dist = stats.invgauss
        self.support = (0, np.inf)
        self._parametrization(mu, lam)

    def _parametrization(self, mu=None, lam=None):
        self.mu = mu
        self.lam = lam
        self.param_names = ("mu", "lam")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if (mu and lam) is not None:
            self._update(mu, lam)

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
        az.style.use('arviz-white')
        alphas = [1., 2, 5.]
        betas = [1., 1., 2.]
        for a, b in zip(alphas, betas):
            Weibull(a, b).plot_pdf(support=(0,5))

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
        self.name = "weibull"
        self.dist = stats.weibull_min
        self.support = (0, np.inf)
        self._parametrization(alpha, beta)

    def _parametrization(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ("alpha", "beta")
        self.params_support = ((eps, np.inf), (eps, np.inf))
        if (alpha and beta) is not None:
            self._update(alpha, beta)

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
