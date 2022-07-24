"""
Continuous probability distributions.
"""
import numpy as np
from scipy import stats

from .distributions import Continuous

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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        kappa = (mean * (1 - mean) / (sigma) ** 2) - 1
        alpha = max(0.5, mean * kappa)
        beta = max(0.5, (1 - mean) * kappa)
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        alpha, beta, _, _ = self.dist.fit(sample, **kwargs)
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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(scale=1 / self.lam)

    def _update(self, lam):
        self.lam = lam
        self.params = (self.lam,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        lam = 1 / mean
        self._update(lam)

    def fit_mle(self, sample, **kwargs):
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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(a=self.alpha, scale=1 / self.beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        alpha = mean**2 / sigma**2
        beta = mean / sigma**2
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, 1 / beta)


class HalfNormal(Continuous):
    r"""
    HalfNormal Distribution
    """

    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma
        self.name = "halfnormal"
        self.params = (self.sigma,)
        self.param_names = ("sigma",)
        self.params_support = ((eps, np.inf),)
        self.dist = stats.halfnorm
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(scale=self.sigma)

    def _update(self, sigma):
        self.sigma = sigma
        self.params = (self.sigma,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        sigma = sigma * (1 - 2 / np.pi) ** 0.5
        self._update(sigma)

    def fit_mle(self, sample, **kwargs):
        _, sigma = self.dist.fit(sample, **kwargs)
        self._update(sigma)


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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(loc=self.mu, scale=self.b)

    def _update(self, mu, b):
        self.mu = mu
        self.b = b
        self.params = (self.mu, self.b)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = mean
        b = (sigma / 2) * (2**0.5)
        self._update(mu, b)

    def fit_mle(self, sample, **kwargs):
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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.sigma, scale=np.exp(self.mu))

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def fit_mle(self, sample, **kwargs):
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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.mu, self.sigma)

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


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
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.alpha, self.mu, self.sigma)

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

    def fit_mle(self, sample, **kwargs):
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

    def __init__(self, nu=None, mu=None, sigma=None):
        super().__init__()
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.name = "student"
        self.params = (self.nu, self.mu, self.sigma)
        self.param_names = ("nu", "mu", "sigma")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))
        self.dist = stats.t
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.nu, self.mu, self.sigma)

    def _update(self, mu, sigma, nu=None):
        if nu is not None:
            self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.params = (self.nu, self.mu, self.sigma)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # This is a placeholder!!!
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        nu, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, nu)


class Uniform(Continuous):
    R"""
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
    lower: int
        Lower limit.
    upper: int
        Upper limit (upper > lower).
    """

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.name = "uniform"
        self.params = (self.lower, self.upper)
        self.param_names = ("lower", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf))
        self.dist = stats.uniform
        self.dist.a = -np.inf
        self.dist.b = np.inf
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.lower, self.upper - self.lower)

    def _update(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.params = (self.lower, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        lower = mean - 1.73205 * sigma
        upper = mean + 1.73205 * sigma
        self._update(lower, upper)

    def fit_mle(self, sample, **kwargs):
        lower = np.min(sample)
        upper = np.max(sample)
        self._update(lower, upper)
