"""
Continuous probability distributions.
"""
# pylint: disable=useless-super-delegation
import numpy as np
from scipy import stats

from .distributions import Continuous


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
        self.dist = stats.beta
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        """
        Estimate the parameters of the distribution from the mean and standard deviation.
        """
        kappa = (mean * (1 - mean) / (sigma) ** 2) - 1
        alpha = max(0.5, mean * kappa)
        beta = max(0.5, (1 - mean) * kappa)
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        """
        Estimate the parameters of the distribution from a sample by maximizing the likelihood.
        """
        alpha, beta, _, _ = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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
        self.params = (self.lam, None)
        self.param_names = ("lam",)
        self.dist = stats.expon
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(scale=1 / self.lam)

    def _update(self, lam):
        self.lam = lam
        self.params = (self.lam, None)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        lam = 1 / mean
        self._update(lam)

    def fit_mle(self, sample, **kwargs):
        _, lam = self.dist.fit(sample, **kwargs)
        self._update(1 / lam)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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
        self.dist = stats.gamma
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(a=self.alpha, scale=1 / self.beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        alpha = mean**2 / sigma**2
        beta = mean / sigma**2
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, 1 / beta)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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
        self.dist = stats.lognorm
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.sigma, scale=np.exp(self.mu))

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def fit_mle(self, sample, **kwargs):
        sigma, _, mu = self.dist.fit(sample, **kwargs)
        self._update(np.log(mu), sigma)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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
        self.dist = stats.norm
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.mu, self.sigma)

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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

    def fit_moments(self, mean, sigma):
        # This is a placeholder!!!
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        nu, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, nu)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)
