import numba as nb
import numpy as np
from pytensor_distributions import inversegamma as ptd_inversegamma

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import (
    all_not_none,
    any_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_ml


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

        from preliz import InverseGamma, style
        style.use('preliz-doc')
        alphas = [1., 2., 3.]
        betas = [1., 1., .5]
        for alpha, beta in zip(alphas, betas):
            InverseGamma(alpha, beta).plot_pdf(support=(0, 3))

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}` for :math:`\alpha > 2`
    ========  ===============================

    Inverse gamma distribution has 2 alternative parameterization. In terms of alpha and
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
                alpha, beta = ptd_inversegamma.from_mu_sigma(mu, sigma)

        self.alpha = alpha
        self.beta = beta
        if all_not_none(self.alpha, self.beta):
            self._update(self.alpha, self.beta)

    def _update(self, alpha, beta):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu = _to_mu(self.alpha, self.beta)
        self.sigma = _to_sigma(self.alpha, self.beta)

        if self.param_names[0] == "alpha":
            self.params = (self.alpha, self.beta)
        elif self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.alpha, self.beta)

    def cdf(self, x):
        return ptd_cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        return ptd_ppf(q, self.alpha, self.beta)

    def logpdf(self, x):
        return ptd_logpdf(x, self.alpha, self.beta)

    def entropy(self):
        return ptd_entropy(self.alpha, self.beta)

    def mean(self):
        return ptd_mean(self.alpha, self.beta)

    def mode(self):
        return ptd_mode(self.alpha, self.beta)

    def median(self):
        return ptd_median(self.alpha, self.beta)

    def var(self):
        return ptd_var(self.alpha, self.beta)

    def std(self):
        return ptd_std(self.alpha, self.beta)

    def skewness(self):
        return ptd_skewness(self.alpha, self.beta)

    def kurtosis(self):
        return ptd_kurtosis(self.alpha, self.beta)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.alpha, self.beta, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        alpha, beta = _from_mu_sigma(mean, sigma)
        self._update(alpha, beta)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, alpha, beta):
    return ptd_inversegamma.pdf(x, alpha, beta)


@pytensor_jit
def ptd_cdf(x, alpha, beta):
    return ptd_inversegamma.cdf(x, alpha, beta)


@pytensor_jit
def ptd_ppf(q, alpha, beta):
    return ptd_inversegamma.ppf(q, alpha, beta)


@pytensor_jit
def ptd_logpdf(x, alpha, beta):
    return ptd_inversegamma.logpdf(x, alpha, beta)


@pytensor_jit
def ptd_entropy(alpha, beta):
    return ptd_inversegamma.entropy(alpha, beta)


@pytensor_jit
def ptd_mean(alpha, beta):
    return ptd_inversegamma.mean(alpha, beta)


@pytensor_jit
def ptd_mode(alpha, beta):
    return ptd_inversegamma.mode(alpha, beta)


@pytensor_jit
def ptd_median(alpha, beta):
    return ptd_inversegamma.median(alpha, beta)


@pytensor_jit
def ptd_var(alpha, beta):
    return ptd_inversegamma.var(alpha, beta)


@pytensor_jit
def ptd_std(alpha, beta):
    return ptd_inversegamma.std(alpha, beta)


@pytensor_jit
def ptd_skewness(alpha, beta):
    return ptd_inversegamma.skewness(alpha, beta)


@pytensor_jit
def ptd_kurtosis(alpha, beta):
    return ptd_inversegamma.kurtosis(alpha, beta)


@pytensor_rng_jit
def ptd_rvs(alpha, beta, size, rng):
    return ptd_inversegamma.rvs(alpha, beta, size=size, random_state=rng)


def _from_mu_sigma(mu, sigma):
    alpha = mu**2 / sigma**2 + 2
    beta = mu**3 / sigma**2 + mu
    return alpha, beta


@nb.vectorize(nopython=True, cache=True)
def _to_mu(alpha, beta):
    if alpha > 1:
        return beta / (alpha - 1)
    else:
        return np.nan


@nb.vectorize(nopython=True, cache=True)
def _to_sigma(alpha, beta):
    if alpha > 2:
        return beta / ((alpha - 1) * (alpha - 2) ** 0.5)
    else:
        return np.nan
