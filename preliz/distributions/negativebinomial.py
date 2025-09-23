import numpy as np
from pytensor_distributions import negativebinomial as ptd_negativebinomial

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import (
    all_not_none,
    any_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class NegativeBinomial(Discrete):
    R"""
    Negative binomial distribution.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    Its pmf, parametrized by the parameters alpha and mu of the gamma distribution, is

    .. math::

       f(x \mid \mu, \alpha) =
           \binom{x + \alpha - 1}{x}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    .. plot::
        :context: close-figs

        from preliz import NegativeBinomial, style
        style.use('preliz-doc')
        mus = [1, 2, 8]
        alphas = [0.9, 2, 4]
        for mu, alpha in zip(mus, alphas):
            NegativeBinomial(mu, alpha).plot_pdf(support=(0, 20))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\frac{\mu (\alpha + \mu)}{\alpha}`
    ========  ==========================

    The negative binomial distribution can be parametrized either in terms of mu and alpha,
    or in terms of n and p. The link between the parametrizations is given by

    .. math::

        p &= \frac{\alpha}{\mu + \alpha} \\
        n &= \alpha

    If it is parametrized in terms of n and p, the negative binomial describes the probability
    to have x failures before the n-th success, given the probability p of success in each trial.
    Its pmf is

    .. math::

        f(x \mid n, p) =
           \binom{x + n - 1}{x}
           (p)^n (1 - p)^x

    Parameters
    ----------
    alpha : float
        Gamma distribution shape parameter (alpha > 0).
    mu : float
        Gamma distribution mean (mu > 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    n : float
        Number of target success trials (n > 0)
    """

    def __init__(self, mu=None, alpha=None, p=None, n=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu, alpha, p, n)

    def _parametrization(self, mu=None, alpha=None, p=None, n=None):
        if any_not_none(mu, alpha) and any_not_none(p, n):
            raise ValueError("Incompatible parametrization. Either use mu and alpha, or p and n.")

        self.param_names = ("mu", "alpha")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if any_not_none(p, n):
            self.p = p
            self.n = n
            self.param_names = ("p", "n")
            if all_not_none(p, n):
                mu, alpha = self._from_n_p(n, p)

        self.mu = mu
        self.alpha = alpha
        if all_not_none(mu, alpha):
            self._update(mu, alpha)

    def _from_n_p(self, n, p):
        mu = n * (1 / p - 1)
        return mu, n

    def _to_n_p(self, mu, alpha):
        p = alpha / (mu + alpha)
        return alpha, p

    def _update(self, mu, alpha):
        self.mu = np.float64(mu)
        self.alpha = np.float64(alpha)
        self.n, self.p = self._to_n_p(self.mu, self.alpha)

        if self.param_names[0] == "mu":
            self.params = (self.mu, self.alpha)
        elif self.param_names[0] == "p":
            self.params = (self.p, self.n)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.n, self.p)

    def cdf(self, x):
        return ptd_cdf(x, self.n, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.n, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.n, self.p)

    def entropy(self):
        return ptd_entropy(self.n, self.p)

    def mean(self):
        return ptd_mean(self.n, self.p)

    def mode(self):
        return ptd_mode(self.n, self.p)

    def median(self):
        return ptd_median(self.n, self.p)

    def var(self):
        return ptd_var(self.n, self.p)

    def std(self):
        return ptd_std(self.n, self.p)

    def skewness(self):
        return ptd_skewness(self.n, self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.n, self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.n, self.p, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma=None):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, n, p):
    return ptd_negativebinomial.pdf(x, n, p)


@pytensor_jit
def ptd_cdf(x, n, p):
    return ptd_negativebinomial.cdf(x, n, p)


@pytensor_jit
def ptd_ppf(q, n, p):
    return ptd_negativebinomial.ppf(q, n, p)


@pytensor_jit
def ptd_logpdf(x, n, p):
    return ptd_negativebinomial.logpdf(x, n, p)


@pytensor_jit
def ptd_entropy(n, p):
    return ptd_negativebinomial.entropy(n, p)


@pytensor_jit
def ptd_mean(n, p):
    return ptd_negativebinomial.mean(n, p)


@pytensor_jit
def ptd_mode(n, p):
    return ptd_negativebinomial.mode(n, p)


@pytensor_jit
def ptd_median(n, p):
    return ptd_negativebinomial.median(n, p)


@pytensor_jit
def ptd_var(n, p):
    return ptd_negativebinomial.var(n, p)


@pytensor_jit
def ptd_std(n, p):
    return ptd_negativebinomial.std(n, p)


@pytensor_jit
def ptd_skewness(n, p):
    return ptd_negativebinomial.skewness(n, p)


@pytensor_jit
def ptd_kurtosis(n, p):
    return ptd_negativebinomial.kurtosis(n, p)


@pytensor_rng_jit
def ptd_rvs(n, p, size, rng):
    return ptd_negativebinomial.rvs(n, p, size=size, random_state=rng)
