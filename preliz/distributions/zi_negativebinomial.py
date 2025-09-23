import numpy as np
from pytensor_distributions import zi_negativebinomial as ptd_zinegativebinomial

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import (
    all_not_none,
    any_not_none,
    eps,
    pytensor_jit,
    pytensor_rng_jit,
)
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


class ZeroInflatedNegativeBinomial(Discrete):
    R"""
    Zero-Inflated Negative binomial distribution.

    The Zero-inflated version of the Negative Binomial (NB).
    The NB distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    The pmf of this distribution is

    .. math::

       f(x \mid \psi, \mu, \alpha) = \left\{
         \begin{array}{l}
           (1-\psi) + \psi \left (
             \frac{\alpha}{\alpha+\mu}
           \right) ^\alpha, \text{if } x = 0 \\
           \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (
             \frac{\alpha}{\mu+\alpha}
           \right)^\alpha \left(
             \frac{\mu}{\mu+\alpha}
           \right)^x, \text{if } x=1,2,3,\ldots
         \end{array}
       \right.

    .. plot::
        :context: close-figs

        from preliz import ZeroInflatedNegativeBinomial, style
        style.use('preliz-doc')
        psis = [0.7, 0.7]
        mus = [2, 8]
        alphas = [2, 4]
        for psi, mu, alpha in zip(psis, mus, alphas):
            ZeroInflatedNegativeBinomial(psi, mu=mu, alpha=alpha).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Variance .. math::
                  \psi \left(\frac{{\mu^2}}{{\alpha}}\right) +\
                  \psi \mu + \psi \mu^2 - \psi^2 \mu^2
    ========  ==========================

    The zero inflated negative binomial distribution can be parametrized
    either in terms of mu and alpha, or in terms of n and p.
    The link between the parametrizations is given by

    .. math::

        \mu &= \frac{n(1-p)}{p} \\
        \alpha &= n

    Parameters
    ----------
    psi : float
        Expected proportion of NegativeBinomial variates (0 < psi < 1)
    mu : float
        Poisson distribution parameter (mu > 0).
    alpha : float
        Gamma distribution parameter (alpha > 0).
    p : float
        Alternative probability of success in each trial (0 < p < 1).
    n : float
        Alternative number of target success trials (n > 0)
    """

    def __init__(self, psi=None, mu=None, alpha=None, p=None, n=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(psi, mu, alpha, p, n)

    def _parametrization(self, psi=None, mu=None, alpha=None, p=None, n=None):
        if any_not_none(mu, alpha) and any_not_none(p, n):
            raise ValueError(
                "Incompatible parametrization. Either use psi, mu and alpha, or psi, p and n."
            )

        self.psi = psi
        self.param_names = ("psi", "mu", "alpha")
        self.params_support = ((eps, 1 - eps), (eps, np.inf), (eps, np.inf))

        if any_not_none(p, n):
            self.p = p
            self.n = n
            self.param_names = ("psi", "p", "n")
            if all_not_none(p, n):
                mu, alpha = self._from_p_n(p, n)

        self.mu = mu
        self.alpha = alpha
        self.params = (self.psi, self.mu, self.alpha)
        if all_not_none(mu, alpha):
            self._update(psi, mu, alpha)

    def _from_p_n(self, p, n):
        alpha = n
        mu = n * (1 / p - 1)
        return mu, alpha

    def _to_p_n(self, mu, alpha):
        p = alpha / (mu + alpha)
        n = alpha
        return p, n

    def _update(self, psi, mu, alpha):
        self.psi = np.float64(psi)
        self.mu = np.float64(mu)
        self.alpha = np.float64(alpha)
        self.p, self.n = self._to_p_n(self.mu, self.alpha)

        if self.param_names[1] == "mu":
            self.params = (self.psi, self.mu, self.alpha)
        elif self.param_names[1] == "p":
            self.params = (self.psi, self.p, self.n)

        self.is_frozen = True

    def pdf(self, x):
        x = np.asarray(x)
        result = ptd_pdf(x, self.psi, self.n, self.p)
        # Return 0 for negative values and NaN for infinity, consistent with scipy.stats.nbinom
        result = np.where(x < 0, 0, result)
        result = np.where(~np.isfinite(x), np.nan, result)
        return result

    def cdf(self, x):
        return ptd_cdf(x, self.psi, self.n, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.psi, self.n, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.psi, self.n, self.p)

    def entropy(self):
        return ptd_entropy(self.psi, self.n, self.p)

    def mean(self):
        return ptd_mean(self.psi, self.n, self.p)

    def mode(self):
        return ptd_mode(self.psi, self.n, self.p)

    def median(self):
        return ptd_median(self.psi, self.n, self.p)

    def var(self):
        return ptd_var(self.psi, self.n, self.p)

    def std(self):
        return ptd_std(self.psi, self.n, self.p)

    def skewness(self):
        return ptd_skewness(self.psi, self.n, self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.psi, self.n, self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.psi, self.n, self.p, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, psi, n, p):
    return ptd_zinegativebinomial.pdf(x, psi, n, p)


@pytensor_jit
def ptd_cdf(x, psi, n, p):
    return ptd_zinegativebinomial.cdf(x, psi, n, p)


@pytensor_jit
def ptd_ppf(q, psi, n, p):
    return ptd_zinegativebinomial.ppf(q, psi, n, p)


@pytensor_jit
def ptd_logpdf(x, psi, n, p):
    return ptd_zinegativebinomial.logpdf(x, psi, n, p)


@pytensor_jit
def ptd_entropy(psi, n, p):
    return ptd_zinegativebinomial.entropy(psi, n, p)


@pytensor_jit
def ptd_mean(psi, n, p):
    return ptd_zinegativebinomial.mean(psi, n, p)


@pytensor_jit
def ptd_mode(psi, n, p):
    return ptd_zinegativebinomial.mode(psi, n, p)


@pytensor_jit
def ptd_median(psi, n, p):
    return ptd_zinegativebinomial.median(psi, n, p)


@pytensor_jit
def ptd_var(psi, n, p):
    return ptd_zinegativebinomial.var(psi, n, p)


@pytensor_jit
def ptd_std(psi, n, p):
    return ptd_zinegativebinomial.std(psi, n, p)


@pytensor_jit
def ptd_skewness(psi, n, p):
    return ptd_zinegativebinomial.skewness(psi, n, p)


@pytensor_jit
def ptd_kurtosis(psi, n, p):
    return ptd_zinegativebinomial.kurtosis(psi, n, p)


@pytensor_rng_jit
def ptd_rvs(psi, n, p, size, rng):
    return ptd_zinegativebinomial.rvs(psi, n, p, size=size, random_state=rng)
