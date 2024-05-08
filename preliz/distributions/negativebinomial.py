# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import nbdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.distribution_helper import eps, any_not_none, all_not_none
from ..internal.optimization import optimize_moments, optimize_ml
from ..internal.special import betainc, gammaln, xlogy, cdf_bounds, ppf_bounds_disc


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

        import arviz as az
        from preliz import NegativeBinomial
        az.style.use('arviz-doc')
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.n, self.p))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.n, self.p, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.n, self.p, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.n, self.p)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.n, self.p)

    def entropy(self):
        x = self.xvals("full", 5000)
        logpdf = self.logpdf(x)
        return -np.sum(np.exp(logpdf) * logpdf)

    def mean(self):
        return self.mu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return self.mu**2 / self.alpha + self.mu

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (2 - self.p) / ((1 - self.p) * self.n) ** 0.5

    def kurtosis(self):
        return 6 / self.n + self.p**2 / ((1 - self.p) * self.n)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.negative_binomial(self.n, self.p, size=size)

    def _fit_moments(self, mean, sigma=None):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, n, p, lower, upper):
    prob = betainc(n, x + 1, p)
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
# bdtrik not supported by numba
def nb_ppf(q, n, p, lower, upper):
    x_vals = np.ceil(nbdtrik(q, n, p))
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(y, n, p):
    if y < 0:
        return -np.inf
    else:
        return gammaln(y + n) - gammaln(n) - gammaln(y + 1) + xlogy(n, p) + xlogy(y, 1 - p)


@nb.njit(cache=True)
def nb_neg_logpdf(y, n, p):
    return -(nb_logpdf(y, n, p)).sum()
