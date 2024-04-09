# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import nbdtrik  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.distribution_helper import eps, any_not_none, all_not_none
from ..internal.optimization import optimize_moments, optimize_ml
from ..internal.special import betainc, gammaln, xlogy, cdf_bounds, ppf_bounds_disc


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

        import arviz as az
        from preliz import ZeroInflatedNegativeBinomial
        az.style.use('arviz-doc')
        psis = [0.7, 0.7]
        mus = [2, 8]
        alphas = [2, 4]
        for psi, mu, alpha in zip(psis, mus, alphas):
            ZeroInflatedNegativeBinomial(psi, mu=mu, alpha=alpha).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Var       .. math::
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.psi, self.n, self.p, self.mu))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.psi, self.n, self.p, self.support[0], self.support[1])

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.psi, self.n, self.p, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.psi, self.n, self.p, self.mu)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.psi, self.n, self.p, self.mu)

    def entropy(self):
        x = self.xvals("full", 5000)
        logpdf = self.logpdf(x)
        return -np.sum(np.exp(logpdf) * logpdf)

    def mean(self):
        return self.psi * self.mu

    def median(self):
        # missing explicit expression
        return self.ppf(0.5)

    def var(self):
        var_nb = self.mu**2 / self.alpha + self.mu
        return self.psi * (var_nb + self.mu**2) - (self.psi * self.mu) ** 2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        # implement skewness
        return np.nan

    def kurtosis(self):
        # implement kurtosis
        return np.nan

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        zeros = random_state.uniform(size=size) > (1 - self.psi)
        nbinomial = random_state.negative_binomial(self.n, self.p, size=size)
        return zeros * nbinomial

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, psi, n, p, lower, upper):
    nb_prob = betainc(n, x + 1, p)
    prob = (1 - psi) + psi * nb_prob
    return cdf_bounds(prob, x, lower, upper)


# @nb.jit
# bdtrik not supported by numba
def nb_ppf(q, psi, n, p, lower, upper):
    nb_vals = np.ceil(nbdtrik(q, n, p))
    x_vals = (1 - psi) + psi * nb_vals
    return ppf_bounds_disc(x_vals, q, lower, upper)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(y, psi, n, p, mu):
    if y == 0:
        return np.log((1 - psi) + psi * (n / (n + mu)) ** n)
    else:
        return (
            np.log(psi)
            + gammaln(y + n)
            - gammaln(n)
            - gammaln(y + 1)
            + xlogy(n, p)
            + xlogy(y, 1 - p)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(y, psi, n, p, mu):
    return -(nb_logpdf(y, psi, n, p, mu)).sum()
