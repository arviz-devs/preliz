# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init
# pylint: disable=invalid-unary-operand-type
# pylint: disable=invalid-name
"""
Discrete probability distributions.
"""
from copy import copy
import logging
from math import ceil

import numpy as np
from scipy import stats
from scipy.special import logit, expit, gamma  # pylint: disable=no-name-in-module


from .distributions import Discrete
from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.distribution_helper import all_not_none, any_not_none


_log = logging.getLogger("preliz")

eps = np.finfo(float).eps


class Bernoulli(Discrete):
    R"""Bernoulli distribution

    The Bernoulli distribution describes the probability of successes (x=1) and failures (x=0).
    The pmf of this distribution is

    .. math::
        f(x \mid p) = p^{x} (1-p)^{1-x}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Bernoulli
        az.style.use('arviz-white')
        for p in [0, 0.5, 0.8]:
            Bernoulli(p).plot_pdf()

    ========  ======================
    Support   :math:`x \in \{0, 1\}`
    Mean      :math:`p`
    Variance  :math:`p (1 - p)`
    ========  ======================

    The Bernoulli distribution has 2 alternative parametrizations. In terms of p or logit_p.

    The link between the 2 alternatives is given by

    .. math::

        logit(p) = ln(\frac{p}{1-p})

    Parameters
    ----------
    p : float
        Probability of success (0 < p < 1).
    logit_p : float
        Alternative log odds for the probability of success.
    """

    def __init__(self, p=None, logit_p=None):
        super().__init__()
        self.dist = copy(stats.bernoulli)
        self.support = (0, 1)
        self._parametrization(p, logit_p)

    def _parametrization(self, p=None, logit_p=None):
        if all_not_none(p, logit_p):
            raise ValueError("Incompatible parametrization. Either use p or logit_p.")

        self.param_names = "p"
        self.params_support = ((eps, 1),)

        if logit_p is not None:
            p = self._from_logit_p(logit_p)
            self.param_names = ("logit_p",)

        self.p = p
        self.logit_p = logit_p
        if self.p is not None:
            self._update(self.p)

    def _from_logit_p(self, logit_p):
        return expit(logit_p)

    def _to_logit_p(self, p):
        return logit(p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.p)
        return frozen

    def _update(self, p):
        self.p = np.float64(p)
        self.logit_p = self._to_logit_p(p)

        if self.param_names[0] == "p":
            self.params = (self.p,)
        elif self.param_names[0] == "logit_p":
            self.params = (self.logit_p,)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        p = mean
        self._update(p)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class BetaBinomial(Discrete):
    R"""
    Beta-binomial distribution.

    Equivalent to binomial random variable with success probability
    drawn from a beta distribution.

    The pmf of this distribution is

    .. math::

       f(x \mid \alpha, \beta, n) =
           \binom{n}{x}
           \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import BetaBinomial
        az.style.use('arviz-white')
        alphas = [0.5, 1, 2.3]
        betas = [0.5, 1, 2]
        n = 10
        for a, b in zip(alphas, betas):
            BetaBinomial(a, b, n).plot_pdf()

    ========  =================================================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n \dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{n \alpha \beta (\alpha+\beta+n)}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
    ========  =================================================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.
    """

    def __init__(self, alpha=None, beta=None, n=None):
        super().__init__()
        self.dist = copy(stats.betabinom)
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, n)

    def _parametrization(self, alpha=None, beta=None, n=None):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.params = (self.alpha, self.beta, self.n)
        self.param_names = ("alpha", "beta", "n")
        self.params_support = ((eps, np.inf), (eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta, n)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(n=self.n, a=self.alpha, b=self.beta)
        return frozen

    def _update(self, alpha, beta, n):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.n = np.int64(n)
        self.params = (self.alpha, self.beta, self.n)
        self.support = (0, self.n)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Crude aproximation for n (as in Binomial distribution)
        # For alpha and beta see:
        # https://en.wikipedia.org/wiki/Beta-binomial_distribution#Method_of_moments
        n = mean + sigma * 2
        p = mean / n
        rho = ((sigma**2 / (mean * (1 - p))) - 1) / (n - 1)
        alpha = max(0.5, (p / rho) - p)
        beta = max(0.5, (alpha / p) - alpha)
        params = alpha, beta, n
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class Binomial(Discrete):
    R"""
    Binomial distribution.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.

    The pmf of this distribution is

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Binomial
        az.style.use('arviz-white')
        ns = [5, 10, 10]
        ps = [0.5, 0.5, 0.7]
        for n, p in zip(ns, ps):
            Binomial(n, p).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n p`
    Variance  :math:`n p (1 - p)`
    ========  ==========================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n=None, p=None):
        super().__init__()
        self.dist = copy(stats.binom)
        self.support = (0, np.inf)
        self._parametrization(n, p)

    def _parametrization(self, n=None, p=None):
        self.n = n
        self.p = p
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.params_support = ((eps, np.inf), (eps, 1 - eps))
        if all_not_none(n, p):
            self._update(n, p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.n, self.p)
        return frozen

    def _update(self, n, p):
        self.n = np.int64(n)
        self.p = np.float64(p)
        self.params = (self.n, self.p)
        self.support = (0, self.n)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p
        n = mean + sigma * 2
        p = mean / n
        params = n, p
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        # see https://doi.org/10.1016/j.jspi.2004.02.019 for details
        x_bar = np.mean(sample)
        x_std = np.std(sample)
        x_max = np.max(sample)
        n = ceil(x_max ** (1.5) * x_std / (x_bar**0.5 * (x_max - x_bar) ** 0.5))
        p = x_bar / n
        self._update(n, p)


class Categorical(Discrete):
    R"""
    Categorical distribution.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Categorical
        az.style.use('arviz-white')
        ps = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
        for p in ps:
            Categorical(p).plot_pdf()

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1.
    logit_p : float
        Alternative log odds for the probability of success.
    """

    def __init__(self, p=None, logit_p=None):
        super().__init__()
        self.dist = copy(stats.multinomial)
        self._parametrization(p, logit_p)

    def pdf(self, x):  # pylint: disable=arguments-differ
        x = np.asarray(x)
        pmf = np.zeros_like(x, dtype=float)
        valid_categories = np.where((x >= 0) & (x < len(self.p)))[0]
        pmf[valid_categories] = self.p[x[valid_categories]]
        return pmf

    def cdf(self, x):  # pylint: disable=arguments-differ
        x = np.asarray(x, dtype=int)
        cdf = np.ones_like(x, dtype=float)
        cdf[x < 0] = 0
        valid_categories = np.where((x >= 0) & (x < len(self.p)))[0]
        cdf[valid_categories] = np.cumsum(self.p)[x[valid_categories]]
        return cdf

    def ppf(self, q):  # pylint: disable=arguments-differ
        cumsum = np.cumsum(self.p)
        return np.searchsorted(cumsum, q)

    def _parametrization(self, p=None, logit_p=None):
        if all_not_none(p, logit_p):
            raise ValueError("Incompatible parametrization. Either use p or logit_p.")

        self.param_names = "p"
        self.params_support = ((eps, np.inf),)

        if logit_p is not None:
            p = self._from_logit_p(logit_p)
            self.param_names = ("logit_p",)

        self.p = p
        self.logit_p = logit_p
        if self.p is not None:
            self.support = (0, len(p) - 1)
            self._update(self.p)

    def _from_logit_p(self, logit_p):
        return expit(logit_p)

    def _to_logit_p(self, p):
        return logit(p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(n=1, p=self.p)
        return frozen

    def _update(self, p):
        self.p = np.array(p)
        self.logit_p = self._to_logit_p(p)

        if self.param_names[0] == "p":
            self.params = (self.p,)
        elif self.param_names[0] == "logit_p":
            self.params = (self.logit_p,)

        self._update_rv_frozen()

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class DiscreteUniform(Discrete):
    R"""
    Discrete Uniform distribution.

    The pmf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower+1}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import DiscreteUniform
        az.style.use('arviz-white')
        ls = [1, -2]
        us = [6, 2]
        for l, u in zip(ls, us):
            ax = DiscreteUniform(l, u).plot_pdf()
            ax.set_ylim(0, 0.25)

    ========  ===============================================
    Support   :math:`x \in {lower, lower + 1, \ldots, upper}`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower + 1)^2 - 1}{12}`
    ========  ===============================================

    Parameters
    ----------
    lower: int
        Lower limit.
    upper: int
        Upper limit (upper > lower).
    """

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.dist = copy(stats.randint)
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
        if all_not_none(lower, upper):
            self._update(lower, upper)
        else:
            self.lower = lower
            self.upper = upper

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.lower, self.upper + 1)
        return frozen

    def _update(self, lower, upper):
        self.lower = np.floor(lower)
        self.upper = np.ceil(upper)
        self.params = (self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        spr = (12 * sigma**2 + 1) ** 0.5
        lower = 0.5 * (2 * mean - spr + 1)
        upper = 0.5 * (2 * mean + spr - 1)
        self._update(lower, upper)

    def _fit_mle(self, sample):
        lower = np.min(sample)
        upper = np.max(sample)
        self._update(lower, upper)


class DiscreteWeibull(Discrete):
    R"""
    Discrete Weibull distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x+1)^{\beta}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import DiscreteWeibull
        az.style.use('arviz-white')
        qs = [0.1, 0.9, 0.9]
        betas = [0.3, 1.3, 3]
        for q, b in zip(qs, betas):
            DiscreteWeibull(q, b).plot_pdf(support=(0,10))

    ========  ===============================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu = \sum_{x = 1}^{\infty} q^{x^{\beta}}`
    Variance  :math:`2 \sum_{x = 1}^{\infty} x q^{x^{\beta}} - \mu - \mu^2`
    ========  ===============================================

    Parameters
    ----------
    q: float
        Shape parameter (0 < q < 1).
    beta: float
        Shape parameter (beta > 0).
    """

    def __init__(self, q=None, beta=None):
        super().__init__()
        self.dist = _DiscreteWeibull
        self.support = (0, np.inf)
        self._parametrization(q, beta)

    def _parametrization(self, q=None, beta=None):
        self.q = q
        self.beta = beta
        self.params = (self.q, self.beta)
        self.param_names = ("q", "beta")
        self.params_support = ((eps, 1 - eps), (eps, np.inf))
        if all_not_none(q, beta):
            self._update(q, beta)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.q, self.beta)
        return frozen

    def _update(self, q, beta):
        self.q = np.float64(q)
        self.beta = np.float64(beta)
        self.support = (0, np.inf)
        self.params = (self.q, self.beta)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class _DiscreteWeibull(stats.rv_continuous):
    def __init__(self, q=None, beta=None):
        super().__init__()
        self.q = q
        self.beta = beta

    def support(self, *args, **kwds):  # pylint: disable=unused-argument
        return (0, np.inf)

    def cdf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        x = np.asarray(x)
        return np.nan_to_num(1 - self.q ** ((x + 1) ** self.beta))

    def pmf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        x = np.asarray(x)
        return self.q ** (x**self.beta) - self.q ** ((x + 1) ** self.beta)

    def logpmf(self, x, *args, **kwds):  # pylint: disable=unused-argument
        return np.log(self.pmf(x, *args, **kwds))

    def ppf(self, p, *args, **kwds):  # pylint: disable=arguments-differ unused-argument
        p = np.asarray(p)
        p[p == 1] = 0.999999
        ppf = np.ceil((np.log(1 - p) / np.log(self.q)) ** (1 / self.beta) - 1)
        return ppf

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        x_max = np.nan_to_num(self._ppf(0.999), nan=1)
        if x_max < 10000:
            x_range = np.arange(1, x_max + 1, dtype=int)
            mean = np.sum(self.q ** (x_range**self.beta))
            var = 2 * np.sum(x_range * self.q ** (x_range**self.beta)) - mean - mean**2
        else:
            lam = (-1 / np.log(self.q)) ** (1 / self.beta)
            kappa = gamma(1 + 1 / self.beta)
            mean = lam * kappa - 0.5
            var = lam**2 * (gamma(1 + 2 / self.beta) - (kappa**2)) - 1
        return (mean, var, np.nan, np.nan)

    def entropy(self):  # pylint: disable=arguments-differ
        entropy = 0.0
        x = 0
        while True:
            p_x = self.q ** (x**self.beta) - self.q ** ((x + 1) ** self.beta)
            if p_x < 1e-6:
                break
            entropy -= p_x * np.log(p_x)
            x += 1
        return entropy

        # return self.q / np.log(self.beta)

    def rvs(self, size=1, random_state=None):  # pylint: disable=arguments-differ
        return self.ppf(np.random.uniform(size=size), random_state=random_state)


class Geometric(Discrete):
    R"""
    Geometric distribution.

    The probability that the first success in a sequence of Bernoulli trials
    occurs on the x'th trial.
    The pmf of this distribution is

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Geometric
        az.style.use('arviz-white')
        for p in [0.1, 0.25, 0.75]:
            Geometric(p).plot_pdf(support=(1,10))

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{>0}`
    Mean      :math:`\dfrac{1}{p}`
    Variance  :math:`\dfrac{1 - p}{p^2}`
    ========  =============================

    Parameters
    ----------
    p : float
        Probability of success on an individual trial (0 < p <= 1).
    """

    def __init__(self, p=None):
        super().__init__()
        self.dist = copy(stats.geom)
        self.support = (eps, np.inf)
        self._parametrization(p)

    def _parametrization(self, p=None):
        self.p = p
        self.param_names = "p"
        self.params_support = ((eps, 1),)
        if self.p is not None:
            self._update(self.p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.p)
        return frozen

    def _update(self, p):
        self.p = np.float64(p)
        self.params = (self.p,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        p = 1 / mean
        self._update(p)

    def _fit_mle(self, sample):
        mean = np.mean(sample)
        p = 1 / mean
        self._update(p)


class HyperGeometric(Discrete):
    R"""
    Discrete hypergeometric distribution.

    The probability of :math:`x` successes in a sequence of :math:`n` bernoulli
    trials taken without replacement from a population of :math:`N` objects,
    containing :math:`k` good (or successful or Type I) objects.
    The pmf of this distribution is

    .. math:: f(x \mid N, n, k) = \frac{\binom{k}{x}\binom{N-k}{n-x}}{\binom{N}{n}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HyperGeometric
        az.style.use('arviz-white')
        N = 50
        k = 10
        for n in [20, 25]:
            HyperGeometric(N, k, n).plot_pdf(support=(1,15))

    ========  =============================
    Support   :math:`x \in \left[\max(0, n - N + k), \min(k, n)\right]`
    Mean      :math:`\dfrac{nk}{N}`
    Variance  :math:`\dfrac{(N-n)nk(N-k)}{(N-1)N^2}`
    ========  =============================

    Parameters
    ----------
    N : int
        Total size of the population (N > 0)
    k : int
        Number of successful individuals in the population (0 <= k <= N)
    n : int
        Number of samples drawn from the population (0 <= n <= N)
    """

    def __init__(self, N=None, k=None, n=None):
        super().__init__()
        self.dist = copy(stats.hypergeom)
        self._parametrization(N, k, n)
        self.support = (0, np.inf)

    def _parametrization(self, N=None, k=None, n=None):
        self.N = N
        self.k = k
        self.n = n
        self.param_names = ("N", "k", "n")
        self.params_support = ((eps, np.inf), (eps, self.N), (eps, self.N))
        if all_not_none(self.N, self.k, self.n):
            self._update(N, k, n)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(M=self.N, N=self.n, n=self.k)
        return frozen

    def _update(self, N, k, n):
        self.N = np.int64(N)
        self.k = np.int64(k)
        self.n = np.int64(n)
        self.params = (self.N, self.k, self.n)
        self.support = (max(0, n - N + k), min(k, n))
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        n = mean + sigma * 4
        k = n
        N = k * n / mean
        params = N, k, n
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


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
        az.style.use('arviz-white')
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
        self.dist = copy(stats.nbinom)
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
                mu, alpha = self._from_p_n(p, n)

        self.mu = mu
        self.alpha = alpha
        if all_not_none(mu, alpha):
            self._update(mu, alpha)

    def _from_p_n(self, p, n):
        alpha = n
        mu = n * (1 / p - 1)
        return mu, alpha

    def _to_p_n(self, mu, alpha):
        p = alpha / (mu + alpha)
        n = alpha
        return p, n

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.n, self.p)
        return frozen

    def _update(self, mu, alpha):
        self.mu = np.float64(mu)
        self.alpha = np.float64(alpha)
        self.p, self.n = self._to_p_n(self.mu, self.alpha)

        if self.param_names[0] == "mu":
            self.params = (self.mu, self.alpha)
        elif self.param_names[0] == "p":
            self.params = (self.p, self.n)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class Poisson(Discrete):
    R"""
    Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Poisson
        az.style.use('arviz-white')
        for mu in [0.5, 3, 8]:
            Poisson(mu).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """

    def __init__(self, mu=None):
        super().__init__()
        self.mu = mu
        self.dist = copy(stats.poisson)
        self.support = (0, np.inf)
        self._parametrization(mu)

    def _parametrization(self, mu=None):
        self.mu = mu
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.params_support = ((eps, np.inf),)
        if mu is not None:
            self._update(mu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.mu)
        return frozen

    def _update(self, mu):
        self.mu = np.float64(mu)
        self.params = (self.mu,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        mu = np.mean(sample)
        self._update(mu)


class ZeroInflatedBinomial(Discrete):
    R"""
    Zero-inflated Binomial distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
            (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
            \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
            \end{array} \right.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ZeroInflatedBinomial
        az.style.use('arviz-white')
        ns = [10, 20]
        ps = [0.5, 0.7]
        psis = [0.7, 0.4]
        for n, p, psi in zip(ns, ps, psis):
            ZeroInflatedBinomial(psi, n, p).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi n p`
    Variance  :math:`(1-\psi) n p [1 - p(1 - \psi n)].`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of Binomial variates (0 < psi < 1)
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, psi=None, n=None, p=None):
        super().__init__()
        self.psi = psi
        self.n = n
        self.p = p
        self.dist = _ZIBinomial
        self.support = (0, np.inf)
        self._parametrization(psi, n, p)

    def _parametrization(self, psi=None, n=None, p=None):
        self.psi = psi
        self.n = n
        self.p = p
        self.params = (self.psi, self.n, self.p)
        self.param_names = ("psi", "n", "p")
        self.params_support = ((eps, 1 - eps), (eps, np.inf), (eps, 1 - eps))
        if all_not_none(psi, n, p):
            self._update(psi, n, p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.psi, self.n, self.p)
        return frozen

    def _update(self, psi, n, p):
        self.psi = np.float64(psi)
        self.n = np.int64(n)
        self.p = np.float64(p)
        self.params = (self.psi, self.n, self.p)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p (same as Binomial)
        n = mean + sigma * 2
        p = mean / n
        psi = 0.9
        params = psi, n, p
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


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
        az.style.use('arviz-white')
        psis = [0.7, 0.7]
        mus = [2, 8]
        alphas = [2, 4]
        for psi, mu, alpha in zip(psis, mus, alphas):
        ZeroInflatedNegativeBinomial(psi, mu=mu, alpha=alpha).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Var       :math:`\psi\mu +  \left (1 + \frac{\mu}{\alpha} + \frac{1-\psi}{\mu} \right)`
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
        self.psi = psi
        self.n = n
        self.p = p
        self.alpha = alpha
        self.mu = mu
        self.dist = _ZINegativeBinomial
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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.psi, self.p, self.n)
        return frozen

    def _update(self, psi, mu, alpha):
        self.psi = np.float64(psi)
        self.mu = np.float64(mu)
        self.alpha = np.float64(alpha)
        self.p, self.n = self._to_p_n(self.mu, self.alpha)

        if self.param_names[1] == "mu":
            self.params = (self.psi, self.mu, self.alpha)
        elif self.param_names[1] == "p":
            self.params = (self.psi, self.p, self.n)

        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        psi = 0.9
        mu = mean / psi
        alpha = mean**2 / (sigma**2 - mean)
        params = psi, mu, alpha
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class ZeroInflatedPoisson(Discrete):
    R"""
    Zero-inflated Poisson distribution.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
            \psi \frac{e^{-\mu}\mu^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import ZeroInflatedPoisson
        az.style.use('arviz-white')
        psis = [0.7, 0.4]
        mus = [8, 4]
        for psi, mu in zip(psis, mus):
            ZeroInflatedPoisson(psi, mu).plot_pdf()

    ========  ================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi \mu`
    Variance  :math:`\psi \mu (1+(1-\psi) \mu`
    ========  ================================

    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    mu : float
        Expected number of occurrences during the given interval
        (mu >= 0).
    """

    def __init__(self, psi=None, mu=None):
        super().__init__()
        self.psi = psi
        self.mu = mu
        self.dist = _ZIPoisson
        self.support = (0, np.inf)
        self._parametrization(psi, mu)

    def _parametrization(self, psi=None, mu=None):
        self.psi = psi
        self.mu = mu
        self.params = (self.psi, self.mu)
        self.param_names = ("psi", "mu")
        self.params_support = ((eps, 1 - eps), (eps, np.inf))
        if all_not_none(psi, mu):
            self._update(psi, mu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.psi, self.mu)
        return frozen

    def _update(self, psi, mu):
        self.psi = np.float64(psi)
        self.mu = np.float64(mu)
        self.params = (self.psi, self.mu)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        psi = min(0.99, max(0.01, mean**2 / (mean**2 - mean + sigma**2)))
        mean = mean / psi
        self._update(psi, mean)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


class _ZIBinomial(stats.rv_continuous):
    def __init__(self, psi=None, n=None, p=None):
        super().__init__()
        self.psi = psi
        self.n = n
        self.p = p

    def support(self, *args, **kwd):  # pylint: disable=unused-argument
        return (0, np.inf)

    def cdf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return (1 - self.psi) + self.psi * stats.binom(self.n, self.p, *args, **kwds).cdf(x)

    def pmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        x = np.array(x, ndmin=1)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = (1 - self.psi) + self.psi * (1 - self.p) ** self.n
        result[x != 0] = self.psi * stats.binom(self.n, self.p, *args, **kwds).pmf(x[x != 0])
        return result

    def logpmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = np.log((1 - self.psi) + self.psi * (1 - self.p) ** self.n)
        result[x != 0] = np.log(self.psi) + stats.binom(self.n, self.p, *args, **kwds).logpmf(
            x[x != 0]
        )
        return result

    def ppf(self, q, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return np.round(
            (1 - self.psi) + self.psi * stats.binom(self.n, self.p, *args, **kwds).ppf(q)
        )

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        if psi_not_valid(self.psi):
            return (np.nan, np.nan, np.nan, np.nan)
        mean = self.psi * self.n * self.p
        var = (1 - self.psi) * self.n * self.p * (1 - self.p * (1 - self.psi * self.n))
        return (mean, var, np.nan, np.nan)

    def entropy(self):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        binomial_entropy = stats.binom.entropy(self.n, self.p)
        # The variable can be 0 with probability 1-psi or something else with probability psi
        zero_entropy = -(1 - self.psi) * np.log(1 - self.psi) - self.psi * np.log(self.psi)
        # The total entropy is the weighted sum of the two entropies
        return (1 - self.psi) * zero_entropy + self.psi * binomial_entropy

    def rvs(self, size=1):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        samples = np.zeros(size, dtype=int)
        non_zero_indices = np.where(np.random.uniform(size=size) < (self.psi))[0]
        samples[~non_zero_indices] = 0
        samples[non_zero_indices] = stats.binom.rvs(self.n, self.p, size=len(non_zero_indices))
        return samples


class _ZINegativeBinomial(stats.rv_continuous):
    def __init__(self, psi=None, p=None, n=None):
        super().__init__()
        self.psi = psi
        self.n = n
        self.p = p
        self.mu = self.n * (1 / self.p - 1)

    def support(self, *args, **kwd):  # pylint: disable=unused-argument
        return (0, np.inf)

    def cdf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return (1 - self.psi) + self.psi * stats.nbinom(self.n, self.p, *args, **kwds).cdf(x)

    def pmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        x = np.array(x, ndmin=1)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = (1 - self.psi) + self.psi * (self.n / (self.n + self.mu)) ** self.n
        result[x != 0] = self.psi * stats.nbinom(self.n, self.p, *args, **kwds).pmf(x[x != 0])
        return result

    def logpmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = np.log((1 - self.psi) + self.psi * (self.n / (self.n + self.mu)) ** self.n)
        result[x != 0] = np.log(self.psi) + stats.nbinom(self.n, self.p, *args, **kwds).logpmf(
            x[x != 0]
        )
        return result

    def ppf(self, q, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return np.round(
            (1 - self.psi) + self.psi * stats.nbinom(self.n, self.p, *args, **kwds).ppf(q)
        )

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        if psi_not_valid(self.psi):
            return (np.nan, np.nan, np.nan, np.nan)
        mean = self.psi * self.mu
        var = self.psi * self.mu + (1 + (self.mu / self.n) + ((1 - self.psi) / self.mu))
        return (mean, var, np.nan, np.nan)

    def entropy(self):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        negative_binomial_entropy = stats.nbinom.entropy(self.n, self.p)
        # The variable can be 0 with probability 1-psi or something else with probability psi
        zero_entropy = -(1 - self.psi) * np.log(1 - self.psi) - self.psi * np.log(self.psi)
        # The total entropy is the weighted sum of the two entropies
        return (1 - self.psi) * zero_entropy + self.psi * negative_binomial_entropy

    def rvs(self, size=1):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        samples = np.zeros(size, dtype=int)
        non_zero_indices = np.where(np.random.uniform(size=size) < (self.psi))[0]
        samples[~non_zero_indices] = 0
        samples[non_zero_indices] = stats.nbinom.rvs(self.n, self.p, size=len(non_zero_indices))
        return samples


class _ZIPoisson(stats.rv_continuous):
    def __init__(self, psi=None, mu=None):
        super().__init__()
        self.psi = psi
        self.mu = mu

    def support(self, *args, **kwd):  # pylint: disable=unused-argument
        return (0, np.inf)

    def cdf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return (1 - self.psi) + self.psi * stats.poisson(self.mu, *args, **kwds).cdf(x)

    def pmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        x = np.array(x, ndmin=1)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = (1 - self.psi) + self.psi * np.exp(-self.mu)
        result[x != 0] = self.psi * stats.poisson(self.mu, *args, **kwds).pmf(x[x != 0])
        return result

    def logpmf(self, x, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.full(len(x), np.nan)
        result = np.zeros_like(x, dtype=float)
        result[x == 0] = np.log(np.exp(-self.mu) * self.psi - self.psi + 1)
        result[x != 0] = np.log(self.psi) + stats.poisson(self.mu, *args, **kwds).logpmf(x[x != 0])
        return result

    def ppf(self, q, *args, **kwds):
        if psi_not_valid(self.psi):
            return np.nan
        return np.round((1 - self.psi) + self.psi * stats.poisson(self.mu, *args, **kwds).ppf(q))

    def _stats(self, *args, **kwds):  # pylint: disable=unused-argument
        if psi_not_valid(self.psi):
            return (np.nan, np.nan, np.nan, np.nan)
        mean = self.psi * self.mu
        var = self.psi * self.mu * (1 + (1 - self.psi) * self.mu)
        return (mean, var, np.nan, np.nan)

    def entropy(self):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        poisson_entropy = stats.poisson.entropy(self.mu)
        # The variable can be 0 with probability 1-psi or something else with probability psi
        zero_entropy = -(1 - self.psi) * np.log(1 - self.psi) - self.psi * np.log(self.psi)
        # The total entropy is the weighted sum of the two entropies
        return (1 - self.psi) * zero_entropy + self.psi * poisson_entropy

    def rvs(self, size=1):  # pylint: disable=arguments-differ
        if psi_not_valid(self.psi):
            return np.nan
        samples = np.zeros(size, dtype=int)
        non_zero_indices = np.where(np.random.uniform(size=size) < (self.psi))[0]
        samples[~non_zero_indices] = 0
        samples[non_zero_indices] = stats.poisson.rvs(self.mu, size=len(non_zero_indices))
        return samples


def psi_not_valid(psi):
    return not 0 <= psi <= 1
