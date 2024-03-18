# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init
# pylint: disable=invalid-unary-operand-type
# pylint: disable=invalid-name
"""
Discrete probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats
from scipy.special import logit, expit, gamma  # pylint: disable=no-name-in-module

from .distributions import Discrete
from .bernoulli import Bernoulli  # pylint: disable=unused-import
from .binomial import Binomial  # pylint: disable=unused-import
from .poisson import Poisson  # pylint: disable=unused-import
from .negativebinomial import NegativeBinomial  # pylint: disable=unused-import
from .zi_binomial import ZeroInflatedBinomial  # pylint: disable=unused-import
from .zi_negativebinomial import ZeroInflatedNegativeBinomial  # pylint: disable=unused-import
from .zi_poisson import ZeroInflatedPoisson  # pylint: disable=unused-import

from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.distribution_helper import all_not_none


eps = np.finfo(float).eps


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
        az.style.use('arviz-doc')
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


class Categorical(Discrete):
    R"""
    Categorical distribution.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Categorical
        az.style.use('arviz-doc')
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
        az.style.use('arviz-doc')
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
        az.style.use('arviz-doc')
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
        az.style.use('arviz-doc')
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
        az.style.use('arviz-doc')
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
