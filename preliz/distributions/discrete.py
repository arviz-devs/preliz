"""
Discrete probability distributions.
"""
import logging
from math import ceil, floor

import numpy as np
from scipy import stats


from .distributions import Discrete
from ..utils.optimization import optimize_matching_moments

_log = logging.getLogger("preliz")

eps = np.finfo(float).eps


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
        self.n = n
        self.p = p
        self.name = "binomial"
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.params_support = ((eps, np.inf), (eps, 1 - eps))
        self.dist = stats.binom
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.n, self.p)
        return frozen

    def _update(self, n, p):
        self.n = int(n)
        self.p = p
        self.params = (self.n, self.p)
        self.support = (0, self.n)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # crude approximation for n and p
        n = mean + sigma * 2
        p = mean / n
        self._update(n, p)
        optimize_matching_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        # see https://doi.org/10.1016/j.jspi.2004.02.019 for details
        x_bar = np.mean(sample)
        x_std = np.std(sample)
        x_max = np.max(sample)
        n = ceil(x_max ** (1.5) * x_std / (x_bar**0.5 * (x_max - x_bar) ** 0.5))
        p = x_bar / n
        self._update(n, p)


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

    def __init__(self, lower=-np.inf, upper=np.inf):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.name = "discreteuniform"
        self.params = (self.lower, self.upper)
        self.param_names = ("lower", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf))
        self.dist = stats.randint
        self.support = (lower, upper)
        self.dist.a = -np.inf
        self.dist.b = np.inf
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.lower, self.upper + 1)
        return frozen

    def _update(self, lower, upper):
        self.lower = floor(lower)
        self.upper = ceil(upper)
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


class NegativeBinomial(Discrete):
    R"""
    Negative binomial distribution.

    If it is parametrized in terms of n and p, the negative binomial describes
    the probability to have x failures before the n-th success, given the
    probability p of success in each trial. Its pmf is

    .. math::

        f(x \mid n, p) =
           \binom{x + n - 1}{x}
           (p)^n (1 - p)^x

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import NegativeBinomial
        az.style.use('arviz-white')
        ns = [5, 10, 10]
        ps = [0.5, 0.5, 0.7]
        for n, p in zip(ns, ps):
            NegativeBinomial(n, p).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\frac{(1-p) r}{p}`
    Variance  :math:`\frac{(1-p) r}{p^2}`
    ========  ==========================

    Parameters
    ----------
    n: float
        Number of target success trials (n > 0)
    p: float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n=None, p=None):
        super().__init__()
        self.n = n
        self.p = p
        self.name = "negativebinomial"
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.params_support = ((eps, np.inf), (eps, 1 - eps))
        self.dist = stats.nbinom
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.n, self.p)
        return frozen

    def _update(self, n, p):
        self.n = n
        self.p = p
        self.params = (self.n, p)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        n = mean**2 / (sigma**2 - mean)
        p = mean / sigma**2
        self._update(n, p)

    def _fit_mle(self, sample):
        # the upper bound is based on a quick heuristic. The fit will underestimate
        # the value of n when p is very close to 1.
        fitted = stats.fit(self.dist, sample, bounds={"n": (1, max(sample) * 5)})
        if not fitted.success:
            _log.info("Optimization did not terminate successfully.")
        self._update(fitted.params.n, fitted.params.p)  # pylint: disable=no-member


class Poisson(Discrete):
    R"""
    Poisson log-likelihood.

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
        self.name = "poisson"
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.params_support = ((eps, np.inf),)
        self.dist = stats.poisson
        self.support = (0, np.inf)
        self._update_rv_frozen()

    def _get_frozen(self):
        frozen = None
        if any(self.params):
            frozen = self.dist(self.mu)
        return frozen

    def _update(self, mu):
        self.mu = mu
        self.params = (self.mu,)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        mu = np.mean(sample)
        self._update(mu)
