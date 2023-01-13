# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init
"""
Discrete probability distributions.
"""
from copy import copy
import logging
from math import ceil

import numpy as np
from scipy import stats


from .distributions import Discrete
from ..internal.optimization import optimize_ml
from ..internal.distribution_helper import all_not_none


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
        ps = [0, 0.5, 0.8]
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
        if p is not None and logit_p is not None:
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
        p = np.e**logit_p / (1 + np.e**logit_p)
        return p

    def _to_logit_p(self, p):
        logit_p = np.log(p / (1 - p))
        return logit_p

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
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
        if (n and p) is not None:
            self._update(n, p)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
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
        self._update(n, p)

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
        if (lower and upper) is not None:
            self._update(lower, upper)
        else:
            self.lower = lower
            self.upper = upper

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
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
        if (mu or alpha) is not None and (p or n) is not None:
            raise ValueError("Incompatible parametrization. Either use mu and alpha, or p and n.")

        self.param_names = ("mu", "alpha")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if (p or n) is not None:
            self.p = p
            self.n = n
            self.param_names = ("p", "n")
            if (p and n) is not None:
                mu, alpha = self._from_p_n(p, n)

        self.mu = mu
        self.alpha = alpha
        if (mu and alpha) is not None:
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
        if all_not_none(self):
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
        mu = mean
        alpha = mean**2 / (sigma**2 - mean)
        self._update(mu, alpha)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


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
        self.dist = copy(stats.poisson)
        self.support = (0, np.inf)
        self._parametrization(mu)

    def _parametrization(self, mu=None):
        self.mu = mu
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.params_support = ((eps, np.inf),)
        if (mu) is not None:
            self._update(mu)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
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
