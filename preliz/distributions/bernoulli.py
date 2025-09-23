import numpy as np
from pytensor_distributions import bernoulli as ptd_bernoulli

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import expit, logit


class Bernoulli(Discrete):
    R"""Bernoulli distribution.

    The Bernoulli distribution describes the probability of successes (x=1) and failures (x=0).
    The pmf of this distribution is

    .. math::
        f(x \mid p) = p^{x} (1-p)^{1-x}

    .. plot::
        :context: close-figs


        from preliz import Bernoulli, style
        style.use('preliz-doc')
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

    def _update(self, p):
        self.p = np.float64(p)
        self._q = 1 - self.p
        self.logit_p = self._to_logit_p(p)

        if self.param_names[0] == "p":
            self.params = (self.p,)
        elif self.param_names[0] == "logit_p":
            self.params = (self.logit_p,)

        self.is_frozen = True

    def _fit_moments(self, mean, sigma):
        self._update(mean)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)

    def pdf(self, x):
        return ptd_pdf(x, self.p)

    def cdf(self, x):
        return ptd_cdf(x, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.p)

    def entropy(self):
        return ptd_entropy(self.p)

    def mean(self):
        return ptd_mean(self.p)

    def mode(self):
        return ptd_mode(self.p)

    def median(self):
        return ptd_median(self.p)

    def var(self):
        return ptd_var(self.p)

    def std(self):
        return ptd_std(self.p)

    def skewness(self):
        return ptd_skewness(self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.p, size=size, rng=random_state)


@pytensor_jit
def ptd_pdf(x, p):
    return ptd_bernoulli.pdf(x, p)


@pytensor_jit
def ptd_cdf(x, p):
    return ptd_bernoulli.cdf(x, p)


@pytensor_jit
def ptd_ppf(q, p):
    return ptd_bernoulli.ppf(q, p)


@pytensor_jit
def ptd_logpdf(x, p):
    return ptd_bernoulli.logpdf(x, p)


@pytensor_jit
def ptd_entropy(p):
    return ptd_bernoulli.entropy(p)


@pytensor_jit
def ptd_mean(p):
    return ptd_bernoulli.mean(p)


@pytensor_jit
def ptd_mode(p):
    return ptd_bernoulli.mode(p)


@pytensor_jit
def ptd_median(p):
    return ptd_bernoulli.median(p)


@pytensor_jit
def ptd_var(p):
    return ptd_bernoulli.var(p)


@pytensor_jit
def ptd_std(p):
    return ptd_bernoulli.std(p)


@pytensor_jit
def ptd_skewness(p):
    return ptd_bernoulli.skewness(p)


@pytensor_jit
def ptd_kurtosis(p):
    return ptd_bernoulli.kurtosis(p)


@pytensor_rng_jit
def ptd_rvs(p, size, rng):
    return ptd_bernoulli.rvs(p, size=size, random_state=rng)
