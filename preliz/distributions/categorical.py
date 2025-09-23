import numpy as np
from pytensor_distributions import categorical as ptd_categorical

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml
from preliz.internal.special import expit, logit


class Categorical(Discrete):
    R"""
    Categorical distribution.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::
        :context: close-figs


        from preliz import Categorical, style
        style.use('preliz-doc')
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
        self._parametrization(p, logit_p)

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

    def _update(self, p):
        self.p = np.array(p)
        self._n = len(p)
        self.logit_p = self._to_logit_p(self.p)

        if self.param_names[0] == "p":
            self.params = (self.p,)
        elif self.param_names[0] == "logit_p":
            self.params = (self.logit_p,)

        self.is_frozen = True

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

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, p):
    return ptd_categorical.pdf(x, p)


@pytensor_jit
def ptd_cdf(x, p):
    return ptd_categorical.cdf(x, p)


@pytensor_jit
def ptd_ppf(q, p):
    return ptd_categorical.ppf(q, p)


@pytensor_jit
def ptd_logpdf(x, p):
    return ptd_categorical.logpdf(x, p)


@pytensor_jit
def ptd_entropy(p):
    return ptd_categorical.entropy(p)


@pytensor_jit
def ptd_mean(p):
    return ptd_categorical.mean(p)


@pytensor_jit
def ptd_mode(p):
    return ptd_categorical.mode(p)


@pytensor_jit
def ptd_median(p):
    return ptd_categorical.median(p)


@pytensor_jit
def ptd_var(p):
    return ptd_categorical.var(p)


@pytensor_jit
def ptd_std(p):
    return ptd_categorical.std(p)


@pytensor_jit
def ptd_skewness(p):
    return ptd_categorical.skewness(p)


@pytensor_jit
def ptd_kurtosis(p):
    return ptd_categorical.kurtosis(p)


@pytensor_rng_jit
def ptd_rvs(p, size, rng):
    return ptd_categorical.rvs(p, size=size, random_state=rng)
