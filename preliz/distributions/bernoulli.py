# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import logit, expit  # pylint: disable=no-name-in-module

from .distributions import Discrete
from ..internal.optimization import optimize_ml
from ..internal.distribution_helper import eps, all_not_none


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
        az.style.use('arviz-doc')
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

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        self._update(mean)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_pdf(x, self.p)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.p)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.p)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_logpdf(x, self.p)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.p)

    def entropy(self):
        return nb_entropy(self.p)

    def mean(self):
        return self.p

    def median(self):
        return np.where(self.p <= 0.5, 0, 1)

    def var(self):
        return self.p * self._q

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (self._q - self.p) / self.std()

    def kurtosis(self):
        return (1 - 6 * self.p * self._q) / (self.p * self._q)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.binomial(1, self.p, size=size)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, p):
    if x < 0:
        return 0
    elif x < 1:
        return 1 - p
    else:
        return 1


@nb.vectorize(nopython=True, cache=True)
def nb_ppf(q, p):
    if q < 0:
        return np.nan
    elif q > 1:
        return np.nan
    elif q == 0:
        return -1
    elif q < 1 - p:
        return 0
    else:
        return 1


@nb.vectorize(nopython=True, cache=True)
def nb_pdf(x, p):
    if x == 1:
        return p
    elif x == 0:
        return 1 - p
    else:
        return 0.0


@nb.njit(cache=True)
def nb_entropy(p):
    q = 1 - p
    return -q * np.log(q) - p * np.log(p)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, p):
    if x == 1:
        return np.log(p)
    elif x == 0:
        return np.log(1 - p)
    else:
        return -np.inf


@nb.njit(cache=True)
def nb_neg_logpdf(x, p):
    return -(nb_logpdf(x, p)).sum()
