# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np
from scipy.special import expit, logit

from .distributions import Discrete
from ..internal.distribution_helper import all_not_none, eps
from ..internal.optimization import optimize_ml


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

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(n=1, p=self.p)
        return frozen

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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.atleast_1d(x)
        return nb_pdf(x, self.p)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.atleast_1d(x)
        return nb_cdf(x, self.p)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.atleast_1d(q)
        return nb_ppf(q, self.p)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        x = np.atleast_1d(x)
        return nb_logpdf(x, self.p)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.p)

    def entropy(self):
        return nb_entropy(self.p)

    def mean(self):
        return NotImplemented

    def median(self):
        return NotImplemented

    def var(self):
        return NotImplemented

    def std(self):
        return NotImplemented

    def skewness(self):
        return NotImplemented

    def kurtosis(self):
        return NotImplemented

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.choice(self.p, size)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


def nb_pdf(x, p):
    pmf = np.zeros_like(x, dtype=float)
    valid_categories = np.where((x >= 0) & (x < len(p)))[0]
    pmf[valid_categories] = p[x[valid_categories]]
    return pmf


def nb_cdf(x, p):
    x = np.asarray(x, dtype=int)
    cdf = np.ones_like(x, dtype=float)
    cdf[x < 0] = 0
    valid_categories = np.where((x >= 0) & (x < len(p)))[0]
    cdf[valid_categories] = np.cumsum(p)[x[valid_categories]]
    return cdf


def nb_ppf(q, p):
    cumsum = np.cumsum(p)
    return np.searchsorted(cumsum, q)


@nb.njit(cache=True)
def nb_entropy(p):
    return -np.sum(p * np.log(p))


def nb_logpdf(x, p):
    log_pmf = np.full_like(x, -np.inf, dtype=float)
    valid_categories = np.where((x >= 0) & (x < len(p)))[0]
    log_pmf[valid_categories] = np.log(p[x[valid_categories]])
    return log_pmf


def nb_neg_logpdf(x, p):
    return -(nb_logpdf(x, p)).sum()
