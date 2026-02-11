import numpy as np
from pytensor_distributions import zi_binomial as ptd_zibinomial

from preliz.distributions.distributions import Discrete
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_mean_sigma, optimize_ml


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

        from preliz import ZeroInflatedBinomial, style
        style.use('preliz-doc')
        ns = [10, 20]
        ps = [0.5, 0.7]
        psis = [0.7, 0.4]
        for psi, n, p in zip(ns, ps, psis):
            ZeroInflatedBinomial(psi, n, p).plot_pdf(support=(0,25))

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi n p`
    Variance  :math:`\psi n p (1 - p) + n^2 p^2 (\psi - \psi^2)`
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

    def _update(self, psi, n, p):
        self.psi = np.float64(psi)
        self.n = np.int64(n)
        self.p = np.float64(p)
        self.params = (self.psi, self.n, self.p)
        if self.psi == 0:
            self.support = (0, 0)
        else:
            self.support = (0, self.n)
        self.is_frozen = True

    def pdf(self, x):
        x = np.asarray(x)
        result = ptd_pdf(x, self.psi, self.n, self.p)
        # Return 0 for values outside support or infinity, consistent with scipy.stats.binom
        result = np.where((x < 0) | (x > self.n) | ~np.isfinite(x), 0, result)
        return result

    def cdf(self, x):
        return ptd_cdf(x, self.psi, self.n, self.p)

    def ppf(self, q):
        return ptd_ppf(q, self.psi, self.n, self.p)

    def logpdf(self, x):
        return ptd_logpdf(x, self.psi, self.n, self.p)

    def entropy(self):
        return ptd_entropy(self.psi, self.n, self.p)

    def mean(self):
        return ptd_mean(self.psi, self.n, self.p)

    def mode(self):
        return ptd_mode(self.psi, self.n, self.p)

    def median(self):
        return ptd_median(self.psi, self.n, self.p)

    def var(self):
        return ptd_var(self.psi, self.n, self.p)

    def std(self):
        return ptd_std(self.psi, self.n, self.p)

    def skewness(self):
        return ptd_skewness(self.psi, self.n, self.p)

    def kurtosis(self):
        return ptd_kurtosis(self.psi, self.n, self.p)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.psi, self.n, self.p, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        optimize_mean_sigma(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, psi, n, p):
    return ptd_zibinomial.pdf(x, psi, n, p)


@pytensor_jit
def ptd_cdf(x, psi, n, p):
    return ptd_zibinomial.cdf(x, psi, n, p)


@pytensor_jit
def ptd_ppf(q, psi, n, p):
    return ptd_zibinomial.ppf(q, psi, n, p)


@pytensor_jit
def ptd_logpdf(x, psi, n, p):
    return ptd_zibinomial.logpdf(x, psi, n, p)


@pytensor_jit
def ptd_entropy(psi, n, p):
    return ptd_zibinomial.entropy(psi, n, p)


@pytensor_jit
def ptd_mean(psi, n, p):
    return ptd_zibinomial.mean(psi, n, p)


@pytensor_jit
def ptd_mode(psi, n, p):
    return ptd_zibinomial.mode(psi, n, p)


@pytensor_jit
def ptd_median(psi, n, p):
    return ptd_zibinomial.median(psi, n, p)


@pytensor_jit
def ptd_var(psi, n, p):
    return ptd_zibinomial.var(psi, n, p)


@pytensor_jit
def ptd_std(psi, n, p):
    return ptd_zibinomial.std(psi, n, p)


@pytensor_jit
def ptd_skewness(psi, n, p):
    return ptd_zibinomial.skewness(psi, n, p)


@pytensor_jit
def ptd_kurtosis(psi, n, p):
    return ptd_zibinomial.kurtosis(psi, n, p)


@pytensor_rng_jit
def ptd_rvs(psi, n, p, size, rng):
    return ptd_zibinomial.rvs(psi, n, p, size=size, random_state=rng)
