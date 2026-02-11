import numpy as np
from pytensor_distributions import triangular as ptd_triangular

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, pytensor_jit, pytensor_rng_jit


class Triangular(Continuous):
    r"""
    Triangular distribution.

    The pdf of this distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < a, \\
            \frac{2(x-a)}{(b-a)(c-a)} & \text{for } a \le x < c, \\[4pt]
            \frac{2}{b-a}             & \text{for } x = c, \\[4pt]
            \frac{2(b-x)}{(b-a)(b-c)} & \text{for } c < x \le b, \\[4pt]
            0 & \text{for } b < x.
        \end{cases}

    .. plot::
        :context: close-figs

        from preliz import Triangular, style
        style.use('preliz-doc')
        lowers = [0., -1, 2]
        cs = [2., 0., 6.5]
        uppers = [4., 1, 8]
        for lower, c, upper in zip(lowers, cs, uppers):
            scale = upper - lower
            c_ = (c - lower) / scale
            Triangular(lower, c, upper).plot_pdf()

    ========  ============================================================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper + c}{3}`
    Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
    ========  ============================================================================

    Parameters
    ----------
    lower : float
        Lower limit.
    c : float
        Mode.
    upper : float
        Upper limit.
    """

    def __init__(self, lower=None, c=None, upper=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(lower, c, upper)

    def _parametrization(self, lower=None, c=None, upper=None):
        self.lower = lower
        self.c = c
        self.upper = upper
        self.params = (self.lower, self.c, self.upper)
        self.param_names = ("lower", "c", "upper")
        self.params_support = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        if all_not_none(lower, c, upper):
            self._update(lower, c, upper)

    def _update(self, lower, c, upper):
        self.lower = np.float64(lower)
        self.c = np.float64(c)
        self.upper = np.float64(upper)
        self.support = (self.lower, self.upper)
        self.params = (self.lower, self.c, self.upper)
        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.lower, self.c, self.upper)

    def cdf(self, x):
        return ptd_cdf(x, self.lower, self.c, self.upper)

    def ppf(self, q):
        return ptd_ppf(q, self.lower, self.c, self.upper)

    def logpdf(self, x):
        return ptd_logpdf(x, self.lower, self.c, self.upper)

    def entropy(self):
        return ptd_entropy(self.lower, self.c, self.upper)

    def mean(self):
        return ptd_mean(self.lower, self.c, self.upper)

    def mode(self):
        return ptd_mode(self.lower, self.c, self.upper)

    def median(self):
        return ptd_median(self.lower, self.c, self.upper)

    def var(self):
        return ptd_var(self.lower, self.c, self.upper)

    def std(self):
        return ptd_std(self.lower, self.c, self.upper)

    def skewness(self):
        return ptd_skewness(self.lower, self.c, self.upper)

    def kurtosis(self):
        return ptd_kurtosis(self.lower, self.c, self.upper)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.lower, self.c, self.upper, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        # Assume symmetry
        lower = mean - 6**0.5 * sigma
        upper = mean + 6**0.5 * sigma
        c = mean
        self._update(lower, c, upper)

    def _fit_mle(self, sample):
        lower = np.min(sample)
        upper = np.max(sample)
        middle = (np.mean(sample) * 3) - lower - upper
        self._update(lower, middle, upper)


@pytensor_jit
def ptd_pdf(x, lower, c, upper):
    return ptd_triangular.pdf(x, lower, c, upper)


@pytensor_jit
def ptd_cdf(x, lower, c, upper):
    return ptd_triangular.cdf(x, lower, c, upper)


@pytensor_jit
def ptd_ppf(q, lower, c, upper):
    return ptd_triangular.ppf(q, lower, c, upper)


@pytensor_jit
def ptd_logpdf(x, lower, c, upper):
    return ptd_triangular.logpdf(x, lower, c, upper)


@pytensor_jit
def ptd_entropy(lower, c, upper):
    return ptd_triangular.entropy(lower, c, upper)


@pytensor_jit
def ptd_mean(lower, c, upper):
    return ptd_triangular.mean(lower, c, upper)


@pytensor_jit
def ptd_mode(lower, c, upper):
    return ptd_triangular.mode(lower, c, upper)


@pytensor_jit
def ptd_median(lower, c, upper):
    return ptd_triangular.median(lower, c, upper)


@pytensor_jit
def ptd_var(lower, c, upper):
    return ptd_triangular.var(lower, c, upper)


@pytensor_jit
def ptd_std(lower, c, upper):
    return ptd_triangular.std(lower, c, upper)


@pytensor_jit
def ptd_skewness(lower, c, upper):
    return ptd_triangular.skewness(lower, c, upper)


@pytensor_jit
def ptd_kurtosis(lower, c, upper):
    return ptd_triangular.kurtosis(lower, c, upper)


@pytensor_rng_jit
def ptd_rvs(lower, c, upper, size, rng):
    return ptd_triangular.rvs(lower, c, upper, size=size, random_state=rng)
