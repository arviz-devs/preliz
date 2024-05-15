# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
# pylint: disable=invalid-name
import numpy as np
import numba as nb

from ..internal.distribution_helper import all_not_none
from .distributions import Continuous


class Triangular(Continuous):
    r"""
    Triangular distribution

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

        import arviz as az
        from preliz import Triangular
        az.style.use('arviz-doc')
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
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.lower, self.c, self.upper))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.lower, self.c, self.upper)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.lower, self.c, self.upper)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.lower, self.c, self.upper)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.lower, self.c, self.upper)

    def entropy(self):
        return nb_entropy(self.lower, self.upper)

    def mean(self):
        return (self.lower + self.c + self.upper) / 3

    def median(self):
        return np.where(
            self.c >= (self.lower + self.upper) / 2,
            self.lower + ((self.upper - self.lower) * (self.c - self.lower) / 2) ** 0.5,
            self.upper - ((self.upper - self.lower) * (self.upper - self.c) / 2) ** 0.5,
        )

    def var(self):
        return (
            self.lower**2
            + self.upper**2
            + self.c**2
            - self.lower * self.c
            - self.c * self.upper
            - self.lower * self.upper
        ) / 18

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return (
            2**0.5
            * (self.lower + self.upper - 2 * self.c)
            * (2 * self.lower - self.upper - self.c)
            * (self.lower - 2 * self.upper + self.c)
        ) / (
            5
            * (
                self.lower**2
                + self.upper**2
                + self.c**2
                - self.lower * self.c
                - self.c * self.upper
                - self.lower * self.upper
            )
            ** (3 / 2)
        )

    def kurtosis(self):
        return -3 / 5

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(0, 1, size)
        return nb_rvs(random_samples, self.lower, self.c, self.upper)

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


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, lower, c, upper):
    if x <= lower:
        return 0
    elif lower < x <= c:
        return (x - lower) ** 2 / ((upper - lower) * (c - lower))
    elif c < x < upper:
        return 1 - (upper - x) ** 2 / ((upper - lower) * (upper - c))
    return 1


@nb.vectorize(nopython=True, cache=True)
def nb_ppf(q, lower, c, upper):
    if 0 <= q < (c - lower) / (upper - lower):
        return lower + (q * (upper - lower) * (c - lower)) ** 0.5
    elif (c - lower) / (upper - lower) <= q <= 1:
        return upper - ((1 - q) * (upper - lower) * (upper - c)) ** 0.5
    return np.nan


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, lower, c, upper):
    if x < lower:
        return -np.inf
    elif lower <= x < c:
        return np.log(2 * (x - lower) / ((upper - lower) * (c - lower)))
    elif x == c:
        return np.log(2 / (upper - lower))
    elif c < x <= upper:
        return np.log(2 * (upper - x) / ((upper - lower) * (upper - c)))
    return -np.inf


@nb.njit(cache=True)
def nb_neg_logpdf(x, lower, c, upper):
    return -(nb_logpdf(x, lower, c, upper)).sum()


@nb.njit(cache=True)
def nb_entropy(lower, upper):
    return 0.5 + np.log((upper - lower) / 2)


@nb.vectorize(nopython=True, cache=True)
def nb_rvs(random_samples, lower, c, upper):
    if 0 < random_samples < (c - lower) / (upper - lower):
        return lower + (random_samples * (upper - lower) * (c - lower)) ** 0.5
    return upper - ((1 - random_samples) * (upper - lower) * (upper - c)) ** 0.5
