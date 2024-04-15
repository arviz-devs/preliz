# pylint: disable=arguments-differ
import numpy as np

from preliz.distributions.distributions import DistributionTransformer
from preliz.internal.distribution_helper import all_not_none


class Truncated(DistributionTransformer):
    r"""
    Truncated distribution

    This is not a distribution per se, but a modifier of univariate distributions.

    Given a base distribution with cumulative distribution function (CDF) and
    probability density mass/function (PDF). The pdf of a Truncated distribution is:

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \frac{\text{PDF}(x, dist)}{\text{CDF}(upper, dist) - \text{CDF}(lower, dist)}
            & \text{for } lower <= x <= upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gamma, Truncated
        az.style.use('arviz-doc')
        Truncated(Gamma(mu=2, sigma=1), 1, 4.5).plot_pdf()
        Gamma(mu=2, sigma=1).plot_pdf()
        

    Parameters
    ----------
    dist: PreliZ distribution
        Univariate PreliZ distribution which will be truncated.
    lower: float or int
        Lower (left) truncation point. Use np.inf for no truncation.
    upper: float or int
        Upper (right) truncation point. Use np.inf for no truncation.
    """

    def __init__(self, dist, lower=None, upper=None, **kwargs):
        self.dist = dist
        super().__init__()
        self._parametrization(lower, upper, **kwargs)

    def _parametrization(self, lower=None, upper=None, **kwargs):
        dist_params = []
        if not kwargs:
            if hasattr(self.dist, "params"):
                kwargs = dict(zip(self.dist.param_names, self.dist.params))
            else:
                kwargs = dict(zip(self.dist.param_names, [None] * len(self.dist.param_names)))

        for key, value in kwargs.items():
            dist_params.append(value)
            setattr(self, key, value)

        if upper is None:
            self.upper = np.inf
        else:
            self.upper = upper

        if lower is None:
            self.lower = -np.inf
        else:
            self.lower = lower

        self.params = (*dist_params, self.lower, self.upper)
        self.param_names = (*self.dist.param_names, "lower", "upper")
        if all_not_none(*dist_params):
            self.dist._parametrization(**kwargs)
            self.is_frozen = True

        self.support = (
            max(self.dist.support[0], self.lower),
            min(self.dist.support[1], self.upper),
        )
        self.params_support = (*self.dist.params_support, self.dist.support, self.dist.support)

    def mean(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum(x_values * pdf)
        else:
            return np.trapz(x_values * pdf, x_values)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum((x_values - self.mean()) ** 2 * pdf)
        else:
            return np.trapz((x_values - self.mean()) ** 2 * pdf, x_values)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum(((x_values - mean) / std) ** 3 * pdf)
        else:
            return np.trapz(((x_values - mean) / std) ** 3 * pdf, x_values)

    def kurtosis(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum(((x_values - mean) / std) ** 4 * pdf) - 3
        else:
            return np.trapz(((x_values - mean) / std) ** 4 * pdf, x_values) - 3

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.uniform(size=size))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.asarray(x)
        lower = adjust_lower(self.kind, self.lower)
        lcdf = self.dist.cdf(lower)
        vals = (self.dist.cdf(x) - lcdf) / (self.dist.cdf(self.upper) - lcdf)
        return np.where(x < lower, 0, np.where(x > self.upper, 1, vals))

    def ppf(self, q):
        q = np.asarray(q)
        lower = adjust_lower(self.kind, self.lower)
        lcdf = self.dist.cdf(lower)
        vals = self.dist.ppf(lcdf + q * (self.dist.cdf(self.upper) - lcdf))
        return np.where((q < 0) | (q > 1), np.nan, vals)

    def logpdf(self, x):
        x = np.asarray(x)
        lower = adjust_lower(self.kind, self.lower)
        vals = self.dist.logpdf(x) - np.log(self.dist.cdf(self.upper) - self.dist.cdf(lower))
        return np.where((x < self.lower) | (x > self.upper), -np.inf, vals)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        if self.kind == "discrete":
            return -np.sum(np.exp(logpdf) * logpdf)
        else:
            return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def _fit_moments(self, mean, sigma):
        self.dist._fit_moments(mean, sigma)
        self._parametrization(**dict(zip(self.dist.param_names, self.dist.params)))


def adjust_lower(kind, lower):
    if kind == "discrete":
        lower -= 1
    return lower
