# pylint: disable=arguments-differ
import numpy as np
from preliz.distributions.distributions import TruncatedCensored


class Truncated(TruncatedCensored):
    r"""
    Truncated distribution

    The pdf of a censored distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \frac{\text{PDF}(x, dist)}{\text{CDF}(upper, dist) - \text{CDF}(lower, dist)}
            & \text{for } lower <= x <= upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    Parameters
    ----------
    dist: PreliZ distribution
        Univariate PreliZ distribution which will be truncated.
    lower: float or int 
        Lower (left) truncation point. Use np.inf for no truncation.
    upper: float or int 
        Upper (right) truncation point. Use np.inf for no truncation.
    """

    def __init__(self, dist, lower, upper, **kwargs):
        self.dist = dist
        super().__init__()
        if not kwargs:
            kwargs = dict(zip(self.dist.param_names, self.dist.params))

        self._parametrization(lower, upper, **kwargs)

    def _parametrization(self, lower, upper, **kwargs):
        dist_params = []
        for key, value in kwargs.items():
            dist_params.append(value)
            setattr(self, key, value)

        self.dist._parametrization(**kwargs)
        if self.kind == "discrete":
            self.lower = lower - 1
        else:
            self.lower = lower
        self.upper = upper
        self.params = (*dist_params, self.lower, self.upper)
        self.param_names = (*self.dist.param_names, "lower", "upper")
        self.support = (
            max(self.dist.support[0], self.lower),
            min(self.dist.support[1], self.upper),
        )
        self.params_support = (*self.dist.params_support, self.dist.support, self.dist.support)
        self.is_frozen = True

    def rvs(self, size=1, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.uniform(size=size))

    def pdf(self, x):
        x = np.asarray(x)
        vals = self.dist.pdf(x) / (self.dist.cdf(self.upper) - self.dist.cdf(self.lower))
        return np.where((x < self.lower) | (x > self.upper), 0, vals)

    def cdf(self, x):
        x = np.asarray(x)
        lcdf = self.dist.cdf(self.lower)
        vals = (self.dist.cdf(x) - lcdf) / (self.dist.cdf(self.upper) - lcdf)
        return np.where(x < self.lower, 0, np.where(x > self.upper, 1, vals))

    def ppf(self, q):
        q = np.asarray(q)
        lcdf = self.dist.cdf(self.lower)
        vals = self.dist.ppf(lcdf + q * (self.dist.cdf(self.upper) - lcdf))
        return np.where((q < 0) | (q > 1), np.nan, vals)

    def logpdf(self, x):
        x = np.asarray(x)
        vals = self.dist.logpdf(x) - np.log(self.dist.cdf(self.upper) - self.dist.cdf(self.lower))
        return np.where((x < self.lower) | (x > self.upper), -np.inf, vals)

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def median(self):
        return self.ppf(0.5)
