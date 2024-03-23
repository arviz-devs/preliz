# pylint: disable=arguments-differ
import numpy as np

from preliz.distributions.distributions import TruncatedCensored
from preliz.internal.distribution_helper import all_not_none


class Censored(TruncatedCensored):
    r"""
    Truncated distribution

    This is not a distribution per se, but a modifier of univariate distributions.

    The pdf of a Truncated distribution is

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower, dist) & \text{for } x = lower, \\
            \text{PDF}(x, dist) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper, dist) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Gamma, Censored
        az.style.use('arviz-doc')
        Censored(Normal(0, 1), -1, 1).plot_pdf(support=(-4, 4))
        Normal(0, 1).plot_pdf(alpha=0.5)
        

    Parameters
    ----------
    dist: PreliZ distribution
        Univariate PreliZ distribution which will be censored.
    lower: float or int
        Lower (left) censoring point. Use np.inf for no censoring.
    upper: float or int
        Upper (right) censoring point. Use np.inf for no censoring.

    Note
    ----

    Some methods like mean or variance are not available censored distributions.
    Functions like maxent or quantile are experimental when applied to censored
    distributions and may not work as expected.
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

    def median(self):
        return np.clip(self.dist.median(), self.lower, self.upper)

    def rvs(self, size=1, random_state=None):
        return np.clip(self.dist.rvs(size, random_state), self.lower, self.upper)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        x = np.asarray(x)
        vals = self.dist.cdf(x)
        return np.where(x < self.lower, 0, np.where(x > self.upper, 1, vals))

    def ppf(self, q):
        return np.clip(self.dist.ppf(q), self.lower, self.upper)

    def logpdf(self, x):
        x = np.asarray(x)
        vals = self.dist.logpdf(x)
        vals = np.where((x < self.lower) | (x > self.upper), -np.inf, vals)
        vals = np.where(x == self.lower, np.log(self.dist.cdf(self.lower)), vals)
        if self.kind == "discrete":
            vals = np.where(x == self.upper, np.log1p(-self.dist.cdf(self.upper - 1)), vals)
        else:
            vals = np.where(x == self.upper, np.log1p(-self.dist.cdf(self.upper)), vals)

        return vals

    def entropy(self):
        """
        This is the entropy of the UNcensored distribution
        """
        if self.dist.rv_frozen is None:
            return self.dist.entropy()
        else:
            return self.dist.rv_frozen.entropy()

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def _fit_moments(self, mean, sigma):
        self.dist._fit_moments(mean, sigma)
        self._parametrization(**dict(zip(self.dist.param_names, self.dist.params)))
