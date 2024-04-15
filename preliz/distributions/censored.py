# pylint: disable=arguments-differ
import numpy as np

from preliz.distributions.distributions import DistributionTransformer
from preliz.internal.distribution_helper import all_not_none
from preliz.internal.special import xlogx, xprody
from preliz.distributions.truncated import Truncated


class Censored(DistributionTransformer):
    r"""
    Censored distribution

    This is not a distribution per se, but a modifier of univariate distributions.

    Given a base distribution with cumulative distribution function (CDF) and
    probability density mass/function (PDF). The pdf of a Censored distribution is:

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower) & \text{for } x = lower, \\
            \text{PDF}(x) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Normal, Censored
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
        self.lower, self.upper = self.support
        self.params_support = (*self.dist.params_support, self.dist.support, self.dist.support)

    def mean(self):
        if self.kind == "discrete":
            p_low = self.dist.cdf(self.lower - 1)
        else:
            p_low = self.dist.cdf(self.lower)
        p_up = 1 - self.dist.cdf(self.upper)
        p_int = 1 - (p_low + p_up)
        mean_trunc = Truncated(self.dist, self.lower, self.upper).mean()
        return xprody(self.lower, p_low) + mean_trunc * p_int + xprody(self.upper, p_up)

    def median(self):
        return np.clip(self.dist.median(), self.lower, self.upper)

    def var(self):
        mean = self.mean()
        if self.kind == "discrete":
            p_low = self.dist.cdf(self.lower - 1)
        else:
            p_low = self.dist.cdf(self.lower)
        p_up = 1 - self.dist.cdf(self.upper)
        p_int = 1 - (p_low + p_up)
        trunc = Truncated(self.dist, self.lower, self.upper)
        mean_trunc = trunc.mean()
        var_trunc = trunc.var() + (mean_trunc - mean) ** 2
        return (
            xprody((self.lower - mean) ** 2, p_low)
            + var_trunc * p_int
            + xprody((self.upper - mean) ** 2, p_up)
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mean = self.mean()
        std = self.std()
        if self.kind == "discrete":
            p_low = self.dist.cdf(self.lower - 1)
        else:
            p_low = self.dist.cdf(self.lower)
        p_up = 1 - self.dist.cdf(self.upper)
        p_int = 1 - (p_low + p_up)
        trunc = Truncated(self.dist, self.lower, self.upper)
        mean_trunc = trunc.mean()
        std_trunc = trunc.std()

        skew_trunc = trunc.skewness() + ((mean_trunc - mean) / std_trunc) ** 3
        return (
            xprody(((self.lower - mean) / std) ** 3, p_low)
            + skew_trunc * p_int
            + xprody(((self.upper - mean) / std) ** 3, p_up)
        )

    def kurtosis(self):
        mean = self.mean()
        std = self.std()
        if self.kind == "discrete":
            p_low = self.dist.cdf(self.lower - 1)
        else:
            p_low = self.dist.cdf(self.lower)
        p_up = 1 - self.dist.cdf(self.upper)
        p_int = 1 - (p_low + p_up)
        trunc = Truncated(self.dist, self.lower, self.upper)
        mean_trunc = trunc.mean()
        std_trunc = trunc.std()

        kurt_trunc = trunc.kurtosis() + 3 + ((mean_trunc - mean) / std_trunc) ** 4
        return (
            xprody(((self.lower - mean) / std) ** 4, p_low)
            + kurt_trunc * p_int
            + xprody(((self.upper - mean) / std) ** 4, p_up)
        ) - 3

    def rvs(self, size=None, random_state=None):
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
        with np.errstate(divide="ignore"):
            vals = np.where(x == self.lower, np.log(self.dist.cdf(self.lower)), vals)
            if self.kind == "discrete":
                vals = np.where(x == self.upper, np.log1p(-self.dist.cdf(self.upper - 1)), vals)
            else:
                vals = np.where(x == self.upper, np.log1p(-self.dist.cdf(self.upper)), vals)

        return vals

    def entropy(self):
        p_low_inc = self.dist.cdf(self.lower)
        p_up = 1 - self.dist.cdf(self.upper)
        if self.kind == "discrete":
            p_l = np.nan_to_num(self.dist.pdf(self.lower))
            p_u = np.nan_to_num(self.dist.pdf(self.upper))
            p_low = p_low_inc - p_l
            p_up_inc = p_up + p_u
            xlogx_pl = xlogx(p_low)
            xlogx_pu = xlogx(p_up)
        else:
            p_low = p_low_inc
            p_up_inc = p_up
            xlogx_pl = xlogx_pu = 0

        p_int = 1 - (p_low + p_up)
        entropy_bound = -(xlogx(p_low_inc) + xlogx(p_up_inc))
        trunc_ent = Truncated(self.dist, self.lower, self.upper).entropy()
        entropy_interval = trunc_ent * p_int - xlogx(p_int) + xlogx_pl + xlogx_pu
        return entropy_interval + entropy_bound

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def _fit_moments(self, mean, sigma):
        self.dist._fit_moments(mean, sigma)
        self._parametrization(**dict(zip(self.dist.param_names, self.dist.params)))
