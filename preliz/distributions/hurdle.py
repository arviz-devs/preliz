import numpy as np

from preliz.distributions.distributions import DistributionTransformer
from preliz.internal.distribution_helper import all_not_none, eps


class Hurdle(DistributionTransformer):
    r"""
    Hurdle distribution.

    This is not a distribution per se, but a modifier of univariate distributions.

    Given a base distribution with parameters $\theta$, cumulative distribution function (CDF),
    and probability density mass/function (PDF). The density of a Hurdle distribution is:

    .. math::

        f(x \mid \psi, \mu) =
            \left\{
                \begin{array}{l}
                (1 - \psi)  \ \text{if } x = 0 \\
                \psi
                \frac{\text{PDF}(x \mid \theta))}
                {1 - \text{CDF}(\epsilon \mid \theta)} \ \text{if } x \neq 0
                \end{array}
            \right.

    where $\psi$ is the expected proportion of the base distribution (0 < $\psi$ < 1), and
    $\epsilon$ is the machine precision for continuous distributions and 0 for discrete ones.

    The following figure shows the difference between a Gamma distribution and a HurdleGamma, with
    the same parameters for the base distribution (Gamma).

    .. plot::
        :context: close-figs

        from preliz import Gamma, Hurdle, style
        style.use('preliz-doc')
        Hurdle(Gamma(mu=2, sigma=1), 0.8).plot_pdf()
        Gamma(mu=2, sigma=1).plot_pdf()

    Parameters
    ----------
    dist: PreliZ distribution
        Univariate PreliZ distribution which will be truncated.
    psi : float
        Expected proportion of the base distribution (0 < psi < 1)
    """

    def __init__(self, dist, psi=None, **kwargs):
        self.dist = dist
        super().__init__()
        self._parametrization(psi, **kwargs)

    def _parametrization(self, psi=None, **kwargs):
        dist_params = []
        if not kwargs:
            if hasattr(self.dist, "params"):
                kwargs = dict(zip(self.dist.param_names, self.dist.params))
            else:
                kwargs = dict(zip(self.dist.param_names, [None] * len(self.dist.param_names)))

        for key, value in kwargs.items():
            dist_params.append(value)
            setattr(self, key, value)

        self.psi = psi
        self.params = (*dist_params, self.psi)
        self.param_names = (*self.dist.param_names, "psi")
        if all_not_none(*dist_params):
            self.dist._parametrization(**kwargs)
            self.is_frozen = True

        self.support = self.dist.support
        self.params_support = (*self.dist.params_support, (0, 1))

    def mean(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum(x_values * pdf)
        else:
            return np.trapezoid(x_values * pdf, x_values)

    def mode(self):
        if self.kind == "discrete":
            from preliz.internal.optimization import find_discrete_mode

            return find_discrete_mode(self)
        else:
            from preliz.internal.optimization import find_mode

            return find_mode(self)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        if self.kind == "discrete":
            return np.sum((x_values - self.mean()) ** 2 * pdf)
        else:
            return np.trapezoid((x_values - self.mean()) ** 2 * pdf, x_values)

    def std(self):
        return self.var() ** 0.5

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return self.ppf(random_state.uniform(size=size))

    def pdf(self, x):
        if self.dist == "discrete":
            return np.where(
                x == 0, 1 - self.psi, self.psi * self.dist.pdf(x) / (1 - self.dist.cdf(0))
            )
        else:
            return np.where(
                x == 0, 1 - self.psi, self.psi * self.dist.pdf(x) / (1 - self.dist.cdf(eps))
            )

    def cdf(self, x):
        if self.dist == "discrete":
            return np.where(
                x <= 0, 1 - self.psi, 1 - self.psi * (1 - self.dist.cdf(x)) / (1 - self.dist.cdf(0))
            )
        else:
            return np.where(
                x <= 0,
                1 - self.psi,
                1 - self.psi * (1 - self.dist.cdf(x)) / (1 - self.dist.cdf(eps)),
            )

    def ppf(self, q):
        if self.kind == "discrete":
            lower = 0
        else:
            lower = eps
        q = np.asarray(q)
        psi = self.psi
        lcdf = self.dist.cdf(lower)
        vals = np.zeros_like(q)
        vals[q >= (1 - psi)] = self.dist.ppf(
            lcdf + (q[q >= (1 - psi)] - (1 - psi)) * (1 - lcdf) / psi
        )
        return np.where((q < 0) | (q > 1), np.nan, vals)

    def logpdf(self, x):
        if self.dist == "discrete":
            pdf_values = np.where(
                x == 0,
                np.log(1 - self.psi),
                np.log(self.psi) + self.dist.logpdf(x) - np.log(1 - self.dist.cdf(0)),
            )
        else:
            pdf_values = np.where(
                x == 0,
                np.log(1 - self.psi),
                np.log(self.psi) + self.dist.logpdf(x) - np.log(1 - self.dist.cdf(eps)),
            )
        return pdf_values

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        if self.kind == "discrete":
            return -np.sum(np.exp(logpdf) * logpdf)
        else:
            return -np.trapezoid(np.exp(logpdf) * logpdf, x_values)

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def _fit_moments(self, mean, sigma):
        self.dist._fit_moments(mean, sigma)
        self._parametrization(**dict(zip(self.dist.param_names, self.dist.params)))
