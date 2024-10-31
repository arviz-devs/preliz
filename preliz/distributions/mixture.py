# pylint: disable=arguments-differ
import numpy as np

from preliz.distributions.distributions import DistributionTransformer
from preliz.internal.distribution_helper import all_not_none, num_skewness, num_kurtosis
from preliz.internal.optimization import find_ppf


class Mixture(DistributionTransformer):
    r"""
    Mixture distribution

    This is not a distribution per se, but a modifier of univariate distributions.

    Given a series of base distributions with probability density mass/function ($p_i$).
    The pdf/pmf of a mixture distribution is:

    .. math::

        f(x) = \sum_{i=1}^n \, w_i \, p_i(x)

    .. plot::
        :context: close-figs

        from preliz import Normal, Mixture, style
        style.use('preliz-doc')
        Mixture([Normal(0, 0.5), Normal(2, 0.5)], [0.2, 0.8]).plot_pdf()

    Parameters
    ----------
    dists: List of Univariate PreliZ distributions
        Components of the mixture. They should be all discrete or all continuous.
    weights: list of floats
        Weights must be larger or equal to 0 and their sum must be positive.
        If the weights do not sum up to 1, they will be normalized.
    """

    def __init__(self, dists, weights=None):
        self.dist = dists
        self.weights = None
        super().__init__()
        if all(dist.kind == "discrete" for dist in self.dist):
            self.kind = "discrete"
        elif all(dist.kind == "continuous" for dist in self.dist):
            self.kind = "continuous"
        else:
            raise ValueError("mixture of discrete and continuous distributions are not supported")

        self._parametrization(weights)

    def _parametrization(self, weights=None):
        self.params = []
        self.param_names = []
        self.params_support = []
        for dist in self.dist:
            if not hasattr(dist, "params"):
                self.params.extend([None] * len(dist.param_names))
            else:
                self.params.extend(dist.params)
            self.param_names.extend(dist.param_names)
            self.params_support.append(dist.params_support)

        self.params.append(weights)
        self.param_names.append("weights")
        self.params_support.append((0, 1))

        self.support = np.min([dist.support[0] for dist in self.dist]), np.max(
            [dist.support[1] for dist in self.dist]
        )
        self.weights = np.asarray(weights)
        self.weights = self.weights / np.sum(self.weights)

        if all_not_none(*self.params):
            self.is_frozen = True

    def pdf(self, x):
        return np.sum(
            [dist.pdf(x) * weight for dist, weight in zip(self.dist, self.weights)], axis=0
        )

    def cdf(self, x):
        return np.sum(
            [dist.cdf(x) * weight for dist, weight in zip(self.dist, self.weights)], axis=0
        )

    def ppf(self, q):
        return find_ppf(self, q)

    def logpdf(self, x):
        return np.sum(
            [dist.logpdf(x) * weight for dist, weight in zip(self.dist, self.weights)], axis=0
        )

    def _neg_logpdf(self, x):
        return -self.logpdf(x).sum()

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        if self.kind == "discrete":
            return -np.sum(np.exp(logpdf) * logpdf)
        else:
            return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return np.sum(
            [dist.mean() * weight for dist, weight in zip(self.dist, self.weights)], axis=0
        )

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return (
            np.sum(
                [
                    weight * (dist.var() + (dist.mean() ** 2))
                    for dist, weight in zip(self.dist, self.weights)
                ],
                axis=0,
            )
            - self.mean() ** 2
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return num_skewness(self)

    def kurtosis(self):
        return num_kurtosis(self)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        dist_idx = random_state.choice(len(self.dist), size=size, p=self.weights)
        samples = []
        for idx, dist in enumerate(self.dist):
            n_samples = np.sum(dist_idx == idx)
            if n_samples:
                samples.append(dist.rvs(n_samples, random_state=random_state))
        return np.concatenate(samples)

    def _fit_moments(self, mean, sigma):
        for dist in self.dist:
            dist._fit_moments(mean, sigma)
        self.is_frozen = True
        self.weights = np.ones(len(self.dist)) / len(self.dist)
