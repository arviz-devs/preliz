# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init
# pylint: disable=invalid-unary-operand-type
# pylint: disable=invalid-name
"""
Discrete probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats

from .distributions import Discrete
from .bernoulli import Bernoulli  # pylint: disable=unused-import
from .binomial import Binomial  # pylint: disable=unused-import
from .categorical import Categorical  # pylint: disable=unused-import
from .discrete_uniform import DiscreteUniform  # pylint: disable=unused-import
from .discrete_weibull import DiscreteWeibull  # pylint: disable=unused-import
from .geometric import Geometric  # pylint: disable=unused-import
from .hypergeometric import HyperGeometric  # pylint: disable=unused-import
from .poisson import Poisson  # pylint: disable=unused-import
from .negativebinomial import NegativeBinomial  # pylint: disable=unused-import
from .zi_binomial import ZeroInflatedBinomial  # pylint: disable=unused-import
from .zi_negativebinomial import ZeroInflatedNegativeBinomial  # pylint: disable=unused-import
from .zi_poisson import ZeroInflatedPoisson  # pylint: disable=unused-import

from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.distribution_helper import all_not_none


eps = np.finfo(float).eps


class BetaBinomial(Discrete):
    R"""
    Beta-binomial distribution.

    Equivalent to binomial random variable with success probability
    drawn from a beta distribution.

    The pmf of this distribution is

    .. math::

       f(x \mid \alpha, \beta, n) =
           \binom{n}{x}
           \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import BetaBinomial
        az.style.use('arviz-doc')
        alphas = [0.5, 1, 2.3]
        betas = [0.5, 1, 2]
        n = 10
        for a, b in zip(alphas, betas):
            BetaBinomial(a, b, n).plot_pdf()

    ========  =================================================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n \dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{n \alpha \beta (\alpha+\beta+n)}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
    ========  =================================================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.
    """

    def __init__(self, alpha=None, beta=None, n=None):
        super().__init__()
        self.dist = copy(stats.betabinom)
        self.support = (0, np.inf)
        self._parametrization(alpha, beta, n)

    def _parametrization(self, alpha=None, beta=None, n=None):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.params = (self.alpha, self.beta, self.n)
        self.param_names = ("alpha", "beta", "n")
        self.params_support = ((eps, np.inf), (eps, np.inf), (eps, np.inf))
        if all_not_none(alpha, beta):
            self._update(alpha, beta, n)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(n=self.n, a=self.alpha, b=self.beta)
        return frozen

    def _update(self, alpha, beta, n):
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.n = np.int64(n)
        self.params = (self.alpha, self.beta, self.n)
        self.support = (0, self.n)
        self._update_rv_frozen()

    def _fit_moments(self, mean, sigma):
        # Crude aproximation for n (as in Binomial distribution)
        # For alpha and beta see:
        # https://en.wikipedia.org/wiki/Beta-binomial_distribution#Method_of_moments
        n = mean + sigma * 2
        p = mean / n
        rho = ((sigma**2 / (mean * (1 - p))) - 1) / (n - 1)
        alpha = max(0.5, (p / rho) - p)
        beta = max(0.5, (alpha / p) - alpha)
        params = alpha, beta, n
        optimize_moments(self, mean, sigma, params)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)
