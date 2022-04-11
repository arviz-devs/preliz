"""
Discrete probability distributions.
"""
# pylint: disable=useless-super-delegation
from scipy import stats

from .distributions import Discrete


class Poisson(Discrete):
    R"""
    Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Poisson
        az.style.use('arviz-white')
        for mu in [0.5, 3, 8]:
            Poisson(mu).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """

    def __init__(self, mu=None):
        super().__init__()
        self.mu = mu
        self.name = "poisson"
        self.params = (self.mu,)
        self.param_names = ("mu",)
        self.dist = stats.poisson
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.mu)

    def _update(self, mu):
        self.mu = mu
        self.params = (self.mu,)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        self._update(mean)

    def fit_mle(self, sample, **kwargs):
        # This is not available from scipy. We will use our own implementation
        raise NotImplementedError

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)
