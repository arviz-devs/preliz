"""
Discrete probability distributions.
"""
# pylint: disable=useless-super-delegation
import numpy as np
from scipy import stats


from .distributions import Discrete


class Binomial(Discrete):
    R"""
    Binomial distribution.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.

    The pmf of this distribution is

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Binomial
        az.style.use('arviz-white')
        ns = [5, 10, 10]
        ps = [0.5, 0.5, 0.7]
        for n, p in zip(ns, ps):
            Binomial(n, p).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n p`
    Variance  :math:`n p (1 - p)`
    ========  ==========================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n=None, p=None):
        super().__init__()
        self.n = n
        self.p = p
        self.name = "binomial"
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.dist = stats.binom
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.n, self.p)

    def _update(self, n, p):
        self.n = n
        self.p = p
        self.params = (self.n, p)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        # check if this is a good
        n_hat = mean**2 / (mean - sigma**2)
        n = int(max(1, n_hat))
        p = mean / n_hat
        self._update(n, p)

    def fit_mle(self, sample):
        # see https://doi.org/10.1016/j.jspi.2004.02.019 for details
        x_bar = np.mean(sample)
        x_std = np.std(sample)
        x_max = np.max(sample)
        n = x_max ** (1.5) * x_std / (x_bar**0.5 * (x_max - x_bar) ** 0.5)
        p = x_bar / n
        self._update(n, p)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


class NegativeBinomial(Discrete):
    R"""
    Negative binomial distribution.

    If it is parametrized in terms of n and p, the negative binomial describes
    the probability to have x failures before the n-th success, given the
    probability p of success in each trial. Its pmf is

    .. math::

        f(x \mid n, p) =
           \binom{x + n - 1}{x}
           (p)^n (1 - p)^x

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import NegativeBinomial
        az.style.use('arviz-white')
        ns = [5, 10, 10]
        ps = [0.5, 0.5, 0.7]
        for n, p in zip(ns, ps):
            NegativeBinomial(n, p).plot_pdf()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\frac{(1-p) r}{p}`
    Variance  :math:`\frac{(1-p) r}{p^2}`
    ========  ==========================

    Parameters
    ----------
    n: float
        Number of target success trials (n > 0)
    p: float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n=None, p=None):
        super().__init__()
        self.n = n
        self.p = p
        self.name = "negativebinomial"
        self.params = (self.n, self.p)
        self.param_names = ("n", "p")
        self.dist = stats.nbinom
        self._update_rv_frozen()

    def _get_frozen(self):
        return self.dist(self.n, self.p)

    def _update(self, n, p):
        self.n = n
        self.p = p
        self.params = (self.n, p)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        n = mean**2 / (sigma**2 - mean)
        p = mean / sigma**2
        self._update(n, p)

    def fit_mle(self, sample):
        raise NotImplementedError

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)


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

    def fit_mle(self, sample):
        mu = np.mean(sample)
        self._update(mu)

    def plot_pdf(
        self, box=False, quantiles=None, support="full", legend="legend", figsize=None, ax=None
    ):
        return super().plot_pdf(box, quantiles, support, legend, figsize, ax)

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        return super().plot_cdf(support, legend, figsize, ax)

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        return super().plot_ppf(legend, figsize, ax)
