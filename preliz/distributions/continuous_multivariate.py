# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init
"""
Continuous multivariate probability distributions.
"""
from copy import copy

import numpy as np
from scipy import stats

from .distributions_multivariate import Continuous
from .continuous import Beta, Normal
from ..internal.distribution_helper import all_not_none
from ..internal.plot_helper_multivariate import plot_dirichlet, plot_mvnormal


eps = np.finfo(float).eps


class Dirichlet(Continuous):
    r"""
    Dirichlet distribution.

    .. math::

       f(\mathbf{x}|\mathbf{a}) =
           \frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}
           \prod_{i=1}^k x_i^{a_i - 1}

    .. plot::
        :context: close-figs

        import arviz as az
        import matplotlib.pyplot as plt
        from preliz import Dirichlet
        _, axes = plt.subplots(2, 2)
        alphas = [[0.5, 0.5, 0.5], [1, 1, 1], [5, 5, 5], [5, 2, 1]]
        for alpha, ax in zip(alphas, axes.ravel()):
            pz.Dirichlet(alpha).plot_pdf(marginals=False, ax=ax)


    ========  ===============================================
    Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum x_i = 1`
    Mean      :math:`\dfrac{a_i}{\sum a_i}`
    Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}`
              where :math:`a_0 = \sum a_i`
    ========  ===============================================

    Parameters
    ----------
    alpha : array of floats
        Concentration parameter (alpha > 0).
    """

    def __init__(self, alpha=None):
        super().__init__()
        self.dist = copy(stats.dirichlet)
        self.marginal = Beta
        self.support = (eps, 1 - eps)
        self._parametrization(alpha)

    def _parametrization(self, alpha=None):
        self.param_names = "alpha"
        self.params_support = ((eps, np.inf),)

        self.alpha = alpha
        if alpha is not None:
            self._update(alpha)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
            frozen = self.dist(self.alpha)
        return frozen

    def _update(self, alpha):
        self.alpha = np.array(alpha, dtype=float)
        self.params = (self.alpha,)
        self._update_rv_frozen()

    def _fit_mle(self, sample, **kwargs):
        raise NotImplementedError

    def plot_pdf(
        self,
        marginals=True,
        pointinterval=False,
        interval="hdi",
        levels=None,
        support="full",
        figsize=None,
        ax=None,
    ):
        """
        Plot the  pdf of the marginals or the joint pdf on the simplex.
        The joint representation is only available for a dirichlet with a alpha of length 3.

        Parameters
        ----------
        marginals : True
            Defaults to True, plor the marginal distributions, if False plot the joint distribution
            (only valid for an alpha of length 3).
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        figsize : tuple
            Size of the
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_dirichlet(
            self, "pdf", marginals, pointinterval, interval, levels, support, figsize, ax
        )

    def plot_cdf(
        self,
        pointinterval=False,
        interval="hdi",
        levels=None,
        support="full",
        figsize=None,
        ax=None,
    ):
        """
        Plot the cumulative distribution function.

        Parameters
        ----------
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        figsize : tuple
            Size of the figure
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_dirichlet(
            self, "cdf", "marginals", pointinterval, interval, levels, support, figsize, ax
        )

    def plot_ppf(
        self,
        pointinterval=False,
        interval="hdi",
        levels=None,
        figsize=None,
        ax=None,
    ):
        """
        Plot the quantile function.

        Parameters
        ----------
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        figsize : tuple
            Size of the figure
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_dirichlet(
            self, "ppf", "marginals", pointinterval, interval, levels, None, figsize, ax
        )


class MvNormal(Continuous):
    r"""
    Multivariate Normal distribution.

    .. math::

       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{k/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        import matplotlib.pyplot as plt
        from preliz import MvNormal
        _, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
        mus = [[0., 0], [3, -2], [0., 0], [0., 0]]
        sigmas = [np.eye(2), np.eye(2), np.array([[2, 2], [2, 4]]), np.array([[2, -2], [-2, 4]])]
        for mu, sigma, ax in zip(mus, sigmas, axes.ravel()):
            pz.MvNormal(mu, sigma).plot_pdf(marginals=False, ax=ax)

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Vector of means.
    cov : tensor_like of float, optional
        Covariance matrix.
    """

    def __init__(self, mu=None, cov=None):
        super().__init__()
        self.dist = copy(stats.multivariate_normal)
        self.marginal = Normal
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, cov)

    def _parametrization(self, mu=None, cov=None):
        self.param_names = ("mu", "cov")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        self.mu = mu
        self.cov = cov
        if mu is not None and cov is not None:
            self._update(mu, cov)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
            frozen = self.dist(mean=self.mu, cov=self.cov)
        return frozen

    def _update(self, mu, cov):
        self.mu = np.array(mu, dtype=float)
        self.cov = np.array(cov, dtype=float)
        self.params = (mu, cov)
        self._update_rv_frozen()
        self.rv_frozen.var = lambda: np.diag(self.cov)

    def _fit_mle(self, sample, **kwargs):
        raise NotImplementedError

    def plot_pdf(
        self,
        marginals=True,
        pointinterval=False,
        interval="hdi",
        levels=None,
        support="full",
        figsize=None,
        ax=None,
    ):
        """
        Plot the  pdf of the marginals or the joint pdf
        The joint representation is only available for a 2D Multivariate Normal.

        Parameters
        ----------
        marginals : True
            Defaults to True, plor the marginal distributions, if False plot the joint distribution
            (only valid for a bivariate normal).
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        figsize : tuple
            Size of the figure
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_mvnormal(
            self, "pdf", marginals, pointinterval, interval, levels, support, figsize, ax
        )

    def plot_cdf(
        self,
        pointinterval=False,
        interval="hdi",
        levels=None,
        support="full",
        figsize=None,
        ax=None,
    ):
        """
        Plot the cumulative distribution function.

        Parameters
        ----------
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        figsize : tuple
            Size of the figure
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_mvnormal(
            self, "cdf", "marginals", pointinterval, interval, levels, support, figsize, ax
        )

    def plot_ppf(
        self,
        pointinterval=False,
        interval="hdi",
        levels=None,
        figsize=None,
        ax=None,
    ):
        """
        Plot the quantile function.

        Parameters
        ----------
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        interval : str
            Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        figsize : tuple
            Size of the figure
        ax : matplotlib axis
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis
        """
        return plot_mvnormal(
            self, "ppf", "marginals", pointinterval, interval, levels, None, figsize, ax
        )
