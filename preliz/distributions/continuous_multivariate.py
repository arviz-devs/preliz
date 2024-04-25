# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=attribute-defined-outside-init
"""
Continuous multivariate probability distributions.
"""
from copy import copy

import numpy as np

try:
    from ipywidgets import interactive, widgets
except ImportError:
    pass
from scipy import stats

from .beta import Beta  # pylint: disable=no-name-in-module
from .normal import Normal  # pylint: disable=no-name-in-module
from .distributions_multivariate import Continuous
from ..internal.distribution_helper import all_not_none
from ..internal.plot_helper_multivariate import plot_dirichlet, plot_mvnormal
from ..internal.plot_helper import check_inside_notebook, get_slider

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
        self.param_names = ("alpha",)
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
        Plot the pdf of the marginals or the joint pdf of the simplex.
        The joint representation is only available for a dirichlet with an alpha of length 3.

        Parameters
        ----------
        marginals : True
            Defaults to True, plot the marginal distributions, if False plot the joint distribution
            (only valid for an alpha of length 3).
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantile ranges.
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
            plot the median and two interquantile ranges.
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
            plot the median and two interquantile ranges.
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

    def plot_interactive(
        self,
        kind="pdf",
        xy_lim="both",
        pointinterval=True,
        interval="hdi",
        levels=None,
        figsize=None,
    ):
        """
        Interactive exploration of parameters

        Parameters
        ----------
        kind : str:
            Type of plot. Available options are `pdf`, `cdf` and `ppf`.
        xy_lim : str or tuple
            Set the limits of the x-axis and/or y-axis.
            Defaults to `"both"`, the limits of both axes are fixed for all subplots.
            Use `"auto"` for automatic rescaling of x-axis and y-axis.
            Or set them manually by passing a tuple of 4 elements,
            the first two for x-axis, the last two for y-axis. The tuple can have `None`.
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False.
            If `True` the default is to plot the median and two inter-quantiles ranges.
        interval : str
            Type of interval. Available options are the highest density interval `"hdi"` (default),
            equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        figsize : tuple
            Size of the figure
        """

        check_inside_notebook()

        args = dict(zip(self.param_names, self.params))
        self.__init__(**args)  # pylint: disable=unnecessary-dunder-call
        if kind == "pdf":
            w_checkbox_marginals = widgets.Checkbox(
                value=True,
                description="marginals",
                disabled=False,
                indent=False,
            )
            plot_widgets = {"marginals": w_checkbox_marginals}
        else:
            plot_widgets = {}
        for index, dim in enumerate(self.params[0]):
            plot_widgets[f"alpha-{index + 1}"] = get_slider(
                f"alpha-{index + 1}", dim, *self.params_support[0]
            )

        def plot(**args):
            if kind == "pdf":
                marginals = args.pop("marginals")
            params = {"alpha": np.asarray(list(args.values()), dtype=float)}
            self.__init__(**params)  # pylint: disable=unnecessary-dunder-call
            if kind == "pdf":
                plot_dirichlet(
                    self,
                    "pdf",
                    marginals,
                    pointinterval,
                    interval,
                    levels,
                    "full",
                    figsize,
                    None,
                    xy_lim,
                )
            elif kind == "cdf":
                plot_dirichlet(
                    self,
                    "cdf",
                    "marginals",
                    pointinterval,
                    interval,
                    levels,
                    "full",
                    figsize,
                    None,
                    xy_lim,
                )
            elif kind == "ppf":
                plot_dirichlet(
                    self,
                    "cdf",
                    "marginals",
                    pointinterval,
                    interval,
                    levels,
                    None,
                    figsize,
                    None,
                    xy_lim,
                )

        return interactive(plot, **plot_widgets)


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
            MvNormal(mu, sigma).plot_pdf(marginals=False, ax=ax)

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    MvNormal distribution has 2 alternative parameterizations. In terms of the mean and
    the covariance matrix, or in terms of the mean and the precision matrix.

    The link between the 2 alternatives is given by

    .. math::

        \Tau = \Sigma^{-1}

    Parameters
    ----------
    mu : array of floats
        Vector of means.
    cov : array of floats, optional
        Covariance matrix.
    tau : array of floats, optional
        Precision matrix.
    """

    def __init__(self, mu=None, cov=None, tau=None):
        super().__init__()
        self.dist = copy(stats.multivariate_normal)
        self.marginal = Normal
        self.support = (-np.inf, np.inf)
        self._parametrization(mu, cov, tau)

    def _parametrization(self, mu=None, cov=None, tau=None):
        if all_not_none(cov, tau):
            raise ValueError("Incompatible parametrization. Either use mu and cov, or mu and tau.")

        names = ("mu", "cov")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))

        if tau is not None:
            self.tau = tau
            cov = np.linalg.inv(tau)
            names = ("mu", "tau")

        self.mu = mu
        self.cov = cov
        self.param_names = names
        if mu is not None and cov is not None:
            self._update(mu, cov)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self):
            frozen = self.dist(mean=self.mu, cov=self.cov, allow_singular=True)
        return frozen

    def _update(self, mu, cov):
        self.mu = np.array(mu, dtype=float)
        self.cov = np.array(cov, dtype=float)
        self.tau = np.linalg.inv(cov)

        if self.param_names[1] == "cov":
            self.params = (self.mu, self.cov)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

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
        Plot the pdf of the marginals or the joint pdf
        The joint representation is only available for a 2D Multivariate Normal.

        Parameters
        ----------
        marginals : True
            Defaults to True, plot the marginal distributions, if False plot the joint distribution
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

    def plot_interactive(
        self,
        kind="pdf",
        xy_lim="both",
        pointinterval=True,
        interval="hdi",
        levels=None,
        figsize=None,
    ):
        """
        Interactive exploration of parameters

        Parameters
        ----------
        kind : str:
            Type of plot. Available options are `pdf`, `cdf` and `ppf`.
        xy_lim : str or tuple
            Set the limits of the x-axis and/or y-axis.
            Defaults to `"both"`, the limits of both axes are fixed for all subplots.
            Use `"auto"` for automatic rescaling of x-axis and y-axis.
            Or set them manually by passing a tuple of 4 elements,
            the first two for x-axis, the last two for y-axis. The tuple can have `None`.
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False.
            If `True` the default is to plot the median and two inter-quantiles ranges.
        interval : str
            Type of interval. Available options are the highest density interval `"hdi"` (default),
            equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        levels : list
            Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
            For quantiles the number of elements should be 5, 3, 1 or 0
            (in this last case nothing will be plotted).
        figsize : tuple
            Size of the figure
        """

        check_inside_notebook()

        args = dict(zip(self.param_names, self.params))
        cov, tau = args.get("cov", None), args.get("tau", None)
        self.__init__(**args)  # pylint: disable=unnecessary-dunder-call
        if kind == "pdf":
            w_checkbox_marginals = widgets.Checkbox(
                value=True,
                description="marginals",
                disabled=False,
                indent=False,
            )
            plot_widgets = {"marginals": w_checkbox_marginals}
        else:
            plot_widgets = {}
        for index, mu in enumerate(self.params[0]):
            plot_widgets[f"mu-{index + 1}"] = get_slider(
                f"mu-{index + 1}", mu, *self.params_support[0]
            )

        def plot(**args):
            if kind == "pdf":
                marginals = args.pop("marginals")
            params = {"mu": np.asarray(list(args.values()), dtype=float), "cov": cov, "tau": tau}
            self.__init__(**params)  # pylint: disable=unnecessary-dunder-call
            if kind == "pdf":
                plot_mvnormal(
                    self,
                    "pdf",
                    marginals,
                    pointinterval,
                    interval,
                    levels,
                    "full",
                    figsize,
                    None,
                    xy_lim,
                )
            elif kind == "cdf":
                plot_mvnormal(
                    self,
                    "cdf",
                    "marginals",
                    pointinterval,
                    interval,
                    levels,
                    "full",
                    figsize,
                    None,
                    xy_lim,
                )
            elif kind == "ppf":
                plot_mvnormal(
                    self,
                    "cdf",
                    "marginals",
                    pointinterval,
                    interval,
                    levels,
                    None,
                    figsize,
                    None,
                    xy_lim,
                )

        return interactive(plot, **plot_widgets)
