from functools import reduce
from operator import mul

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from scipy.special import gamma
from .plot_helper import repr_to_matplotlib


def get_cols_rows(n_plots):
    """Get number of columns and rows for a grid of plots."""
    if n_plots <= 3:
        return 1, n_plots
    else:
        rows = int(np.ceil(n_plots / 3))
        return 3, rows


class DirichletOnSimplex:
    def __init__(self, alpha):
        """Creates Dirichlet distribution with parameter `alpha`.

        Adapted from Thomas Boggs' blogpost
        https://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
        """

        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])

        self._corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.75**0.5]])
        self._triangle = tri.Triangulation(self._corners[:, 0], self._corners[:, 1])
        self._midpoints = [
            (self._corners[(i + 1) % 3] + self._corners[(i + 2) % 3]) / 2.0 for i in range(3)
        ]

    def xy2bc(self, x_y, tol=1.0e-3):
        """
        Converts 2D Cartesian coordinates to barycentric coordinates.

        Parameters
        ----------
        xy : A length-2 sequence containing the x and y value.
        """
        bsc = [
            (self._corners[i] - self._midpoints[i]).dot(x_y - self._midpoints[i]) / 0.75
            for i in range(3)
        ]
        return np.clip(bsc, tol, 1.0 - tol)

    def pdf(self, x):
        """Returns pdf value for `x`."""
        return self._coef * reduce(mul, [xx ** (aa - 1) for (xx, aa) in zip(x, self._alpha)])

    def plot(self, ax=None):
        """
        Draws pdf contours over the 2-simplex.

        Parameters
        ----------
        dist : A distribution instance with a `pdf` method.
        nlevels: int
            Number of contours to draw.
        subdiv: int
            Number of recursive mesh subdivisions to create.
        """
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=8)
        pvals = [self.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

        hdi_probs = [0.1, 0.5, 0.94]
        contour_levels = find_hdi_contours(pvals, hdi_probs)
        if all(contour_levels == contour_levels[0]):
            ax.tricontourf(trimesh, pvals)
        else:
            ax.tricontour(
                trimesh,
                pvals,
                levels=contour_levels,
            )
            ax.triplot(self._triangle, color="0.8", linestyle="--", linewidth=2)
        ax.axis("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)
        ax.axis("off")


def plot_dirichlet(
    dist,
    representation,
    marginals,
    pointinterval,
    interval,
    levels,
    support,
    figsize,
    axes,
    xy_lim="auto",
):
    """Plot pdf, cdf or ppf of Dirichlet distribution."""

    alpha = dist.alpha
    dim = len(alpha)

    if figsize is None:
        figsize = (12, 4)

    if isinstance(xy_lim, tuple):
        xlim = xy_lim[:2]
        ylim = xy_lim[2:]

    if marginals:
        a_0 = alpha.sum()
        cols, rows = get_cols_rows(dim)

        if axes is None:
            fig, axes = plt.subplots(cols, rows, figsize=figsize, sharex=True, sharey=True)
            axes = axes.flatten()
        if len(axes) > dim:
            for ax in axes[dim:]:
                ax.remove()

        for a_i, ax in zip(alpha, axes):
            marginal_dist = dist.marginal(a_i, a_0 - a_i)
            if xy_lim == "both":
                xlim = marginal_dist._finite_endpoints("full")
                xvals = marginal_dist.xvals("restricted")
                if representation == "pdf":
                    max_pdf = np.max(marginal_dist.pdf(xvals))
                    ylim = (-max_pdf * 0.075, max_pdf * 1.5)
                elif representation == "ppf":
                    max_ppf = marginal_dist.ppf(0.999)
                    ylim = (-max_ppf * 0.075, max_ppf * 1.5)
            if representation == "pdf":
                marginal_dist.plot_pdf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    support=support,
                    legend=False,
                    ax=ax,
                )
            elif representation == "cdf":
                marginal_dist.plot_cdf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    support=support,
                    legend=False,
                    ax=ax,
                )
            elif representation == "ppf":
                marginal_dist.plot_ppf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    legend=False,
                    ax=ax,
                )
            if xy_lim != "auto" and representation != "ppf":
                ax.set_xlim(*xlim)
            if xy_lim != "auto" and representation != "cdf":
                ax.set_ylim(*ylim)
        fig.text(0.5, 1, repr_to_matplotlib(dist), ha="center", va="center")

    else:
        if dim == 3:
            if axes is None:
                _, axes = plt.subplots(1, 1)
            DirichletOnSimplex(alpha).plot(ax=axes)
            axes.set_title(repr_to_matplotlib(dist))
        else:
            raise ValueError("joint only works for Dirichlet of dim=3")


def joint_normal(dist, ax):
    """Plot pdf of joint normal distribution."""
    x_max = dist.mu[0] + 3 * dist.cov[0, 0] ** 0.5
    y_max = dist.mu[1] + 3 * dist.cov[1, 1] ** 0.5
    x_min = dist.mu[0] - 3 * dist.cov[0, 0] ** 0.5
    y_min = dist.mu[1] - 3 * dist.cov[1, 1] ** 0.5
    x, y = np.mgrid[x_min:x_max:0.01, y_min:y_max:0.01]

    pos = np.dstack((x, y))

    density = dist.pdf(pos)
    contour_levels = find_hdi_contours(density, [0.1, 0.5, 0.94])
    ax.contour(x, y, density, levels=contour_levels)
    ax.set_aspect("equal")


def find_hdi_contours(density, hdi_probs):
    """
    Find contours enclosing regions of highest posterior density.

    Parameters
    ----------
    density : array-like
    hdi_probs : array-like
        An array of highest density interval confidence probabilities.

    Returns
    -------
    contour_levels : list
        The contour levels corresponding to the given HDI probabilities.
    """
    # Using the algorithm from corner.py
    sorted_density = np.sort(density, axis=None)[::-1]
    csd = sorted_density.cumsum()
    csd /= csd[-1]

    contours = np.empty_like(hdi_probs)
    for idx, hdi_prob in enumerate(hdi_probs):
        try:
            contours[idx] = sorted_density[csd <= hdi_prob][-1]
        except IndexError:
            contours[idx] = sorted_density[0]

    contours.sort()

    return contours


def plot_mvnormal(
    dist,
    representation,
    marginals,
    pointinterval,
    interval,
    levels,
    support,
    figsize,
    axes,
    xy_lim="auto",
):
    """Plot pdf, cdf or ppf of Multivariate Normal distribution."""

    mu = dist.mu
    sigma = dist.rv_frozen.var() ** 0.5
    dim = len(mu)

    if figsize is None:
        figsize = (12, 4)

    if isinstance(xy_lim, tuple):
        xlim = xy_lim[:2]
        ylim = xy_lim[2:]

    if marginals:
        cols, rows = get_cols_rows(dim)

        if axes is None:
            fig, axes = plt.subplots(cols, rows, figsize=figsize, sharex=True, sharey=True)
            axes = axes.flatten()
        if len(axes) > dim:
            for ax in axes[dim:]:
                ax.remove()

        for mu_i, sigma_i, ax in zip(mu, sigma, axes):
            marginal_dist = dist.marginal(mu_i, sigma_i)
            if xy_lim == "both":
                xlim = marginal_dist._finite_endpoints("full")
                xvals = marginal_dist.xvals("restricted")
                if representation == "pdf":
                    max_pdf = np.max(marginal_dist.pdf(xvals))
                    ylim = (-max_pdf * 0.075, max_pdf * 1.5)
                elif representation == "ppf":
                    max_ppf = marginal_dist.ppf(0.999)
                    ylim = (-max_ppf * 0.075, max_ppf * 1.5)
            if representation == "pdf":
                marginal_dist.plot_pdf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    support=support,
                    legend=False,
                    ax=ax,
                )
            elif representation == "cdf":
                marginal_dist.plot_cdf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    support=support,
                    legend=False,
                    ax=ax,
                )
            elif representation == "ppf":
                marginal_dist.plot_ppf(
                    pointinterval=pointinterval,
                    interval=interval,
                    levels=levels,
                    legend=False,
                    ax=ax,
                )
            if xy_lim != "auto" and representation != "ppf":
                ax.set_xlim(*xlim)
            if xy_lim != "auto" and representation != "cdf":
                ax.set_ylim(*ylim)
        fig.text(0.5, 1, repr_to_matplotlib(dist), ha="center", va="center")

    else:
        if dim == 2:
            if axes is None:
                _, axes = plt.subplots(1, 1)
            joint_normal(dist, axes)
            axes.set_title(repr_to_matplotlib(dist))
        else:
            raise ValueError("joint only works for Multivariate Normal of dim=2")
