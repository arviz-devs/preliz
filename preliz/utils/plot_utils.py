import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

# pylint: disable=protected-access

default_quantiles = [0.05, 0.25, 0.75, 0.95]


def plot_boxlike(fitted_dist, x_vals, ref_pdf, quantiles, ax, color=None):
    """
    Plot the mean as a dot and two inter-quantile ranges as lines
    """
    q_s = fitted_dist.ppf(quantiles)
    mean = fitted_dist.moment(1)

    ax.plot(x_vals, ref_pdf, color=color)
    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")


def plot_boxlike2(sample, ax):
    """
    Plot the mean as a dot and two inter-quantile ranges as lines
    """
    q_s = np.quantile(sample, default_quantiles)
    mean = np.mean(sample)

    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")


def plot_pdfpmf(dist, box, quantiles, support, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    title = repr_to_matplotlib(dist)

    x = dist._xvals(support)
    if dist.kind == "continuous":
        density = dist.rv_frozen.pdf(x)
        ax.plot(x, density, label=title, color=color)
        ax.set_yticks([])
    else:
        density = dist.rv_frozen.pmf(x)
        ax.plot(x, density, "-o", label=title, color=color)
    if box:
        if quantiles is None:
            quantiles = default_quantiles
        plot_boxlike(dist.rv_frozen, x, density, quantiles, ax, color)

    side_legend(ax)
    return ax


def plot_cdf(dist, support, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    title = repr_to_matplotlib(dist)

    eps = dist._finite_endpoints(support)
    x = np.linspace(*eps, 1000)
    cdf = dist.rv_frozen.cdf(x)
    ax.plot(x, cdf, label=title, color=color)

    side_legend(ax)
    return ax


def plot_ppf(dist, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    title = repr_to_matplotlib(dist)

    x = np.linspace(0, 1, 1000)
    plt.plot(x, dist.rv_frozen.ppf(x), label=title, color=color)

    side_legend(ax)
    return ax


def get_ax(ax, figsize):
    if ax is None:
        fig_manager = _pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            ax = fig_manager.canvas.figure.gca()
        else:
            _, ax = plt.subplots(figsize=figsize)
    return ax


def side_legend(ax):
    bbox = ax.get_position()
    ax.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def repr_to_matplotlib(distribution):
    string = distribution.__repr__()
    string = string.replace("\x1b[1m", r"$\bf{")
    string = string.replace("\x1b[0m", "}$")
    return string
