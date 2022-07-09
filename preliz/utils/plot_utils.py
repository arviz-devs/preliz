import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen


def plot_pointinterval(distribution, quantiles, ax):
    """
    Plot the median as a dot and two inter-quantile ranges as lines
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.75, 0.95]

    if isinstance(distribution, (rv_continuous_frozen, rv_discrete_frozen)):
        q_s = distribution.ppf(quantiles)
        median = distribution.ppf(0.5)
    else:
        q_s = np.quantile(distribution, quantiles)
        median = np.quantile(distribution, 0.5)

    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(median, 0, "w.")


def plot_pdfpmf(dist, pointinterval, quantiles, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)

    x = dist.xvals(support)
    if dist.kind == "continuous":
        density = dist.rv_frozen.pdf(x)
        ax.plot(x, density, label=label, color=color)
        ax.set_yticks([])
    else:
        density = dist.rv_frozen.pmf(x)
        ax.plot(x, density, "-o", label=label, color=color)
    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles, ax)

    if legend == "title":
        ax.set_title(label)
    elif legend == "legend":
        side_legend(ax)
    return ax


def plot_cdf(dist, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)

    eps = dist.finite_endpoints(support)
    x = np.linspace(*eps, 1000)
    cdf = dist.rv_frozen.cdf(x)
    ax.plot(x, cdf, label=label, color=color)

    if legend == "title":
        ax.set_title(label)
    elif legend == "legend":
        side_legend(ax)
    return ax


def plot_ppf(dist, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)

    x = np.linspace(0, 1, 1000)
    ax.plot(x, dist.rv_frozen.ppf(x), label=label, color=color)

    if legend == "title":
        ax.set_title(label)
    elif legend == "legend":
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
