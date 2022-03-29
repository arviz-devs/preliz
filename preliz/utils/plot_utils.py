import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers


default_quantiles = [0.05, 0.25, 0.75, 0.95]


def plot_boxlike(fitted_dist, x_vals, ref_pdf, quantiles, ax, color=None):
    """
    Plot the mean as a dot and two interquantile ranges as lines
    """
    q_s = fitted_dist.ppf(quantiles)
    mean = fitted_dist.moment(1)

    ax.plot(x_vals, ref_pdf, color=color)
    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")


def plot_boxlike2(sample, ax):
    """
    Plot the mean as a dot and two interquantile ranges as lines
    """
    q_s = np.quantile(sample, default_quantiles)
    mean = np.mean(sample)

    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")


def plot_dist(dist, box, quantiles, figsize, ax):
    ax = get_ax(ax, figsize)
    x = dist._xvals()
    color = next(ax._get_lines.prop_cycler)["color"]  # pylint: disable=protected-access
    title = dist.__repr__()
    if dist.kind == "continuous":
        pdf = dist.rv_frozen.pdf(x)
        ax.plot(x, pdf, label=title, color=color)
        ax.set_yticks([])
        if box:
            if quantiles is None:
                quantiles = default_quantiles
            plot_boxlike(dist.rv_frozen, x, pdf, quantiles, ax, color)
    else:
        ax.plot(x, dist.rv_frozen.pmf(x), "-o", label=title, color=color)

    side_legend(ax)


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
