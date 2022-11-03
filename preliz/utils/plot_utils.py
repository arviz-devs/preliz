import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
from scipy.interpolate import interp1d, PchipInterpolator


def plot_pointinterval(distribution, quantiles=None, rotated=False, ax=None):
    """
    By default plot the quantiles [0.05, 0.25, 0.5, 0.75, 0.95]
    The median as dot and the two interquantiles ranges (0.05-0.95) and (0.25-0.75) as lines.

    Parameters
    ----------
    distribution : preliz distribution or array
    quantiles : list
        The number of elements should be 5, 3, 1 or 0 (in this last case nothing will be plotted).
        defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
    rotated : bool
        Whether to do the plot along the x-axis (default) or on the y-axis
    ax : matplotlib axis
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    if isinstance(distribution, (rv_continuous_frozen, rv_discrete_frozen)):
        q_s = distribution.ppf(quantiles).tolist()
    else:
        q_s = np.quantile(distribution, quantiles).tolist()

    q_s_size = len(q_s)
    if not q_s_size in (5, 3, 1, 0):
        raise ValueError("quantiles should have 5, 3, 1 or 0 elements")

    if rotated:
        if q_s_size == 5:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", lw=1.5)
        if q_s_size > 2:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", lw=4)
        if q_s_size > 0:
            ax.plot(0, q_s[0], "wo", mec="k")
    else:
        if q_s_size == 5:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", lw=1.5)
        if q_s_size > 2:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", lw=4)
        if q_s_size > 0:
            ax.plot(q_s[0], 0, "wo", mec="k")


def plot_pdfpmf(dist, moments, pointinterval, quantiles, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)
    if moments is not None:
        label += get_moments(dist, moments)

    x = dist.xvals(support)
    if dist.kind == "continuous":
        density = dist.rv_frozen.pdf(x)
        ax.plot(x, density, label=label, color=color)
        ax.set_yticks([])
    else:
        mass = dist.rv_frozen.pmf(x)
        eps = np.clip(dist._finite_endpoints(support), *dist.support)
        x_c = np.linspace(*eps, 1000)

        if len(x) > 2:
            interp = PchipInterpolator(x, mass)
        else:
            interp = interp1d(x, mass)

        mass_c = np.clip(interp(x_c), np.min(mass), np.max(mass))

        ax.plot(x_c, mass_c, ls="dotted", color=color)
        ax.plot(x, mass, "o", label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, ax=ax)

    if legend == "title":
        ax.set_title(label)
    elif legend == "legend":
        side_legend(ax)
    return ax


def plot_cdf(dist, moments, pointinterval, quantiles, support, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)
    if moments is not None:
        label += get_moments(dist, moments)

    eps = dist._finite_endpoints(support)
    x = np.linspace(*eps, 1000)
    cdf = dist.rv_frozen.cdf(x)
    ax.plot(x, cdf, label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, ax=ax)

    if legend == "title":
        ax.set_title(label)
    elif legend == "legend":
        side_legend(ax)
    return ax


def plot_ppf(dist, moments, pointinterval, quantiles, legend, figsize, ax):
    ax = get_ax(ax, figsize)
    color = next(ax._get_lines.prop_cycler)["color"]
    label = repr_to_matplotlib(dist)
    if moments is not None:
        label += get_moments(dist, moments)

    x = np.linspace(0, 1, 1000)
    ax.plot(x, dist.rv_frozen.ppf(x), label=label, color=color)

    if pointinterval:
        plot_pointinterval(dist.rv_frozen, quantiles=quantiles, rotated=True, ax=ax)

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


def get_moments(dist, moments):
    names = {
        "m": "μ",
        "d": "σ",
        "s": "γ",
        "v": "σ²",
        "k": "κ",
    }
    str_m = []
    seen = []
    for moment in moments:
        if moment not in seen:
            if moment == "d":
                value = dist.rv_frozen.stats("v") ** 0.5
            else:
                value = dist.rv_frozen.stats(moment)
            if isinstance(value, (np.ndarray, int, float)):
                str_m.append(f"{names[moment]}={value:.3g}")

        seen.append(moment)

    return "\n" + ", ".join(str_m)
