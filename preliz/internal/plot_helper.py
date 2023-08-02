import inspect
import logging
import traceback
import sys

from IPython import get_ipython

try:
    from ipywidgets import FloatSlider, IntSlider
except ImportError:
    pass
from arviz import plot_kde, plot_ecdf, hdi
from arviz.stats.density_utils import _kde_linear
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, get_backend
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d, PchipInterpolator

_log = logging.getLogger("preliz")


def plot_pointinterval(distribution, interval="hdi", levels=None, rotated=False, ax=None):
    """
    Plot median as a dot and intervals as lines.
    By defaults the intervals are HDI with 0.5 and 0.94 mass.

    Parameters
    ----------
    distribution : preliz distribution or array
    interval : str
        Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
    levels : list
        Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
        For quantiles the number of elements should be 5, 3, 1 or 0
        (in this last case nothing will be plotted).
    rotated : bool
        Whether to do the plot along the x-axis (default) or on the y-axis
    ax : matplotlib axis
    """
    if interval == "quantiles":
        if levels is None:
            levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        elif len(levels) not in (5, 3, 1, 0):
            raise ValueError("levels should have 5, 3, 1 or 0 elements")

        if isinstance(distribution, (np.ndarray, list, tuple)):
            q_s = np.quantile(distribution, levels).tolist()
        else:
            q_s = distribution.ppf(levels).tolist()

    else:
        if levels is None:
            levels = [0.5, 0.94]

        elif len(levels) not in (2, 1):
            raise ValueError("levels should have 2 or 1 elements")

        if isinstance(distribution, (np.ndarray, list, tuple)):
            if interval == "hdi":
                func = hdi
            if interval == "eti":
                func = eti

            q_tmp = np.concatenate([func(distribution, m) for m in levels])
            median = np.median(distribution)
        else:
            if interval == "hdi":
                func = distribution.hdi
            if interval == "eti":
                func = distribution.eti

            q_tmp = np.concatenate([func(mass=m) for m in levels])
            median = distribution.rv_frozen.median()

        q_s = []
        if len(levels) == 2:
            q_s.extend((q_tmp[2], q_tmp[0], median, q_tmp[1], q_tmp[3]))
        elif len(levels) == 1:
            q_s.extend((q_tmp[0], median, q_tmp[1]))

    q_s_size = len(q_s)

    if rotated:
        if q_s_size == 5:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", solid_capstyle="butt", lw=1.5)
        if q_s_size > 2:
            ax.plot([0, 0], (q_s.pop(0), q_s.pop(-1)), "k", solid_capstyle="butt", lw=4)
        if q_s_size > 0:
            ax.plot(0, q_s[0], "wo", mec="k")
    else:
        if q_s_size == 5:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", solid_capstyle="butt", lw=1.5)
        if q_s_size > 2:
            ax.plot((q_s.pop(0), q_s.pop(-1)), [0, 0], "k", solid_capstyle="butt", lw=4)

        if q_s_size > 0:
            ax.plot(q_s[0], 0, "wo", mec="k")


def eti(distribution, mass):
    lower = (1 - mass) / 2
    return np.quantile(distribution, (lower, mass + lower))


def plot_pdfpmf(
    dist, moments, pointinterval, interval, levels, support, legend, color, alpha, figsize, ax
):
    ax = get_ax(ax, figsize)
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if legend is not None:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    x = dist.xvals(support)
    if dist.kind == "continuous":
        density = dist.pdf(x)
        ax.axhline(0, color="0.8", ls="--", zorder=0)
        ax.plot(x, density, label=label, color=color, alpha=alpha)
        ax.set_yticks([])
    else:
        mass = dist.pdf(x)
        eps = np.clip(dist._finite_endpoints(support), *dist.support)
        x_c = np.linspace(*eps, 1000)

        if len(x) > 2:
            interp = PchipInterpolator(x, mass)
        else:
            interp = interp1d(x, mass)

        mass_c = np.clip(interp(x_c), np.min(mass), np.max(mass))

        ax.axhline(0, color="0.8", ls="--", zorder=0)
        ax.plot(x_c, mass_c, ls="dotted", color=color, alpha=alpha)
        ax.plot(x, mass, "o", label=label, color=color, alpha=alpha)

    if pointinterval:
        plot_pointinterval(dist, interval, levels, ax=ax)

    if legend == "legend":
        side_legend(ax)

    return ax


def plot_cdf(
    dist, moments, pointinterval, interval, levels, support, legend, color, alpha, figsize, ax
):
    ax = get_ax(ax, figsize)
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if legend is not None:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    ax.set_ylim(-0.05, 1.05)
    x = dist.xvals(support)

    if dist.kind == "discrete":
        lower = x[0]
        upper = x[-1]
        x = np.insert(x, [0, len(x)], (lower - 1, upper + 1))
        cdf = dist.cdf(x)
        ax.set_xlim(lower - 0.1, upper + 0.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.step(x, cdf, where="post", label=label, color=color, alpha=alpha)
    else:
        cdf = dist.cdf(x)
        ax.plot(x, cdf, label=label, color=color, alpha=alpha)

    if pointinterval:
        plot_pointinterval(dist, interval, levels, ax=ax)

    if legend == "legend":
        side_legend(ax)
    return ax


def plot_ppf(dist, moments, pointinterval, interval, levels, legend, color, alpha, figsize, ax):
    ax = get_ax(ax, figsize)
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    if legend is not None:
        label = repr_to_matplotlib(dist)

        if moments is not None:
            label += get_moments(dist, moments)

        if legend == "title":
            ax.set_title(label)
            label = None
    else:
        label = None

    x = np.linspace(0, 1, 1000)
    ax.plot(x, dist.ppf(x), label=label, color=color, alpha=alpha)
    if dist.kind == "discrete":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if pointinterval:
        plot_pointinterval(dist, interval, levels, rotated=True, ax=ax)

    if legend == "legend":
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
    string = repr(distribution)
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


def get_slider(name, value, lower, upper, continuous_update=True):

    min_v, max_v, step = generate_range(value, lower, upper)

    if isinstance(value, float):
        slider_type = FloatSlider
    else:
        slider_type = IntSlider
        step = 1

    slider = slider_type(
        min=min_v,
        max=max_v,
        step=step,
        description=f"{name} ({lower:.0f}, {upper:.0f})",
        value=value,
        style={"description_width": "initial"},
        continuous_update=continuous_update,
    )

    return slider


def generate_range(value, lower, upper):
    if value == 0:
        return -10, 10, 0.1

    power = np.floor(np.log10(abs(value)))
    order_of_magnitude = 10**power
    decimal_places = int(-power)
    std = order_of_magnitude * 10 - order_of_magnitude / 10
    min_v = round(value - std, decimal_places)
    max_v = round(value + std, decimal_places)

    min_v = lower if np.isfinite(lower) else min_v
    max_v = upper if np.isfinite(upper) else max_v

    step = order_of_magnitude / 10
    return min_v, max_v, step


def get_sliders(signature, model):
    sliders = {}
    for name, param in signature.parameters.items():
        if isinstance(param.default, (int, float)):
            value = float(param.default)
        else:
            value = None

        dist, idx, func = model[name]
        lower, upper = dist.params_support[idx]
        # ((eps, 1 - eps),
        # (-np.inf, np.inf),
        # (eps, np.inf))
        # ((-np.pi, np.pi)
        if func is not None and lower == np.finfo(float).eps:
            if func in ["exp", "abs", "expit", "logistic"]:
                lower = -np.inf
            elif func in ["log"]:
                lower = 1.0
            elif func.replace(".", "", 1).isdigit():
                if not float(func) % 2:
                    lower = -np.inf

        if func is not None and upper == 1 - np.finfo(float).eps:
            if func in ["expit", "logistic"]:
                lower = np.inf

        if value is None:
            value = getattr(dist, dist.param_names[idx])

        sliders[name] = get_slider(name, value, lower, upper, continuous_update=False)
    return sliders


def plot_decorator(func, iterations, kind_plot):
    def looper(*args, **kwargs):
        results = [func(*args, **kwargs) for _ in range(iterations)]
        _, ax = plt.subplots()

        alpha = max(0.01, 1 - iterations * 0.009)

        if kind_plot == "hist":
            if results[0].dtype.kind == "i":
                bins = np.arange(np.min(results), np.max(results) + 1.5) - 0.5
                if len(bins) < 30:
                    ax.set_xticks(bins + 0.5)
            else:
                bins = "auto"
            ax.hist(
                results,
                alpha=alpha,
                density=True,
                color=["C0"] * iterations,
                bins=bins,
                histtype="step",
            )
            ax.hist(
                np.concatenate(results),
                density=True,
                bins=bins,
                color="k",
                ls="--",
                histtype="step",
            )
        elif kind_plot == "kde":
            for result in results:
                plt.plot(*_kde_linear(result, grid_len=100), "C0", alpha=alpha)
            plt.plot(*_kde_linear(np.concatenate(results), grid_len=100), "k--")
        elif kind_plot == "ecdf":
            plt.plot(
                np.sort(results, axis=1).T,
                np.linspace(0, 1, len(results[0]), endpoint=False),
                color="C0",
            )
            a = np.concatenate(results)
            plt.plot(np.sort(a), np.linspace(0, 1, len(a), endpoint=False), "k--")

    return looper


def plot_pp_samples(pp_samples, pp_samples_idxs, references, kind="pdf", sharex=True, fig=None):
    row_colum = int(np.ceil(len(pp_samples_idxs) ** 0.5))

    if fig is None:
        fig, axes = plt.subplots(row_colum, row_colum, figsize=(8, 6), constrained_layout=True)
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    else:
        axes = np.array(fig.axes)

    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    x_lims = [np.inf, -np.inf]

    for ax, idx in zip(axes, pp_samples_idxs):
        ax.clear()
        for ref in references:
            ax.axvline(ref, ls="--", color="0.5")
        ax.relim()

        sample = pp_samples[idx]

        if sharex:
            min_ = sample.min()
            max_ = sample.max()
            if min_ < x_lims[0]:
                x_lims[0] = min_
            if max_ > x_lims[1]:
                x_lims[1] = max_

        if kind == "pdf":
            plot_kde(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member
        elif kind == "hist":
            bins, *_ = ax.hist(
                sample, color="C0", bins="auto", alpha=0.5, density=True
            )  # pylint:disable=no-member
            ax.set_ylim(-bins.max() * 0.05, None)

        elif kind == "ecdf":
            plot_ecdf(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member

        plot_pointinterval(sample, ax=ax)
        ax.set_title(idx, alpha=0)
        ax.set_yticks([])

    if sharex:
        for ax in axes:
            ax.set_xlim(np.floor(x_lims[0]), np.ceil(x_lims[1]))

    fig.canvas.draw()
    return fig, axes


def plot_pp_mean(pp_samples, selected, references=None, kind="pdf", fig_pp_mean=None):
    if fig_pp_mean is None:
        fig_pp_mean, ax_pp_mean = plt.subplots(1, 1, figsize=(8, 2), constrained_layout=True)
        fig_pp_mean.canvas.header_visible = False
        fig_pp_mean.canvas.footer_visible = False
    else:
        ax_pp_mean = fig_pp_mean.axes[0]

    ax_pp_mean.clear()

    if np.any(selected):
        sample = pp_samples[selected].ravel()
    else:
        sample = pp_samples.ravel()

    for ref in references:
        ax_pp_mean.axvline(ref, ls="--", color="0.5")

    if kind == "pdf":
        plot_kde(
            sample, ax=ax_pp_mean, plot_kwargs={"color": "k", "linestyle": "--"}
        )  # pylint:disable=no-member
    elif kind == "hist":
        bins, *_ = ax_pp_mean.hist(
            sample, color="k", ls="--", bins="auto", alpha=0.5, density=True
        )  # pylint:disable=no-member
        ax_pp_mean.set_ylim(-bins.max() * 0.05, None)

    elif kind == "ecdf":
        plot_ecdf(
            sample, ax=ax_pp_mean, plot_kwargs={"color": "k", "linestyle": "--"}
        )  # pylint:disable=no-member

    plot_pointinterval(sample, ax=ax_pp_mean)
    ax_pp_mean.set_yticks([])
    fig_pp_mean.canvas.draw()

    return fig_pp_mean


def check_inside_notebook(need_widget=False):
    shell = get_ipython()
    name = inspect.currentframe().f_back.f_code.co_name
    try:
        if shell is None:
            raise RuntimeError(
                f"To run {name}, you need to call it from within a Jupyter notebook or Jupyter lab."
            )
        if need_widget:
            shell_name = shell.__class__.__name__
            if shell_name == "ZMQInteractiveShell" and "nbagg" not in get_backend():
                msg = f"To run {name}, you need use the magic `%matplotlib widget`"
                raise RuntimeError(msg)
    except Exception:  # pylint: disable=broad-except
        tb_as_str = traceback.format_exc()
        # Print only the last line of the traceback, which contains the error message
        print(tb_as_str.strip().rsplit("\n", maxsplit=1)[-1], file=sys.stdout)
