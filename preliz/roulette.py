import logging
from math import ceil, floor


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, get_backend


import ipywidgets as widgets
from preliz.utils.optimization import fit_to_ecdf, get_distributions


_log = logging.getLogger("preliz")


def roulette(x_min=0, x_max=10, nrows=10, ncols=10, figsize=None):
    """
    Prior elicitation for 1D distribution using the roulette method.

    Draw 1D distributions using a grid as input.

    Parameters
    ----------
    x_min: Optional[float]
        Minimum value for the domain of the grid and fitted distribution
    x_max: Optional[float]
        Maximum value for the domain of the grid and fitted distribution
    nrows: Optional[int]
        Number of rows for the grid. Defaults to 10.
    ncols: Optional[int]
        Number of columns for the grid. Defaults to 10.
    figsize: Optional[Tuple[int, int]]
        Figure size. If None it will be defined automatically.

    Returns
    -------
    PreliZ distribution

    References
    ----------
    * Morris D.E. et al. (2014) see https://doi.org/10.1016/j.envsoft.2013.10.010
    * See roulette mode http://optics.eee.nottingham.ac.uk/match/uncertainty.php
    """
    try:
        shell = get_ipython().__class__.__name__  # pylint:disable=undefined-variable
        if shell == "ZMQInteractiveShell" and "nbagg" not in get_backend():
            _log.info(
                "To run roulette you need Jupyter notebook, or Jupyter lab."
                "You will also need to use the magic `%matplotlib widget`"
            )
    except NameError:
        pass

    w_x_min, w_x_max, w_ncols, w_nrows, w_repr, w_distributions = get_widgets(
        x_min, x_max, nrows, ncols
    )

    output = widgets.Output()

    with output:
        x_min = w_x_min.value
        x_max = w_x_max.value
        nrows = w_nrows.value
        ncols = w_ncols.value

        if figsize is None:
            figsize = (8, 6)

        fig, ax_grid, ax_fit = create_figure(figsize)

        coll = create_grid(x_min, x_max, nrows, ncols, ax=ax_grid)
        grid = Rectangles(fig, coll, nrows, ncols, ax_grid)

        def update_grid_(_):
            update_grid(
                fig.canvas,
                w_x_min.value,
                w_x_max.value,
                w_nrows.value,
                w_ncols.value,
                grid,
                ax_grid,
                ax_fit,
            )

        w_x_min.observe(update_grid_)
        w_x_max.observe(update_grid_)
        w_nrows.observe(update_grid_)
        w_ncols.observe(update_grid_)

        def on_leave_fig_(_):
            on_leave_fig(
                fig.canvas,
                grid,
                w_distributions.value,
                w_repr.value,
                w_x_min.value,
                w_x_max.value,
                ncols,
                ax_fit,
            )

        w_repr.observe(on_leave_fig_)
        w_distributions.observe(on_leave_fig_)

        def on_value_change(change):
            new_a = change["new"]
            if new_a == w_x_max.value:
                w_x_max.value = new_a + 1

        w_x_min.observe(on_value_change, names="value")

        fig.canvas.mpl_connect(
            "button_release_event",
            lambda event: on_leave_fig(
                fig.canvas,
                grid,
                w_distributions.value,
                w_repr.value,
                w_x_min.value,
                w_x_max.value,
                ncols,
                ax_fit,
            ),
        )

    controls = widgets.VBox([w_x_min, w_x_max, w_nrows, w_ncols])

    display(widgets.HBox([controls, w_repr, w_distributions]))  # pylint:disable=undefined-variable


def create_figure(figsize):
    """
    Initialize a matplotlib figure with two subplots
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    ax_grid = axes[0]
    ax_fit = axes[1]
    ax_fit.set_yticks([])
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.toolbar_position = "right"

    return fig, ax_grid, ax_fit


def create_grid(x_min=0, x_max=1, nrows=10, ncols=10, ax=None):
    """
    Create a grid of rectangles
    """
    xx = np.arange(ncols)
    yy = np.arange(nrows)

    if ncols < 10:
        num = ncols
    else:
        num = 10

    ax.set(
        xticks=np.linspace(0, ncols, num=num),
        xticklabels=[f"{i:.1f}" for i in np.linspace(x_min, x_max, num=num)],
    )

    coll = np.zeros((nrows, ncols), dtype=object)
    for idx, xi in enumerate(xx):
        for idy, yi in enumerate(yy):
            sq = patches.Rectangle((xi, yi), 1, 1, fill=True, facecolor="0.8", edgecolor="w")
            ax.add_patch(sq)
            coll[idy, idx] = sq

    ax.set_yticks([])
    ax.relim()
    ax.autoscale_view()
    return coll


class Rectangles:
    """
    Clickable rectangles
    Clicked rectangles are highlighted
    """

    def __init__(self, fig, coll, nrows, ncols, ax):
        self.fig = fig
        self.coll = coll
        self.nrows = nrows
        self.ncols = ncols
        self.ax = ax
        self.weights = {k: 0 for k in range(0, ncols)}
        fig.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        if event.inaxes == self.ax:
            x = event.xdata
            y = event.ydata
            idx = floor(x)
            idy = ceil(y)

            if 0 <= idx < self.ncols and 0 <= idy <= self.nrows:
                if self.weights[idx] >= idy:
                    idy -= 1
                    for row in range(self.nrows):
                        self.coll[row, idx].set_facecolor("0.8")
                self.weights[idx] = idy
                for row in range(idy):
                    self.coll[row, idx].set_facecolor("C1")
                self.fig.canvas.draw()


def on_leave_fig(canvas, grid, dist_names, kind_plot, x_min, x_max, ncols, ax):
    x_min = float(x_min)
    x_max = float(x_max)
    ncols = float(ncols)
    x_range = x_max - x_min

    x_vals, ecdf, mean, std, filled_columns = weights_to_ecdf(grid.weights, x_min, x_range, ncols)

    if filled_columns > 1:
        selected_distributions = get_distributions(dist_names)

        if selected_distributions:
            reset_dist_panel(x_min, x_max, ax, yticks=False)
            fitted_dist = fit_to_ecdf(
                selected_distributions,
                x_vals,
                ecdf,
                mean,
                std,
                x_min,
                x_max,
            )

            if fitted_dist is None:
                ax.set_title("domain error")
            else:
                representations(fitted_dist, kind_plot, ax)
    else:
        reset_dist_panel(x_min, x_max, ax, yticks=True)
    canvas.draw()


def weights_to_ecdf(weights, x_min, x_range, ncols):
    """
    Turn the weights (chips) into the empirical cdf
    """
    filled_columns = 0
    x_vals = []
    ecdf = []
    cum_sum = 0

    values = list(weights.values())
    mean = np.mean(values)
    std = np.std(values)
    total = sum(values)
    if any(weights.values()):
        for k, v in weights.items():
            if v != 0:
                filled_columns += 1
            x_val = (k / ncols * x_range) + x_min + ((x_range / ncols))
            x_vals.append(x_val)
            cum_sum += v / total
            ecdf.append(cum_sum)

    return x_vals, ecdf, mean, std, filled_columns


def representations(fitted_dist, kind_plot, ax):
    if kind_plot == "pdf":
        fitted_dist.plot_pdf(pointinterval=True, legend="title", ax=ax)
        ax.set_yticks([])

        for bound in fitted_dist.support:
            if np.isfinite(bound):
                ax.plot(bound, 0, "ko")

    elif kind_plot == "cdf":
        fitted_dist.plot_cdf(pointinterval=True, legend="title", ax=ax)

    elif kind_plot == "ppf":
        fitted_dist.plot_ppf(pointinterval=True, legend="title", ax=ax)
        ax.set_xlim(-0.01, 1)


def update_grid(canvas, x_min, x_max, nrows, ncols, grid, ax_grid, ax_fit):
    """
    Update the grid subplot
    """
    ax_grid.cla()
    coll = create_grid(x_min=x_min, x_max=x_max, nrows=nrows, ncols=ncols, ax=ax_grid)
    grid.coll = coll
    grid.ncols = ncols
    grid.nrows = nrows
    grid.weights = {k: 0 for k in range(0, ncols)}
    reset_dist_panel(x_min, x_max, ax_fit, yticks=True)
    ax_grid.set_yticks([])
    ax_grid.relim()
    ax_grid.autoscale_view()
    canvas.draw()


def reset_dist_panel(x_min, x_max, ax, yticks):
    """
    Clean the distribution subplot
    """
    ax.cla()
    if yticks:
        ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.relim()
    ax.autoscale_view()


def get_widgets(x_min, x_max, nrows, ncols):

    width_entry_text = widgets.Layout(width="150px")

    w_x_min = widgets.IntText(
        value=x_min,
        step=1,
        description="x_min:",
        disabled=False,
        layout=width_entry_text,
    )

    w_x_max = widgets.IntText(
        value=x_max,
        step=1,
        description="x_max:",
        disabled=False,
        layout=width_entry_text,
    )

    w_nrows = widgets.BoundedIntText(
        value=nrows,
        min=2,
        step=1,
        description="n_rows:",
        disabled=False,
        layout=width_entry_text,
    )

    w_ncols = widgets.BoundedIntText(
        value=ncols,
        min=2,
        step=1,
        description="n_cols:",
        disabled=False,
        layout=width_entry_text,
    )

    w_repr = widgets.RadioButtons(
        options=["pdf", "cdf", "ppf"],
        value="pdf",
        description="",
        disabled=False,
        layout=width_entry_text,
    )

    dist_names = ["Normal", "BetaScaled", "Gamma", "LogNormal", "Student"]

    w_distributions = widgets.SelectMultiple(
        options=dist_names, value=dist_names, description="", disabled=False
    )

    return w_x_min, w_x_max, w_ncols, w_nrows, w_repr, w_distributions
