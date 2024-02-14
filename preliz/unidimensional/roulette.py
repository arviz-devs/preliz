from math import ceil, floor


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

try:
    import ipywidgets as widgets
except ImportError:
    pass
from ..internal.optimization import fit_to_ecdf, get_distributions
from ..internal.plot_helper import check_inside_notebook, representations
from ..internal.distribution_helper import process_extra
from ..distributions import all_discrete, all_continuous


def roulette(x_min=0, x_max=10, nrows=10, ncols=11, dist_names=None, figsize=None):
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
        Number of columns for the grid. Defaults to 11.
    dist_names: list
        List of distributions names to be used in the elicitation. If None, almost all 1D
        distributions available in PreliZ will be used. Some distributions like Uniform or
        Cauchy are omitted by default.
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

    check_inside_notebook(need_widget=True)

    (
        w_x_min,
        w_x_max,
        w_ncols,
        w_nrows,
        w_extra,
        w_repr,
        w_distributions,
        w_checkbox_cont,
        w_checkbox_disc,
        w_checkbox_none,
    ) = get_widgets(
        x_min,
        x_max,
        nrows,
        ncols,
        dist_names,
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

        def handle_checkbox_change(_):
            dist_names = handle_checkbox_widget(
                w_distributions.options, w_checkbox_cont, w_checkbox_disc, w_checkbox_none
            )
            w_distributions.value = dist_names

        w_checkbox_none.observe(handle_checkbox_change)
        w_checkbox_cont.observe(handle_checkbox_change)
        w_checkbox_disc.observe(handle_checkbox_change)

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
                w_ncols.value,
                w_extra.value,
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
                w_ncols.value,
                w_extra.value,
                ax_fit,
            ),
        )

    controls = widgets.VBox([w_x_min, w_x_max, w_nrows, w_ncols, w_extra])
    control_distribution = widgets.VBox([w_checkbox_cont, w_checkbox_disc, w_checkbox_none])
    display(  # pylint:disable=undefined-variable
        widgets.HBox(
            [
                controls,
                w_repr,
                w_distributions,
                control_distribution,
            ]
        )
    )


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

    if ncols < 11:
        num = ncols
    else:
        num = 11

    ax.set(
        xticks=np.linspace(0, ncols - 1, num=num) + 0.5,
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


def on_leave_fig(canvas, grid, dist_names, kind_plot, x_min, x_max, ncols, extra, ax):
    x_min = float(x_min)
    x_max = float(x_max)
    ncols = float(ncols)
    x_range = x_max - x_min
    extra_pros = process_extra(extra)

    x_vals, ecdf, mean, std, filled_columns = weights_to_ecdf(grid.weights, x_min, x_range, ncols)

    fitted_dist = None
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
                extra_pros,
            )

            if fitted_dist is None:
                ax.set_title("domain error")
            else:
                representations(fitted_dist, kind_plot, ax)
    else:
        reset_dist_panel(x_min, x_max, ax, yticks=True)
    canvas.draw()

    return fitted_dist


def weights_to_ecdf(weights, x_min, x_range, ncols):
    """
    Turn the weights (chips) into the empirical cdf
    """
    step = x_range / (ncols - 1)
    x_vals = [(k + 0.5) * step + x_min for k, v in weights.items() if v != 0]
    total = sum(weights.values())
    probabilities = [v / total for v in weights.values() if v != 0]
    cum_sum = np.cumsum(probabilities)

    mean = sum(value * prob for value, prob in zip(x_vals, probabilities))
    std = (sum(prob * (value - mean) ** 2 for value, prob in zip(x_vals, probabilities))) ** 0.5

    return x_vals, cum_sum, mean, std, len(x_vals)


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


def handle_checkbox_widget(options, w_checkbox_cont, w_checkbox_disc, w_checkbox_none):
    if w_checkbox_none.value:
        w_checkbox_disc.value = False
        w_checkbox_cont.value = False
        return []
    all_cls = []
    if w_checkbox_cont.value:
        all_cont_str = [  # pylint:disable=unnecessary-comprehension
            dist for dist in (cls.__name__ for cls in all_continuous if cls.__name__ in options)
        ]
        all_cls += all_cont_str
    if w_checkbox_disc.value:
        all_dist_str = [  # pylint:disable=unnecessary-comprehension
            dist for dist in (cls.__name__ for cls in all_discrete if cls.__name__ in options)
        ]
        all_cls += all_dist_str
    return all_cls


def get_widgets(x_min, x_max, nrows, ncols, dist_names):

    width_entry_text = widgets.Layout(width="150px")
    width_repr_text = widgets.Layout(width="250px")
    width_distribution_text = widgets.Layout(width="150px", height="125px")

    w_x_min = widgets.FloatText(
        value=x_min,
        step=1,
        description="x_min:",
        disabled=False,
        layout=width_entry_text,
    )

    w_x_max = widgets.FloatText(
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

    w_extra = widgets.Textarea(
        value="",
        placeholder="Pass extra parameters",
        description="params:",
        disabled=False,
        layout=width_repr_text,
    )

    w_repr = widgets.RadioButtons(
        options=["pdf", "cdf", "ppf"],
        value="pdf",
        description="",
        disabled=False,
        layout=width_entry_text,
    )

    if dist_names is None:

        default_dist = ["Normal", "BetaScaled", "Gamma", "LogNormal", "StudentT"]

        dist_names = [
            "AsymmetricLaplace",
            "BetaScaled",
            "ChiSquared",
            "ExGaussian",
            "Exponential",
            "Gamma",
            "Gumbel",
            "HalfNormal",
            "HalfStudentT",
            "InverseGamma",
            "Laplace",
            "LogNormal",
            "Logistic",
            # "LogitNormal", # fails if we add chips at x_value= 1
            "Moyal",
            "Normal",
            "Pareto",
            "Rice",
            "SkewNormal",
            "StudentT",
            "Triangular",
            "VonMises",
            "Wald",
            "Weibull",
            "BetaBinomial",
            "DiscreteWeibull",
            "Geometric",
            "NegativeBinomial",
            "Poisson",
        ]

    else:
        default_dist = dist_names

    w_distributions = widgets.SelectMultiple(
        options=dist_names,
        value=default_dist,
        description="",
        disabled=False,
        layout=width_distribution_text,
    )

    w_checkbox_cont = widgets.Checkbox(
        value=False, description="Continuous", disabled=False, indent=False
    )
    w_checkbox_disc = widgets.Checkbox(
        value=False, description="Discrete", disabled=False, indent=False
    )
    w_checkbox_none = widgets.Checkbox(
        value=False, description="None", disabled=False, indent=False
    )

    return (
        w_x_min,
        w_x_max,
        w_ncols,
        w_nrows,
        w_extra,
        w_repr,
        w_distributions,
        w_checkbox_cont,
        w_checkbox_disc,
        w_checkbox_none,
    )
