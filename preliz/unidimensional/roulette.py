# pylint: disable=too-many-instance-attributes
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

try:
    import ipywidgets as widgets
except ImportError:
    pass

from preliz.internal.optimization import fit_to_epdf
from preliz.internal.plot_helper import check_inside_notebook, representations
from preliz.internal.distribution_helper import process_extra, get_distributions
from preliz.distributions import all_discrete, all_continuous


class Roulette:
    """
    Prior elicitation for 1D distribution using the roulette method.

    Draw 1D distributions using a grid as input.

    Parameters
    ----------
    x_min: Optional[float]
        Minimum value for the domain of the grid and fitted distribution.
    x_max: Optional[float]
        Maximum value for the domain of the grid and fitted distribution.
    nrows: Optional[int]
        Number of rows for the grid. Defaults to 10.
    ncols: Optional[int]
        Number of columns for the grid. Defaults to 11.
    dist_names: list
        List of distribution names to be used in the elicitation.
        Defaults to None. The pre-selected distributions are ["Normal", "BetaScaled",
        "Gamma", "LogNormal", "StudentT"], but almost all 1D PreliZ's distributions
        are available to be selected from the menu with some exceptions like Uniform
        or Cauchy.
    params: Optional[str]
        Extra parameters to be passed to the distributions. The format is a string with the
        PreliZ's distribution name followed by the argument to fix.
        For example: "TruncatedNormal(lower=0), StudentT(nu=8)". If you use the ``params``
        text area, quotation marks are not necessary.
    figsize: Optional[Tuple[int, int]]
        Figure size. If None, it will be defined automatically.

    Returns
    -------
    Roulette object
        The object has many attributes, but the most important are:
        - dist: The fitted distribution.
        - inputs: A tuple with the x values, the empirical pdf, the total
            chips, the x_min, the x_max, the number of rows, and the number of columns.

    References
    ----------
    * Morris D.E. et al. (2014) see https://doi.org/10.1016/j.envsoft.2013.10.010
    * See roulette mode http://optics.eee.nottingham.ac.uk/match/uncertainty.php
    """

    def __init__(
        self, x_min=0, x_max=10, nrows=10, ncols=11, dist_names=None, params=None, figsize=None
    ):
        self._x_min = x_min
        self._x_max = x_max
        self._nrows = nrows
        self._ncols = ncols
        self._dist_names = dist_names
        self._figsize = figsize
        self._w_extra = params
        self.dist = None
        self.inputs = None

        check_inside_notebook(need_widget=True)

        self._widgets = self._get_widgets()
        self._output = widgets.Output()

        with self._output:

            if self._figsize is None:
                self._figsize = (8, 6)

            self._fig, self._ax_grid, self._ax_fit = self._create_figure()
            self._coll = self._create_grid()
            self._grid = _Rectangles(self._fig, self._coll, self._nrows, self._ncols, self._ax_grid)

            self._setup_observers()

            self._fig.canvas.mpl_connect("button_release_event", lambda event: self._on_leave_fig())

        controls = widgets.VBox(
            [
                self._widgets["w_x_min"],
                self._widgets["w_x_max"],
                self._widgets["w_nrows"],
                self._widgets["w_ncols"],
                self._widgets["w_extra"],
            ]
        )
        control_distribution = widgets.VBox(
            [
                self._widgets["w_checkbox_cont"],
                self._widgets["w_checkbox_disc"],
                self._widgets["w_checkbox_none"],
            ]
        )
        display(  # pylint:disable=undefined-variable
            widgets.HBox(
                [
                    controls,
                    self._widgets["w_repr"],
                    self._widgets["w_distributions"],
                    control_distribution,
                ]
            )
        )

    def _create_figure(self):
        fig, axes = plt.subplots(2, 1, figsize=self._figsize, constrained_layout=True)
        ax_grid = axes[0]
        ax_fit = axes[1]
        ax_fit.set_yticks([])
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"
        return fig, ax_grid, ax_fit

    def _create_grid(self):
        xx = np.arange(self._ncols)
        yy = np.arange(self._nrows)

        if self._ncols < 11:
            num = self._ncols
        else:
            num = 11

        self._ax_grid.set(
            xticks=np.linspace(0, self._ncols - 1, num=num) + 0.5,
            xticklabels=[f"{i:.1f}" for i in np.linspace(self._x_min, self._x_max, num=num)],
        )

        coll = np.zeros((self._nrows, self._ncols), dtype=object)
        for idx, xi in enumerate(xx):
            for idy, yi in enumerate(yy):
                sq = patches.Rectangle((xi, yi), 1, 1, fill=True, facecolor="0.8", edgecolor="w")
                self._ax_grid.add_patch(sq)
                coll[idy, idx] = sq

        self._ax_grid.set_yticks([])
        self._ax_grid.relim()
        self._ax_grid.autoscale_view()
        return coll

    def _on_leave_fig(self):
        extra_pros = process_extra(self._widgets["w_extra"].value)

        x_vals, epdf, mean, std, filled_columns = self._weights_to_pdf()

        fitted_dist = None
        if filled_columns > 1:
            selected_distributions = get_distributions(self._widgets["w_distributions"].value)

            if selected_distributions:
                self._reset_dist_panel(yticks=False)
                fitted_dist = fit_to_epdf(
                    selected_distributions,
                    x_vals,
                    epdf,
                    mean,
                    std,
                    self._x_min,
                    self._x_max,
                    extra_pros,
                )

                if fitted_dist is None:
                    self._ax_fit.set_title("domain error")
                else:
                    representations(fitted_dist, self._widgets["w_repr"].value, self._ax_fit)
        else:
            self._reset_dist_panel(yticks=True)
        self._fig.canvas.draw()

        self.inputs = (
            x_vals,
            epdf,
            sum(self._grid._weights.values()),
            self._x_min,
            self._x_max,
            self._nrows,
            self._ncols,
        )
        self.dist = fitted_dist

    def _weights_to_pdf(self):
        step = (self._x_max - self._x_min) / (self._ncols - 1)
        x_vals = [(k + 0.5) * step + self._x_min for k, v in self._grid._weights.items() if v != 0]
        total = sum(self._grid._weights.values())
        epdf = [v / total for v in self._grid._weights.values() if v != 0]

        mean = sum(prob * value for value, prob in zip(x_vals, epdf))
        std = (sum(prob * (value - mean) ** 2 for value, prob in zip(x_vals, epdf))) ** 0.5

        return x_vals, epdf, mean, std, len(x_vals)

    def _update_grid(self):
        self._ax_grid.cla()
        self._coll = self._create_grid()
        self._grid._coll = self._coll
        self._grid._ncols = self._ncols
        self._grid._nrows = self._nrows
        self._grid._weights = {k: 0 for k in range(0, self._ncols)}
        self._reset_dist_panel(yticks=True)
        self._ax_grid.set_yticks([])
        self._ax_grid.relim()
        self._ax_grid.autoscale_view()
        self._fig.canvas.draw()

    def _reset_dist_panel(self, yticks):
        self._ax_fit.cla()
        if yticks:
            self._ax_fit.set_yticks([])
        self._ax_fit.set_xlim(self._x_min, self._x_max)
        self._ax_fit.relim()
        self._ax_fit.autoscale_view()

    def _handle_checkbox_widget(self):
        if self._widgets["w_checkbox_none"].value:
            self._widgets["w_checkbox_disc"].value = False
            self._widgets["w_checkbox_cont"].value = False
            return []
        all_cls = []
        if self._widgets["w_checkbox_cont"].value:
            all_cls += list(
                (
                    cls.__name__
                    for cls in all_continuous
                    if cls.__name__ in self._widgets["w_distributions"].options
                )
            )
        if self._widgets["w_checkbox_disc"].value:
            all_cls += list(
                (
                    cls.__name__
                    for cls in all_discrete
                    if cls.__name__ in self._widgets["w_distributions"].options
                )
            )
        return all_cls

    def _get_widgets(self):
        width_entry_text = widgets.Layout(width="150px")
        width_repr_text = widgets.Layout(width="250px")
        width_distribution_text = widgets.Layout(width="150px", height="125px")

        w_x_min = widgets.FloatText(
            value=self._x_min,
            step=1,
            description="x_min:",
            disabled=False,
            layout=width_entry_text,
        )

        w_x_max = widgets.FloatText(
            value=self._x_max,
            step=1,
            description="x_max:",
            disabled=False,
            layout=width_entry_text,
        )

        w_nrows = widgets.BoundedIntText(
            value=self._nrows,
            min=2,
            step=1,
            description="n_rows:",
            disabled=False,
            layout=width_entry_text,
        )

        w_ncols = widgets.BoundedIntText(
            value=self._ncols,
            min=2,
            step=1,
            description="n_cols:",
            disabled=False,
            layout=width_entry_text,
        )

        w_extra = widgets.Textarea(
            value=self._w_extra,
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

        if self._dist_names is None:
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
            default_dist = self._dist_names
            dist_names = self._dist_names

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

        return {
            "w_x_min": w_x_min,
            "w_x_max": w_x_max,
            "w_ncols": w_ncols,
            "w_nrows": w_nrows,
            "w_extra": w_extra,
            "w_repr": w_repr,
            "w_distributions": w_distributions,
            "w_checkbox_cont": w_checkbox_cont,
            "w_checkbox_disc": w_checkbox_disc,
            "w_checkbox_none": w_checkbox_none,
        }

    def _setup_observers(self):
        self._widgets["w_checkbox_none"].observe(self._handle_checkbox_change)
        self._widgets["w_checkbox_cont"].observe(self._handle_checkbox_change)
        self._widgets["w_checkbox_disc"].observe(self._handle_checkbox_change)

        def _update_grid_(_):
            self._x_min = self._widgets["w_x_min"].value
            self._x_max = self._widgets["w_x_max"].value
            self._nrows = self._widgets["w_nrows"].value
            self._ncols = self._widgets["w_ncols"].value
            self._update_grid()

        self._widgets["w_x_min"].observe(_update_grid_)
        self._widgets["w_x_max"].observe(_update_grid_)
        self._widgets["w_nrows"].observe(_update_grid_)
        self._widgets["w_ncols"].observe(_update_grid_)
        self._widgets["w_x_min"].observe(self._on_value_change, names="value")

        def _on_leave_fig_(_):
            self._on_leave_fig()

        self._widgets["w_repr"].observe(_on_leave_fig_)
        self._widgets["w_distributions"].observe(_on_leave_fig_)
        self._widgets["w_extra"].observe(_on_leave_fig_)

    def _handle_checkbox_change(self, _):
        dist_names = self._handle_checkbox_widget()
        self._widgets["w_distributions"].value = dist_names

    def _on_value_change(self, change):
        new_a = change["new"]
        if new_a == self._widgets["w_x_max"].value:
            self._widgets["w_x_max"].value = new_a + 1


class _Rectangles:
    def __init__(self, fig, coll, nrows, ncols, ax):
        self._fig = fig
        self._coll = coll
        self._nrows = nrows
        self._ncols = ncols
        self._ax = ax
        self._weights = {k: 0 for k in range(0, ncols)}
        fig.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        if event.inaxes == self._ax:
            x = event.xdata
            y = event.ydata
            idx = floor(x)
            idy = ceil(y)

            if 0 <= idx < self._ncols and 0 <= idy <= self._nrows:
                if self._weights[idx] >= idy:
                    idy -= 1
                    for row in range(self._nrows):
                        self._coll[row, idx].set_facecolor("0.8")
                self._weights[idx] = idy
                for row in range(idy):
                    self._coll[row, idx].set_facecolor("C1")
                self._fig.canvas.draw()
