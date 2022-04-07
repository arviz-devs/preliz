"""Prior elicitation using roulette method."""

import tkinter as tk
from math import ceil, floor

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import patches

import numpy as np


from .utils.roulette_utils import weights_to_sample, get_distributions, fit_to_sample

# pylint: disable=unused-argument
def roulette(x_min=0, x_max=10, nrows=10, ncols=10, parametrization="PyMC", figsize=None):
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
    parametrization: Optional[str]
        Parametrization used to report the result of the fit.
        Currently only "PyMC" parametrization is supported.
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

    bg_color = "#F2F2F2"  # tkinter background color
    bu_color = "#E2E2E2"  # tkinter button color

    if figsize is None:
        figsize = (8, 6)

    root = create_main(bg_color)  # pylint: disable=assignment-from-no-return
    frame_grid_controls, frame_matplotib, frame_cbuttons = create_frames(root, bg_color)
    canvas, fig, ax_grid, ax_fit = create_figure(frame_matplotib, figsize)

    coll = create_grid(x_min, x_max, nrows, ncols, ax=ax_grid)
    grid = Rectangles(fig, coll, nrows, ncols, ax_grid)

    cvars = create_cbuttons(frame_cbuttons, bg_color, bu_color)
    x_min_entry, x_max_entry, x_bins_entry = create_entries_xrange_grid(
        x_min, x_max, nrows, ncols, grid, ax_grid, ax_fit, frame_grid_controls, canvas, bg_color
    )

    dist_return = {"rv": None}
    fig.canvas.mpl_connect(
        "figure_leave_event",
        lambda event: on_leave_fig(
            event, grid, cvars, x_min_entry, x_max_entry, x_bins_entry, ax_fit, dist_return
        ),
    )

    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    tk.mainloop()

    return dist_return["rv"]


def create_grid(x_min=0, x_max=1, nrows=10, ncols=10, ax=None):
    """
    Create a grid of rectangles
    """
    xx = np.arange(0, ncols)
    yy = np.arange(0, nrows)

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


def on_leave_fig(event, grid, cvars, x_min, x_max, x_bins_entry, ax, dist_return):
    x_min = float(x_min.get())
    x_max = float(x_max.get())
    ncols = float(x_bins_entry.get())
    x_range = x_max - x_min

    sample, filled_columns = weights_to_sample(grid.weights, x_min, x_range, ncols)

    if filled_columns > 1:
        selected_distributions = get_distributions(cvars)

        if selected_distributions:
            reset_dist_panel(x_min, x_max, ax)
            # fitted_dist, params, x_vals, ref_pdf = fit_to_sample(
            #     selected_distributions, sample, x_min, x_max, x_range
            # )
            fitted_dist = fit_to_sample(selected_distributions, sample, x_min, x_max, x_range)

            dist_return["rv"] = fitted_dist

            if fitted_dist is None:
                ax.set_title("domain error")
            else:
                fitted_dist.plot_pdf(box=True, legend="tittle", ax=ax)
                for bound in fitted_dist.rv_frozen.support():
                    if np.isfinite(bound):
                        ax.plot(bound, 0, "ko")
    else:
        reset_dist_panel(x_min, x_max, ax)
    event.canvas.draw()


def update_grid(canvas, x_min, x_max, x_bins_entry, y_bins_entry, grid, ax_grid, ax_fit):
    """
    Update the grid subplot
    """
    x_min = float(x_min.get())
    x_max = float(x_max.get())
    ncols = int(x_bins_entry.get())
    nrows = int(y_bins_entry.get())
    ax_grid.cla()
    coll = create_grid(x_min=x_min, x_max=x_max, nrows=nrows, ncols=ncols, ax=ax_grid)
    grid.coll = coll
    grid.ncols = ncols
    grid.nrows = nrows
    grid.weights = {k: 0 for k in range(0, ncols)}
    reset_dist_panel(x_min, x_max, ax_fit)
    ax_grid.set_yticks([])
    ax_grid.relim()
    ax_grid.autoscale_view()
    canvas.draw()


def reset_dist_panel(x_min, x_max, ax):
    """
    Clean the distribution subplot
    """
    ax.cla()
    ax.set_yticks([])
    ax.set_xlim(x_min, x_max)
    ax.relim()
    ax.autoscale_view()


def select_all(cbuttons):
    """
    select all cbuttons
    """
    for cbutton in cbuttons:
        cbutton.select()


def deselect_all(cbuttons):
    """
    deselect all cbuttons
    """
    for cbutton in cbuttons:
        cbutton.deselect()


def create_cbuttons(frame_cbuttons, bg_color, bu_color):
    """
    Create check buttons to select distributions
    """
    dist_labels = ["Normal", "Beta", "Gamma", "LogNormal"]
    cbuttons = []
    cvars = []
    for text in dist_labels:
        var = tk.StringVar()
        cbutton = tk.Checkbutton(
            frame_cbuttons,
            text=text,
            variable=var,
            onvalue=text,
            offvalue="",
            bg=bg_color,
            highlightthickness=0,
        )
        cbutton.select()
        cbutton.pack(anchor=tk.W)
        cbuttons.append(cbutton)
        cvars.append(var)

    tk.Button(frame_cbuttons, text="all ", command=lambda: select_all(cbuttons), bg=bu_color).pack(
        side=tk.LEFT, padx=5, pady=5
    )
    tk.Button(
        frame_cbuttons, text="none", command=lambda: deselect_all(cbuttons), bg=bu_color
    ).pack(side=tk.LEFT)

    return cvars


def create_entries_xrange_grid(
    x_min, x_max, nrows, ncols, grid, ax_grid, ax_fit, frame_grid_controls, canvas, bg_color
):
    """
    Create text entries to change grid x_range and numbers of bins
    """

    # Set text boxes for x-range
    x_min = tk.DoubleVar(value=x_min)
    x_min_label = tk.Label(frame_grid_controls, text="x_min", bg=bg_color)
    x_min_entry = tk.Entry(frame_grid_controls, textvariable=x_min, width=10, font="None 11")

    x_max = tk.DoubleVar(value=x_max)
    x_max_label = tk.Label(frame_grid_controls, text="x_max", bg=bg_color)
    x_max_entry = tk.Entry(frame_grid_controls, textvariable=x_max, width=10, font="None 11")

    # Set text boxes for grid rows and columns
    x_bins = tk.IntVar(value=nrows)
    x_bins_label = tk.Label(frame_grid_controls, text="x_bins", bg=bg_color)
    x_bins_entry = tk.Entry(frame_grid_controls, textvariable=x_bins, width=10, font="None 11")

    y_bins = tk.IntVar(value=ncols)
    y_bins_label = tk.Label(frame_grid_controls, text="y_bins", bg=bg_color)
    y_bins_entry = tk.Entry(frame_grid_controls, textvariable=y_bins, width=10, font="None 11")

    x_min_entry.bind(
        "<Return>",
        lambda event: update_grid(
            canvas, x_min_entry, x_max_entry, x_bins_entry, y_bins_entry, grid, ax_grid, ax_fit
        ),
    )
    x_min_label.grid(row=0, column=0)
    x_min_entry.grid(row=0, column=1)
    x_max_entry.bind(
        "<Return>",
        lambda event: update_grid(
            canvas, x_min_entry, x_max_entry, x_bins_entry, y_bins_entry, grid, ax_grid, ax_fit
        ),
    )
    x_max_label.grid(row=1, column=0)
    x_max_entry.grid(row=1, column=1)

    spacer = tk.Label(frame_grid_controls, text="", bg=bg_color)
    spacer.grid(row=2, column=0)

    x_bins_entry.bind(
        "<Return>",
        lambda event: update_grid(
            canvas, x_min_entry, x_max_entry, x_bins_entry, y_bins_entry, grid, ax_grid, ax_fit
        ),
    )
    x_bins_label.grid(row=3, column=0)
    x_bins_entry.grid(row=3, column=1)
    y_bins_entry.bind(
        "<Return>",
        lambda event: update_grid(
            canvas, x_min_entry, x_max_entry, x_bins_entry, y_bins_entry, grid, ax_grid, ax_fit
        ),
    )
    y_bins_label.grid(row=4, column=0)
    y_bins_entry.grid(row=4, column=1)

    return x_min_entry, x_max_entry, x_bins_entry


def create_main(bg_color):
    """
    Create main tkinter window
    """
    root = tk.Tk()
    font = tk.font.nametofont("TkDefaultFont")
    font.configure(size=16)
    root.configure(bg=bg_color)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.wm_title("Draw your distribution")


def create_frames(root, bg_color):
    """
    Create tkinter frames.
    One for the grid controls, one for the matplotlib plot and one for the checkbuttons
    """
    frame_grid_controls = tk.Frame(root, bg=bg_color)
    frame_matplotib = tk.Frame(root)
    frame_cbuttons = tk.Frame(root, bg=bg_color)
    frame_grid_controls.grid(row=0, column=1, padx=15, pady=15)
    frame_matplotib.grid(row=0, rowspan=2, column=0, sticky="we", padx=25, pady=15)
    frame_cbuttons.grid(row=3, sticky="w", padx=25, pady=15)

    return frame_grid_controls, frame_matplotib, frame_cbuttons


def create_figure(frame_matplotib, figsize):
    """
    Initialize a matplotlib figure with two subplots
    """
    fig = Figure(figsize=figsize, constrained_layout=True)
    ax_grid = fig.add_subplot(211)
    ax_fit = fig.add_subplot(212)
    ax_fit.set_yticks([])

    canvas = FigureCanvasTkAgg(fig, master=frame_matplotib)
    canvas.draw()
    return canvas, fig, ax_grid, ax_fit
