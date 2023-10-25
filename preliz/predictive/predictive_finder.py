"""Prior predictive finder."""

import logging

import numpy as np

try:
    import ipywidgets as widgets
except ImportError:
    pass


from ..internal.plot_helper import create_figure, check_inside_notebook, plot_repr, reset_dist_panel
from ..internal.parser import inspect_source, parse_function_for_ppa, get_prior_pp_samples
from ..internal.predictive_helper import back_fitting, select_prior_samples

_log = logging.getLogger("preliz")


def predictive_finder(fmodel, target, draws=100, steps=5, figsize=None):
    """
    Prior predictive finder.

    This is an experimental method under development, use with caution.

    Parameters
    ----------
    fmodel : PreliZ model
        The model should return a list of random variables. The last one should be the
        observed random variable. The rest should be prior random variables.
    target : PreliZ distribution
        Target distribution. This is the expected distribution that the prior predictive
        distribution should match. This distribution should match your previous knowledge
        about the observed random variable. To obtain a target distribution you can use
        other function from Preliz including `roulette`, `quartile_int`, `maxent`, etc.
    draws : int
        Number of draws from the prior and prior predictive distribution
    step : int
        Number of steps to find the best match. Each step will use the previous match as
        initial guess. If your initial prior predictive distribution is far from the target
        distribution you may need to increase the number of steps. Alternatively, you can
        click on the figure or press the `carry on` button many times.
    figsize : tuple
        Figure size. If None, the default is (8, 6).
    """

    _log.info("This is an experimental method under development, use with caution.")

    check_inside_notebook(need_widget=True)

    if figsize is None:
        figsize = (8, 6)

    fig, ax_fit = create_figure(figsize)

    button_carry_on, button_return_prior, w_repr = get_widgets()

    pp_samples, _, obs_rv = get_prior_pp_samples(fmodel, draws)

    source, _ = inspect_source(fmodel)
    model = parse_function_for_ppa(source, obs_rv)

    plot_pp_samples(pp_samples, draws, target, w_repr.value, fig, ax_fit)
    fig.suptitle(
        "This is your target distribution\n and a sample from the prior predictive distribution"
    )

    output = widgets.Output()

    with output:

        def kind_(_):
            kind = w_repr.value
            plot_pp_samples(pp_samples, draws, target, kind, fig, ax_fit)

        w_repr.observe(kind_, names=["value"])

        match_distribution = MatchDistribution(
            fig, w_repr.value, fmodel, model, target, draws, steps, ax_fit
        )

        def on_return_prior_(_):
            on_return_prior(fig, ax_fit)

        def on_return_prior(fig, ax_fit):
            string = match_distribution.string
            if string is None:
                string = (
                    "Please, press 'carry on' or click on the figure at least once "
                    "before pressing 'return prior'"
                )
                fig.suptitle(string)
            else:
                ax_fit.cla()
                ax_fit.set_xticks([])
                ax_fit.set_yticks([])
                fig.text(0.05, 0.5, string, fontsize=14)
            fig.canvas.draw()

        button_return_prior.on_click(on_return_prior_)
        button_carry_on.on_click(lambda event: match_distribution())

        fig.canvas.mpl_connect("button_release_event", lambda event: match_distribution())

    controls = widgets.VBox([button_carry_on, button_return_prior])

    display(widgets.HBox([controls, w_repr]))  # pylint:disable=undefined-variable


class MatchDistribution:  # pylint:disable=too-many-instance-attributes
    def __init__(self, fig, kind_plot, fmodel, model, target, draws, steps, ax):
        self.fig = fig
        self.kind_plot = kind_plot
        self.fmodel = fmodel
        self.model = model
        self.target = target
        self.target_octiles = target.ppf([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        self.draws = draws
        self.steps = steps
        self.ax = ax

        self.pp_samples = None
        self.values = None
        self.string = None

    def __call__(self):
        self.fig.texts = []

        for _ in range(self.steps):
            pp_samples, prior_samples, _ = get_prior_pp_samples(
                self.fmodel, self.draws, self.values
            )
            values_to_fit = select(
                prior_samples, pp_samples, self.draws, self.target_octiles, self.model
            )
            self.string, self.values = back_fitting(self.model, values_to_fit, new_families=False)

        self.pp_samples = [self.fmodel(*self.values)[-1] for _ in range(self.draws)]

        reset_dist_panel(self.ax, True)
        plot_repr(self.pp_samples, self.kind_plot, self.draws, self.ax)

        if self.kind_plot == "ecdf":
            self.target.plot_cdf(color="C0", legend=False, ax=self.ax)

        if self.kind_plot in ["kde", "hist"]:
            self.target.plot_pdf(color="C0", legend=False, ax=self.ax)

        self.fig.canvas.draw()


def select(prior_sample, pp_sample, draws, target, model):
    quants = np.stack(
        [np.quantile(sample, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]) for sample in pp_sample]
    )
    w_un = 1 / (np.mean((target - quants) ** 2, 1) ** 0.5)
    selected = np.random.choice(range(0, draws), p=w_un / w_un.sum(), size=draws, replace=True)

    values_to_fit = select_prior_samples(selected, prior_sample, model)

    return values_to_fit


def plot_pp_samples(pp_samples, draws, target, kind_plot, fig, ax):

    reset_dist_panel(ax, True)
    plot_repr(pp_samples, kind_plot, draws, ax)

    if kind_plot == "ecdf":
        target.plot_cdf(color="C0", legend=False, ax=ax)

    if kind_plot in ["kde", "hist"]:
        target.plot_pdf(color="C0", legend=False, ax=ax)

    fig.canvas.draw()


def get_widgets():

    width_repr_text = widgets.Layout(width="250px")

    button_carry_on = widgets.Button(description="carry on")
    button_return_prior = widgets.Button(description="return prior")

    w_repr = widgets.RadioButtons(
        options=["ecdf", "kde", "hist"],
        value="ecdf",
        description="",
        disabled=False,
        layout=width_repr_text,
    )

    return button_carry_on, button_return_prior, w_repr
