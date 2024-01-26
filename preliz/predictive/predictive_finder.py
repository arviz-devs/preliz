"""Prior predictive finder."""

import logging

import numpy as np

try:
    import ipywidgets as widgets
except ImportError:
    pass

from ..internal.plot_helper import create_figure, check_inside_notebook, plot_repr, reset_dist_panel
from ..internal.parser import get_prior_pp_samples, from_bambi, from_preliz
from ..internal.predictive_helper import back_fitting, select_prior_samples

_log = logging.getLogger("preliz")


def predictive_finder(
    fmodel, target, draws=100, steps=5, references=None, engine="preliz", figsize=None
):
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
        Number of draws from the prior and prior predictive distribution. Defaults to 100
    step : int
        Number of steps to find the best match. Each step will use the previous match as
        initial guess. If your initial prior predictive distribution is far from the target
        distribution you may need to increase the number of steps. Alternatively, you can
        click on the figure or press the `carry on` button many times.
    references : int, float, list, tuple or dictionary
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme. Use a dictionary for labeled references.
    engine : str
        Library used to define the model. Either `preliz` or `bambi`. Defaults to `preliz`.
    figsize : tuple
        Figure size. If None, the default is (8, 6).
    """

    _log.info("This is an experimental method under development, use with caution.")

    check_inside_notebook(need_widget=True)

    if figsize is None:
        figsize = (8, 6)

    fig, ax_fit = create_figure(figsize)

    button_carry_on, button_return_prior, w_repr = get_widgets()

    match_distribution = MatchDistribution(fig, fmodel, target, draws, steps, engine, ax_fit)

    plot_pp_samples(
        match_distribution.pp_samples, draws, target, w_repr.value, references, fig, ax_fit
    )
    fig.suptitle(
        "This is your target distribution\n and a sample from the prior predictive distribution"
    )

    output = widgets.Output()

    with output:

        def kind_(_):
            kind = w_repr.value
            plot_pp_samples(
                match_distribution.pp_samples, draws, target, kind, references, fig, ax_fit
            )

        w_repr.observe(kind_, names=["value"])

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
        button_carry_on.on_click(lambda event: match_distribution(w_repr.value))

        fig.canvas.mpl_connect(
            "button_release_event", lambda event: match_distribution(w_repr.value)
        )

    controls = widgets.VBox([button_carry_on, button_return_prior])

    display(widgets.HBox([controls, w_repr, output]))  # pylint:disable=undefined-variable


class MatchDistribution:  # pylint:disable=too-many-instance-attributes
    def __init__(self, fig, fmodel, target, draws, steps, engine, ax):
        self.fig = fig
        self.fmodel = fmodel
        self.target = target
        self.target_octiles = self.target.ppf([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        self.draws = draws
        self.steps = steps
        self.engine = engine
        self.ax = ax
        self.values = None
        self.string = None

        if self.engine == "preliz":
            self.variables, self.model = from_preliz(self.fmodel)
        elif self.engine == "bambi":
            self.fmodel, self.variables, self.model = from_bambi(self.fmodel, self.draws)

        self.pp_samples, _ = get_prior_pp_samples(
            self.fmodel, self.variables, self.draws, self.engine
        )

    def __call__(self, kind_plot):
        self.fig.texts = []

        for _ in range(self.steps):
            pp_samples, prior_samples = get_prior_pp_samples(
                self.fmodel, self.variables, self.draws, self.engine, self.values
            )
            values_to_fit = select(
                prior_samples, pp_samples, self.draws, self.target_octiles, self.model
            )
            self.string, self.values = back_fitting(self.model, values_to_fit, new_families=False)

        if self.engine == "preliz":
            self.pp_samples = [self.fmodel(*self.values)[-1] for _ in range(self.draws)]
        elif self.engine == "bambi":
            self.pp_samples = self.fmodel(*self.values)[-1]

        reset_dist_panel(self.ax, True)
        plot_repr(self.pp_samples, kind_plot, None, self.draws, self.ax)

        if kind_plot == "ecdf":
            self.target.plot_cdf(color="C0", legend=False, ax=self.ax)

        if kind_plot in ["kde", "hist"]:
            self.target.plot_pdf(color="C0", legend=False, ax=self.ax)

        self.fig.canvas.draw()


def select(prior_sample, pp_sample, draws, target_octiles, model):
    pp_octiles = np.stack(
        [np.quantile(sample, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]) for sample in pp_sample]
    )
    pp_octiles_min = pp_octiles.min()
    pp_octiles_max = pp_octiles.max()
    target_octiles_min = target_octiles.min()
    target_octiles_max = target_octiles.max()
    # target and pp_samples are not overlapping
    if pp_octiles_max < target_octiles_min or pp_octiles_min > target_octiles_max:
        prior_sample = {key: value**2 for key, value in prior_sample.items()}
        selected = range(draws)
    # target is wider than pp_samples
    elif pp_octiles_max < target_octiles_max and pp_octiles_min > target_octiles_min:
        factor = (target_octiles_max - target_octiles_min) / (pp_octiles_max - pp_octiles_min)
        prior_sample = {key: value * factor for key, value in prior_sample.items()}
        selected = range(draws)
    else:
        w_un = 1 / (np.mean((target_octiles - pp_octiles) ** 2, 1) ** 0.5)
        selected = systematic(w_un / w_un.sum())

    values_to_fit = select_prior_samples(selected, prior_sample, model)

    return values_to_fit


def plot_pp_samples(pp_samples, draws, target, kind_plot, references, fig, ax):

    reset_dist_panel(ax, True)
    plot_repr(pp_samples, kind_plot, references, draws, ax)

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


def systematic(normalized_weights):
    """
    Systematic resampling.

    Return indices in the range 0, ..., len(normalized_weights)

    Note: adapted from https://github.com/nchopin/particles
    """
    lnw = len(normalized_weights)
    single_uniform = (np.random.random() + np.arange(lnw)) / lnw
    return inverse_cdf(single_uniform, normalized_weights)


def inverse_cdf(single_uniform, normalized_weights):
    """
    Inverse CDF algorithm for a finite distribution.

    Parameters
    ----------
    single_uniform: npt.NDArray[np.float_]
        Ordered points in [0,1]

    normalized_weights: npt.NDArray[np.float_])
        Normalized weights

    Returns
    -------
    new_indices: ndarray
        a vector of indices in range 0, ..., len(normalized_weights)

    Note: adapted from https://github.com/nchopin/particles
    """
    idx = 0
    a_weight = normalized_weights[0]
    sul = len(single_uniform)
    new_indices = np.empty(sul, dtype=np.int64)
    for i in range(sul):
        while single_uniform[i] > a_weight:
            idx += 1
            a_weight += normalized_weights[idx]
        new_indices[i] = idx
    return new_indices
