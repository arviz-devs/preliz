"""Prior predictive check assistant."""

import logging
from random import shuffle

try:
    import ipywidgets as widgets
except ImportError:
    pass
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


from ..internal.plot_helper import (
    check_inside_notebook,
    plot_pp_samples,
    plot_pp_mean,
)
from ..internal.parser import inspect_source, parse_function_for_ppa, get_prior_pp_samples
from ..internal.predictive_helper import back_fitting, select_prior_samples
from ..distributions.continuous import Normal
from ..distributions.distributions import Distribution

_log = logging.getLogger("preliz")


def ppa(fmodel, draws=2000, references=0, boundaries=(-np.inf, np.inf), target=None):
    """
    Prior predictive check assistant.

    This is an experimental method under development, use with caution.

    Parameters
    ----------
    model : PreliZ model
    draws : int
        Number of draws from the prior and prior predictive distribution
    references : int, float, list or tuple
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme.
    boundaries : tuple
        Hard boundaries (lower, upper). Posterior predictive samples with values outside these
        boundaries will be excluded from the analysis.
    target : tuple or PreliZ distribtuion
        Target distribution. The first shown distributions will be selected to be as close
        as possible to `target`. Available options are, a PreliZ distribution or a 2-tuple with
        the first element representing the mean and the second the standard deviation.
    """
    check_inside_notebook(need_widget=True)

    _log.info(""""This is an experimental method under development, use with caution.""")

    if isinstance(references, (float, int)):
        references = [references]

    filter_dists = FilterDistribution(fmodel, draws, references, boundaries, target)
    filter_dists()

    output = widgets.Output()

    with output:
        button_carry_on = widgets.Button(description="carry on")
        button_return_prior = widgets.Button(description="return prior")
        radio_buttons_kind = widgets.RadioButtons(
            options=["pdf", "hist", "ecdf"],
            value="pdf",
            description=" ",
            disabled=False,
        )

        check_button_sharex = widgets.Checkbox(
            value=True, description="sharex", disabled=False, indent=False
        )

        button_carry_on.on_click(
            lambda event: filter_dists.carry_on(radio_buttons_kind.value, check_button_sharex.value)
        )

        def on_return_prior_(_):
            on_return_prior(
                filter_dists.fig,
                filter_dists.selected,
                filter_dists.model,
                filter_dists.prior_samples,
            )

        button_return_prior.on_click(on_return_prior_)

        def kind_(_):
            kind = radio_buttons_kind.value

            plot_pp_samples(
                filter_dists.pp_samples,
                filter_dists.pp_samples_idxs,
                references,
                kind,
                check_button_sharex.value,
                filter_dists.fig,
            )

        radio_buttons_kind.observe(kind_, names=["value"])

        check_button_sharex.observe(kind_, names=["value"])

        def click(event):
            if event.inaxes is not None:
                if event.inaxes not in filter_dists.clicked:
                    filter_dists.clicked.append(event.inaxes)
                else:
                    filter_dists.clicked.remove(event.inaxes)
                    plt.setp(event.inaxes.spines.values(), color="k", lw=1)

                for ax in filter_dists.clicked:
                    plt.setp(ax.spines.values(), color="C1", lw=3)
                filter_dists.fig.canvas.draw()

        def on_return_prior(fig, selected, model, prior_samples):

            selected = list(selected)

            if len(selected) > 4:
                subsample = select_prior_samples(selected, prior_samples, model)

                string, _ = back_fitting(model, subsample, new_families=False)

                fig.clf()
                plt.text(0.05, 0.5, string, fontsize=14)

                plt.yticks([])
                plt.xticks([])
            else:
                fig.suptitle("Please select more distributions", fontsize=16)

            fig.canvas.draw()

        filter_dists.fig.canvas.mpl_connect("button_press_event", click)

    controls = widgets.VBox([button_carry_on, button_return_prior])

    display(  # pylint:disable=undefined-variable
        widgets.HBox([controls, radio_buttons_kind, check_button_sharex, output])
    )


class FilterDistribution:  # pylint:disable=too-many-instance-attributes
    def __init__(self, fmodel, draws, references, boundaries, target):
        self.fmodel = fmodel
        self.source, _ = inspect_source(fmodel)
        self.draws = draws
        self.references = references
        self.boundaries = boundaries
        self.target = target
        self.pp_samples = None
        self.prior_samples = None
        self.pp_samples_idxs = None
        self.pp_summary = None
        self.obs_rv = None
        self.fig = None
        self.choices = []
        self.clicked = []
        self.selected = set()
        self.collected_distances = {}
        self.model = None
        self.shown = None
        self.distances = None
        self.fig_pp_mean = None
        self.axes = None
        self.kdt = None

    def __call__(self):
        self.pp_samples, self.prior_samples, self.obs_rv = get_prior_pp_samples(
            self.fmodel, self.draws
        )

        self.model = parse_function_for_ppa(self.source, self.obs_rv)

        if self.target is not None:
            self.add_target_dist()

        self.pp_summary, self.kdt = self.compute_summaries()
        self.pp_samples_idxs, self.distances, self.shown = self.initialize_subsamples(self.target)
        self.fig, self.axes = plot_pp_samples(
            self.pp_samples, self.pp_samples_idxs, self.references
        )
        self.fig_pp_mean = plot_pp_mean(self.pp_samples, self.selected, self.references)

    def add_target_dist(self):
        if isinstance(self.target, tuple):
            ref_sample = Normal(*self.target).rvs(self.pp_samples.shape[1])
        elif isinstance(self.target, Distribution):
            ref_sample = self.target.rvs(self.pp_samples.shape[1])

        self.pp_samples = np.vstack([ref_sample, self.pp_samples])

    def compute_summaries(self):
        pp_summary = np.quantile(
            self.pp_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1
        ).T
        kdt = KDTree(pp_summary)
        return pp_summary, kdt

    def initialize_subsamples(self, target):
        samples = []
        distances = {}
        shown = []
        for idx, sample in enumerate(self.pp_samples):
            if np.min(sample) < self.boundaries[0] or np.max(sample) > self.boundaries[1]:
                shown.append(idx)

        shown = set(shown)

        if len(shown) != self.draws:
            if target is None:
                new = np.random.choice(list(set(range(0, len(self.pp_summary))) - shown))
                samples.append(new)

                for _ in range(8):
                    farthest_neighbor = self.draws
                    while new in samples or new in shown:
                        # we search for the farthest_neighbor
                        _, new = self.kdt.query(
                            self.pp_summary[samples[-1]], [farthest_neighbor], workers=-1
                        )
                        new = new.item()
                        farthest_neighbor -= 1
                    # Missing neighbors are indicated with index==sample_size
                    if new != self.draws:
                        samples.append(new)
            else:
                new = 0
                samples.append(new)

                for _ in range(9):
                    nearest_neighbor = 2
                    while new in samples:
                        distance, new = self.kdt.query(
                            self.pp_summary[samples[-1]], [nearest_neighbor], workers=-1
                        )
                        new = new.item()
                        nearest_neighbor += 1

                    if new != self.draws:
                        samples.append(new)
                        distances[new] = distance.item()

                samples = samples[1:]

            shown.update(samples)

        return samples, distances, shown

    def carry_on(self, kind, sharex):
        self.fig.suptitle("")

        if self.clicked:
            self.choices.extend([int(ax.get_title()) for ax in self.clicked])
            shuffle(self.choices)
            self.selected.update(self.choices)
            self.selected, self.shown = collect_more_samples(
                self.selected,
                self.collected_distances,
                self.shown,
                self.pp_summary,
                self.pp_samples,
                self.draws,
                self.boundaries,
                self.kdt,
            )

            for ax in self.clicked:
                plt.setp(ax.spines.values(), color="k", lw=1)
            for ax in self.axes:
                ax.cla()
            for ax in list(self.clicked):
                self.clicked.remove(ax)

        self.pp_samples_idxs, self.distances, self.shown = keep_sampling(
            self.pp_summary, self.choices, self.shown, self.draws, self.kdt
        )

        if not self.pp_samples_idxs:
            self.pp_samples_idxs, self.distances, self.shown = self.initialize_subsamples(None)

        self.collected_distances.update(self.distances)
        plot_pp_mean(self.pp_samples, list(self.selected), self.references, kind, self.fig_pp_mean)

        if self.pp_samples_idxs:
            plot_pp_samples(
                self.pp_samples, self.pp_samples_idxs, self.references, kind, sharex, self.fig
            )
        else:
            # Instead of showing this message, we should resample.
            self.fig.clf()
            self.fig.suptitle("We have seen all the samples", fontsize=16)
            self.fig.canvas.draw()


def keep_sampling(pp_summary, choices, shown, sample_size, kdt):
    """
    Find distribution similar to the ones in `choices`, but not already shown.
    If `choices` is empty return an empty selection.
    """
    if choices:
        new = choices.pop(0)
        samples = [new]
        distances = {}

        for _ in range(9):
            nearest_neighbor = 2
            while new in samples or new in shown:
                distance, new = kdt.query(pp_summary[samples[-1]], [nearest_neighbor], workers=-1)
                new = new.item()
                nearest_neighbor += 1

            # Missing neighbors are indicated with index==sample_size
            if new != sample_size:
                distances[new] = distance.item()
                samples.append(new)

        shown.update(samples[1:])

        return samples[1:], distances, shown
    else:
        return [], [], shown


def collect_more_samples(
    selected, collected_distances, shown, pp_summary, pp_samples, sample_size, boundaries, kdt
):
    """
    Automatically extend the user selected distributions

    Lot of room for improving this function
    """
    selected_distances = np.array([v for k, v in collected_distances.items() if k in selected])

    if len(selected_distances) > 2:
        q_r = np.quantile(selected_distances, [0.1, 0.9])
        max_dist = np.mean(
            selected_distances[(selected_distances > q_r[0]) & (selected_distances < q_r[1])]
        )
        upper = sample_size
    else:
        max_dist = np.inf
        upper = 3

    _, new = kdt.query(
        pp_summary[list(selected)], range(2, upper), distance_upper_bound=max_dist, workers=-1
    )
    new = new[new < sample_size].tolist()

    if np.any(np.isfinite(boundaries)):
        new_ = []
        for n_s in new:
            sample = pp_samples[n_s]
            if np.min(sample) > boundaries[0] and np.max(sample) < boundaries[1]:
                new_.append(n_s)
        new = new_

    if new:
        selected.update(new)
        shown.update(new)
        return selected, shown
    else:
        return selected, shown
