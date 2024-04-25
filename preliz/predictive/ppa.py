"""Prior predictive check assistant."""

import logging
import ast
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
from ..internal.parser import get_prior_pp_samples, from_preliz, from_bambi
from ..internal.predictive_helper import back_fitting, select_prior_samples
from ..distributions import Normal
from ..distributions.distributions import Distribution

_log = logging.getLogger("preliz")


def ppa(
    fmodel, draws=2000, references=0, boundaries=(-np.inf, np.inf), target=None, engine="preliz"
):
    """
    Prior predictive check assistant.

    This is an experimental method under development, use with caution.

    Parameters
    ----------
    model : PreliZ model
    draws : int
        Number of draws from the prior and prior predictive distribution
    references : int, float, list, tuple or dictionary
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme. Use a dictionary for labeled references.
    boundaries : tuple
        Hard boundaries (lower, upper). Posterior predictive samples with values outside these
        boundaries will be excluded from the analysis.
    target : tuple or PreliZ distribtuion
        Target distribution. The first shown distributions will be selected to be as close
        as possible to `target`. Available options are, a PreliZ distribution or a 2-tuple with
        the first element representing the mean and the second the standard deviation.
    engine : str
        Library used to define the model. Either `preliz` or `bambi`. Defaults to `preliz`
    """
    check_inside_notebook(need_widget=True)

    _log.info(""""This is an experimental method under development, use with caution.""")

    filter_dists = FilterDistribution(fmodel, draws, references, boundaries, target, engine)
    filter_dists()

    output = widgets.Output()

    with output:
        references_widget = widgets.Text(
            value=str(references),
            placeholder="Int, Float or tuple",
            description="references: ",
            disabled=False,
            layout=widgets.Layout(width="230px", margin="0 20px 0 0"),
        )
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

        button_return_prior.on_click(lambda event: filter_dists.on_return_prior())

        def kind_(_):
            kind = radio_buttons_kind.value
            plot_pp_samples(
                filter_dists.pp_samples,
                filter_dists.display_pp_idxs,
                ast.literal_eval(references_widget.value),
                kind,
                check_button_sharex.value,
                filter_dists.fig,
            )

        references_widget.observe(kind_, names=["value"])

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

        filter_dists.fig.canvas.mpl_connect("button_press_event", click)

    controls = widgets.VBox([button_carry_on, button_return_prior])
    plot_combine = widgets.VBox([radio_buttons_kind, check_button_sharex])

    display(  # pylint:disable=undefined-variable
        widgets.HBox([references_widget, plot_combine, controls, output])
    )


class FilterDistribution:  # pylint:disable=too-many-instance-attributes
    def __init__(self, fmodel, draws, references, boundaries, target, engine):
        self.fmodel = fmodel
        self.source = ""  # string representation of the model
        self.draws = draws
        self.references = references
        self.boundaries = boundaries
        self.target = target
        self.engine = engine
        self.pp_samples = None  # prior predictive samples
        self.prior_samples = None  # prior samples used for backfitting
        self.display_pp_idxs = None  # indices of the pp_samples to be displayed
        self.pp_octiles = None  # octiles computed from pp_samples
        self.kdt = None  # KDTree used to find similar distributions
        self.model = None  # parsed model used for backfitting
        self.clicked = []  # axes clicked by the user
        self.choices = []  # indices of the pp_samples selected by the user and not yet used to
        # find new distributions, this list can increase or decrease in size
        self.selected = set()  # indices of all the pp_samples selected by the user + machine
        # this set can only increase in size
        self.distances = {}  # distances between as selected distribution and its nearest neighbor
        self.shown = set()  # keep track to avoid showing the same distribution twice.
        self.fig = None  # figure used to display the pp_samples
        self.fig_pp_mean = None  # figure used to display the mean of the pp_samples
        self.axes = None  # axes used to display the pp_samples

    def __call__(self):

        if self.engine == "preliz":
            variables, self.model = from_preliz(self.fmodel)
        elif self.engine == "bambi":
            self.fmodel, variables, self.model = from_bambi(self.fmodel, self.draws)

        self.pp_samples, self.prior_samples = get_prior_pp_samples(
            self.fmodel, variables, self.draws, self.engine
        )

        if self.target is not None:
            self.add_target_dist()

        self.pp_octiles, self.kdt = self.compute_octiles()
        self.display_pp_idxs = self.initialize_subsamples(self.target)
        self.fig, self.axes = plot_pp_samples(
            self.pp_samples, self.display_pp_idxs, self.references
        )
        self.fig_pp_mean = plot_pp_mean(self.pp_samples, self.selected, self.references)

    def add_target_dist(self):
        if isinstance(self.target, tuple):
            ref_sample = Normal(*self.target).rvs(self.pp_samples.shape[1])
        elif isinstance(self.target, Distribution):
            ref_sample = self.target.rvs(self.pp_samples.shape[1])

        self.pp_samples = np.vstack([ref_sample, self.pp_samples])

    def compute_octiles(self):
        """
        Compute the octiles for the prior predictive samples. This is used to find
        similar distributions using a KDTree.

        We have empirically found that octiles are a good choice, but this could
        be the consequence of limited testing.
        """
        pp_octiles = np.quantile(
            self.pp_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1
        ).T
        kdt = KDTree(pp_octiles)
        return pp_octiles, kdt

    def initialize_subsamples(self, target):
        """
        Initialize the subsamples to be displayed.

        If `target` is None, we search for the farthest_neighbor (this increases diversity)
        otherwise we search for the nearest_neighbor of target

        The initialization takes into account the boundaries provided by the user.

        Updates the `shown` set (already shown pp_samples) and if `target` is None
        also updates the  `distances` dictionary.
        """
        pp_idxs_to_display = []

        shown_list = []
        for idx, sample in enumerate(self.pp_samples):
            if np.min(sample) < self.boundaries[0] or np.max(sample) > self.boundaries[1]:
                shown_list.append(idx)

        self.shown.update(shown_list)

        # If we have not seen all the samples yet, we collect more
        if len(self.shown) != self.draws:
            if target is None:
                new = np.random.choice(list(set(range(0, len(self.pp_octiles))) - self.shown))
                pp_idxs_to_display.append(new)

                for _ in range(8):
                    farthest_neighbor = self.draws
                    while new in pp_idxs_to_display or new in self.shown:
                        _, new = self.kdt.query(
                            self.pp_octiles[pp_idxs_to_display[-1]], [farthest_neighbor], workers=-1
                        )
                        new = new.item()
                        farthest_neighbor -= 1
                    # Missing neighbors are indicated with index==sample_size
                    if new != self.draws:
                        pp_idxs_to_display.append(new)
            else:
                new = 0
                pp_idxs_to_display.append(new)

                for _ in range(9):
                    nearest_neighbor = 2
                    while new in pp_idxs_to_display:
                        distance, new = self.kdt.query(
                            self.pp_octiles[pp_idxs_to_display[-1]], [nearest_neighbor], workers=-1
                        )
                        new = new.item()
                        nearest_neighbor += 1

                    if new != self.draws:
                        pp_idxs_to_display.append(new)
                        self.distances[new] = distance.item()

                pp_idxs_to_display = pp_idxs_to_display[1:]

            self.shown.update(pp_idxs_to_display)

        return pp_idxs_to_display

    def keep_sampling(self):
        """
        Find distribution similar to the ones in `choices`, but not already shown.
        If `choices` is empty return an empty selection.
        """
        if self.choices:
            new = self.choices.pop(0)
            samples = [new]

            for _ in range(9):
                nearest_neighbor = 2
                while new in samples or new in self.shown:
                    distance, new = self.kdt.query(
                        self.pp_octiles[samples[-1]], [nearest_neighbor], workers=-1
                    )
                    new = new.item()
                    nearest_neighbor += 1

                # Missing neighbors are indicated with index==self.draws
                if new != self.draws:
                    self.distances[new] = distance.item()
                    samples.append(new)

            self.shown.update(samples[1:])

            return samples[1:]
        else:
            return []

    def collect_more_samples(self):
        """
        Automatically extend the set of user selected distributions

        If the user has selected at least two distributions we automatically extend the selection
        by adding all the distributions that are close to the selected ones. To do so we use
        compute a max_dist, which is the trimmed mean of the distances between the already selected
        distributions. The trimmed mean is computed by discarding the lower and upper 10%.
        """
        selected_distances = np.array([v for k, v in self.distances.items() if k in self.selected])

        if len(selected_distances) > 2:
            q_r = np.quantile(selected_distances, [0.1, 0.9])
            max_dist = np.mean(
                selected_distances[(selected_distances > q_r[0]) & (selected_distances < q_r[1])]
            )
            upper = self.draws
        else:
            max_dist = np.inf
            upper = 3

        _, new = self.kdt.query(
            self.pp_octiles[list(self.selected)],
            range(2, upper),
            distance_upper_bound=max_dist,
            workers=-1,
        )
        new = new[new < self.draws].tolist()

        if np.any(np.isfinite(self.boundaries)):
            new_ = []
            for n_s in new:
                sample = self.pp_samples[n_s]
                if np.min(sample) > self.boundaries[0] and np.max(sample) < self.boundaries[1]:
                    new_.append(n_s)
            new = new_

        if new:
            self.selected.update(new)
            self.shown.update(new)

    def carry_on(self, kind, sharex):
        """
        Collect user input and update the plot.
        """
        self.fig.suptitle("")

        if self.clicked:
            self.choices.extend([int(ax.get_title()) for ax in self.clicked])
            shuffle(self.choices)
            self.selected.update(self.choices)
            self.collect_more_samples()

            for ax in self.clicked:
                plt.setp(ax.spines.values(), color="k", lw=1)
            for ax in self.axes:
                ax.cla()
            for ax in list(self.clicked):
                self.clicked.remove(ax)

        self.display_pp_idxs = self.keep_sampling()

        # if there is nothing to show initialize a new set of samples
        if not self.display_pp_idxs:
            self.display_pp_idxs = self.initialize_subsamples(None)

        plot_pp_mean(self.pp_samples, list(self.selected), self.references, kind, self.fig_pp_mean)

        if self.display_pp_idxs:
            plot_pp_samples(
                self.pp_samples, self.display_pp_idxs, self.references, kind, sharex, self.fig
            )
        else:
            # Instead of showing this message, we should resample.
            self.fig.clf()
            self.fig.suptitle("We have seen all the samples", fontsize=16)
            self.fig.canvas.draw()

    def on_return_prior(self):

        selected = list(self.selected)

        if len(selected) > 4:
            subsample = select_prior_samples(selected, self.prior_samples, self.model)

            string, _ = back_fitting(self.model, subsample, new_families=False)

            self.fig.clf()
            plt.text(0.05, 0.5, string, fontsize=14)

            plt.yticks([])
            plt.xticks([])
        else:
            self.fig.suptitle("Please select more distributions", fontsize=16)

        self.fig.canvas.draw()
