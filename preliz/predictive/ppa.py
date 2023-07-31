"""Prior predictive check assistant."""

import logging
from random import shuffle
from sys import modules

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
    repr_to_matplotlib,
)
from ..internal.parser import inspect_source, parse_function_for_ppa, get_prior_pp_samples
from ..distributions.continuous import Normal
from ..distributions.distributions import Distribution
from ..unidimensional import mle

_log = logging.getLogger("preliz")


def ppa(
    fmodel, draws=2000, summary="octiles", references=0, boundaries=(-np.inf, np.inf), init=None
):
    """
    Prior predictive check assistant.

    This is experimental

    Parameters
    ----------
    model : PreliZ model
        Model associated to ``idata``.
    draws : int
        Number of draws from the prior and prior predictive distribution
    summary : str
        Summary statistics applied to prior samples in order to define (dis)similarity
        of distributions. Current options are `octiles`, `hexiles`, `quantiles`,
        `sort` (sort data) `octi_sum` (robust estimation of first 4 moments from octiles).
    references : int, float, list or tuple
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme.
    boundaries : tuple
        Hard boundaries (lower, upper). Posterior predictive samples with values outside these
        boundaries will be excluded from the analysis.
    init : tuple or PreliZ distribtuion
        Initial distribution. The first shown distributions will be selected to be as close
        as possible to `init`. Available options are, a PreliZ distribution or a 2-tuple with
        the first element representing the mean and the second the standard deviation.
    """
    check_inside_notebook(need_widget=True)

    _log.info(""""This is an experimental method under development, use with caution.""")

    if isinstance(references, (float, int)):
        references = [references]

    source, _ = inspect_source(fmodel)

    pp_samples, prior_samples, obs_rv = get_prior_pp_samples(fmodel, draws)

    sample_size = pp_samples.shape[0]
    model = parse_function_for_ppa(source, obs_rv)

    if init is not None:
        pp_samples = add_init_dist(init, pp_samples)

    shown = []
    for idx, sample in enumerate(pp_samples):
        if np.min(sample) < boundaries[0] or np.max(sample) > boundaries[1]:
            shown.append(idx)

    shown = set(shown)

    choices = []
    clicked = []
    selected = set()
    collected_distances = {}

    pp_summary, kdt = compute_summaries(pp_samples, summary)
    pp_samples_idxs, _, shown = initialize_subsamples(pp_summary, shown, sample_size, kdt, init)
    fig, axes = plot_pp_samples(pp_samples, pp_samples_idxs, references)
    fig_pp_mean = plot_pp_mean(pp_samples, selected, references)

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

        def carry_on_(_):
            carry_on(
                fig,
                axes,
                fig_pp_mean,
                radio_buttons_kind.value,
                check_button_sharex.value,
                references,
                clicked,
                pp_samples,
                pp_summary,
                choices,
                selected,
                collected_distances,
                shown,
                sample_size,
                boundaries,
                kdt,
            )

        button_carry_on.on_click(carry_on_)

        def on_return_prior_(_):
            on_return_prior(fig, selected, model)

        button_return_prior.on_click(on_return_prior_)

        def kind_(_):
            kind = radio_buttons_kind.value

            plot_pp_samples(
                pp_samples,
                pp_samples_idxs,
                references,
                kind,
                check_button_sharex.value,
                fig,
            )

            # plot_pp_mean(
            #     pp_samples,
            #     selected,
            #     references,
            #     kind,
            #     fig_pp_mean,
            # )

        radio_buttons_kind.observe(kind_, names=["value"])

        check_button_sharex.observe(kind_, names=["value"])

        def click(event):
            if event.inaxes is not None:
                if event.inaxes not in clicked:
                    clicked.append(event.inaxes)
                else:
                    clicked.remove(event.inaxes)
                    plt.setp(event.inaxes.spines.values(), color="k", lw=1)

                for ax in clicked:
                    plt.setp(ax.spines.values(), color="C1", lw=3)
                fig.canvas.draw()

        def on_return_prior(fig, selected, model):

            selected = list(selected)

            if len(selected) > 4:
                subsample = select_prior_samples(selected, prior_samples, model)

                string = back_fitting(model, subsample)

                fig.clf()
                plt.text(0.05, 0.5, string, fontsize=14)

                plt.yticks([])
                plt.xticks([])
            else:
                fig.suptitle("Please select more distributions", fontsize=16)

            fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", click)

    controls = widgets.VBox([button_carry_on, button_return_prior])

    display(  # pylint:disable=undefined-variable
        widgets.HBox([controls, radio_buttons_kind, check_button_sharex])
    )


def carry_on(
    fig,
    axes,
    fig_pp_mean,
    kind,
    sharex,
    references,
    clicked,
    pp_samples,
    pp_summary,
    choices,
    selected,
    collected_distances,
    shown,
    sample_size,
    boundaries,
    kdt,
):
    fig.suptitle("")

    if clicked:
        choices.extend([int(ax.get_title()) for ax in clicked])
        shuffle(choices)
        selected.update(choices)
        selected, shown = collect_more_samples(
            selected,
            collected_distances,
            shown,
            pp_summary,
            pp_samples,
            sample_size,
            boundaries,
            kdt,
        )

        for ax in clicked:
            plt.setp(ax.spines.values(), color="k", lw=1)
        for ax in axes:
            ax.cla()
        for ax in list(clicked):
            clicked.remove(ax)

    pp_samples_idxs, distances, shown = keep_sampling(pp_summary, choices, shown, sample_size, kdt)

    if not pp_samples_idxs:
        pp_samples_idxs, distances, shown = initialize_subsamples(
            pp_summary, shown, sample_size, kdt, None
        )

    collected_distances.update(distances)
    plot_pp_mean(pp_samples, list(selected), references, kind, fig_pp_mean)

    if pp_samples_idxs:
        plot_pp_samples(pp_samples, pp_samples_idxs, references, kind, sharex, fig)
    else:
        # Instead of showing this message, we should resample.
        fig.clf()
        fig.suptitle("We have seen all the samples", fontsize=16)
        fig.canvas.draw()


def compute_summaries(pp_samples, summary):
    if summary == "octiles":
        pp_summary = np.quantile(
            pp_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1
        ).T
    elif summary == "quartiles":
        pp_summary = np.quantile(
            pp_samples,
            [
                0.25,
                0.5,
                0.75,
            ],
            axis=1,
        ).T

    elif summary == "hexiles":
        pp_summary = np.quantile(pp_samples, [0.25, 0.375, 0.5, 0.625, 0.75], axis=1).T
    elif summary == "sort":
        pp_summary = np.sort(pp_samples)
    elif summary == "octi_sum":
        suma = np.quantile(pp_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1).T
        s_a = suma[:, 3]
        s_b = suma[:, 5] - suma[:, 1]
        s_g = (suma[:, 5] + suma[:, 1] - 2 * suma[:, 3]) / s_b
        s_k = (suma[:, 6] - suma[:, 4] + suma[:, 2] - suma[:, 0]) / s_b
        pp_summary = np.stack([s_a, s_b, s_g, s_k]).T

    kdt = KDTree(pp_summary)
    return pp_summary, kdt


def add_init_dist(ref_dist, pp_samples):
    sample_size = pp_samples.shape[1]
    if isinstance(ref_dist, tuple):
        ref_sample = Normal(*ref_dist).rvs(sample_size)
    elif isinstance(ref_dist, Distribution):
        ref_sample = ref_dist.rvs(sample_size)

    pp_samples = np.vstack([ref_sample, pp_samples])
    return pp_samples


def initialize_subsamples(pp_summary, shown, sample_size, kdt, ref_dist):
    samples = []
    distances = {}

    if len(shown) != sample_size:
        if ref_dist is None:
            new = np.random.choice(list(set(range(0, len(pp_summary))) - shown))
            samples.append(new)

            for _ in range(8):
                farthest_neighbor = sample_size
                while new in samples or new in shown:
                    # we search for the farthest_neighbor
                    _, new = kdt.query(pp_summary[samples[-1]], [farthest_neighbor], workers=-1)
                    new = new.item()
                    farthest_neighbor -= 1
                # Missing neighbors are indicated with index==sample_size
                if new != sample_size:
                    samples.append(new)
        else:
            new = 0
            samples.append(new)

            for _ in range(9):
                nearest_neighbor = 2
                while new in samples:
                    distance, new = kdt.query(
                        pp_summary[samples[-1]], [nearest_neighbor], workers=-1
                    )
                    new = new.item()
                    nearest_neighbor += 1

                if new != sample_size:
                    samples.append(new)
                    distances[new] = distance.item()

            samples = samples[1:]

        shown.update(samples)

    return samples, distances, shown


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


def select_prior_samples(selected, prior_samples, model):
    """
    Given a selected set of prior predictive samples pick the corresponding
    prior samples.
    """
    subsample = {rv: prior_samples[rv][selected] for rv in model.keys()}

    return subsample


def back_fitting(model, subset):
    """
    Use MLE to fit a subset of the prior samples to the marginal prior distributions
    """
    string = "Your selection is consistent with the priors (original families):\n"

    for name, dist in model.items():
        dist._fit_mle(subset[name])
        string += f"{name} = {repr_to_matplotlib(dist)}\n"

    string += "\nYour selection is consistent with the priors (new families):\n"

    exclude, distributions = get_distributions()
    for name, dist in model.items():
        if dist.__class__.__name__ in exclude:
            dist._fit_mle(subset[name])
        else:
            idx, _ = mle(distributions, subset[name])
            dist = distributions[idx[0]]
        string += f"{name} = {repr_to_matplotlib(dist)}\n"

    return string


def get_distributions():
    exclude = [
        "Beta",
        "BetaScaled",
        "Triangular",
        "TruncatedNormal",
        "Uniform",
        "VonMises",
        "DiscreteUniform",
    ]
    all_distributions = modules["preliz.distributions"].__all__
    distributions = []
    for a_dist in all_distributions:
        dist = getattr(modules["preliz.distributions"], a_dist)()
        if dist.__class__.__name__ not in exclude:
            distributions.append(dist)
    return exclude, distributions
