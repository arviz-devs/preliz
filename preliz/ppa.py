# pylint:disable=global-variable-undefined
"""Prior predictive check assistant."""

import logging

import arviz as az
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


from .utils.plot_utils import plot_pointinterval, repr_to_matplotlib
from .utils.utils import get_pymc_to_preliz
from .distributions.continuous import Normal
from .distributions.distributions import Distribution

_log = logging.getLogger("preliz")


def ppa(idata, model, summary="octiles", references=0, init=None):
    """
    Prior predictive check assistant.

    This is experimental

    Parameters
    ----------

    idata : InferenceData
        With at least the `prior` and `prior_predictive` groups
    model : PyMC model
        Model associated to ``idata``.
    summary : str
        Summary statistics applied to prior samples in order to define (dis)similarity
        of distributions. Current options are `octiles`, `hexiles`, `quantiles`,
        `sort` (sort data) `octi_sum` (robust estimation of first 4 moments from octiles).
    references : int, float, list or tuple
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme.
    init : tuple or PreliZ distribtuion
        Initial distribution. The first shown distributions will be selected to be as close
        as possible to `init`. Available options are, a PreliZ distribution or a 2-tuple with
        the first element representing the mean and the second the standard deviation.
    """
    _log.info(
        """Enter at your own risk. """
        """This is highly experimental code and not recommended for regular use."""
    )

    try:
        shell = get_ipython().__class__.__name__  # pylint:disable=undefined-variable
        if shell == "ZMQInteractiveShell" and "nbagg" not in get_backend():
            _log.info(
                "To run ppa you need Jupyter notebook, or Jupyter lab."
                "You will also need to use the magic `%matplotlib widget`"
            )
    except NameError:
        pass

    if isinstance(references, (float, int)):
        references = [references]

    global pp_samples_idxs  # pylint:disable=invalid-name

    shown = []
    obs_rv = model.observed_RVs[0].name  # only one observed variable for the moment
    pp_samples = idata.prior_predictive[obs_rv].squeeze().values
    prior_samples = idata.prior.squeeze()
    if init is not None:
        pp_samples = add_init_dist(init, pp_samples)

    sample_size = pp_samples.shape[0]
    pp_summary, kdt = compute_summaries(pp_samples, summary)
    pp_samples_idxs, shown = initialize_subsamples(pp_summary, shown, kdt, init)
    fig, axes = plot_samples(pp_samples, references)

    clicked = []
    selected = []
    choices = []

    output = widgets.Output()

    with output:
        button_carry_on = widgets.Button(description="carry on")
        button_return_prior = widgets.Button(description="return prior")
        radio_buttons_kind = widgets.RadioButtons(
            options=["pdf", "hist", "ecdf"],
            value="pdf",
            description="",
            disabled=False,
        )

        check_button_sharex = widgets.Checkbox(
            value=True, description="sharex", disabled=False, indent=False
        )

        def carry_on_(_):
            carry_on(
                fig,
                axes,
                radio_buttons_kind.value,
                check_button_sharex.value,
                references,
                clicked,
                pp_samples,
                pp_summary,
                choices,
                selected,
                shown,
                kdt,
            )

        button_carry_on.on_click(carry_on_)

        def on_return_prior_(_):
            on_return_prior(fig, selected, model, sample_size)

        button_return_prior.on_click(on_return_prior_)

        def kind_(_):
            plot_samples(
                pp_samples, references, radio_buttons_kind.value, check_button_sharex.value, fig
            )

        radio_buttons_kind.observe(kind_, names=["value"])

        check_button_sharex.observe(kind_, names=["value"])

        def click(event):
            if event.inaxes not in clicked:
                clicked.append(event.inaxes)
            else:
                clicked.remove(event.inaxes)
                plt.setp(event.inaxes.spines.values(), color="k", lw=1)

            for ax in clicked:
                plt.setp(ax.spines.values(), color="C1", lw=3)
            fig.canvas.draw()

        def on_return_prior(fig, selected, model, sample_size):
            selected = np.unique(selected)
            num_selected = len(selected)

            if num_selected > 1:
                string = (
                    f"You have selected {num_selected} out of {sample_size} prior "
                    "predictive samples\nThat selection is consistent with the "
                    "following priors:\n"
                )

                subsample = select_prior_samples(selected, prior_samples, model)
                string = back_fitting(model, subsample, string)

                fig.clf()
                plt.text(0.2, 0.5, string, fontsize=14)
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
    kind,
    sharex,
    references,
    clicked,
    pp_samples,
    pp_summary,
    choices,
    selected,
    shown,
    kdt,
):
    global pp_samples_idxs  # pylint:disable=invalid-name

    choices.extend([int(ax.get_title()) for ax in clicked])
    selected.extend(choices)

    fig.suptitle("")
    for ax in clicked:
        plt.setp(ax.spines.values(), color="k", lw=1)
    for ax in axes:
        ax.cla()
    for ax in list(clicked):
        clicked.remove(ax)

    pp_samples_idxs, shown = keep_sampling(pp_summary, choices, shown, kdt)
    if not pp_samples_idxs:
        pp_samples_idxs, shown = initialize_subsamples(pp_summary, shown, kdt, None)
    fig, _ = plot_samples(pp_samples, references, kind, sharex, fig)


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


def initialize_subsamples(pp_summary, shown, kdt, ref_dist):
    if ref_dist is None:
        new = np.random.choice(list(set(range(0, len(pp_summary))) - set(shown)))
        samples = [new]

        for _ in range(8):
            farthest_neighbor = pp_summary.shape[0]
            while new in samples or new in shown:
                _, new = kdt.query(pp_summary[samples[-1]], [farthest_neighbor])
                new = new.item()
                farthest_neighbor -= 1
            samples.append(new)
        shown.extend(samples)
    else:
        new = 0
        samples = [new]

        for _ in range(9):
            nearest_neighbor = 2
            while new in samples:
                _, new = kdt.query(pp_summary[samples[-1]], [nearest_neighbor])
                new = new.item()
                nearest_neighbor += 1
            samples.append(new)

        samples = samples[1:]
        shown.extend(samples)

    return samples, shown


def keep_sampling(pp_summary, choices, shown, kdt):
    """
    Find distribution similar to the ones in `choices`, but not already shown.
    If `choices` is empty return an empty selection.
    """
    if choices:
        new = choices.pop(0)
        samples = [new]

        for _ in range(9):
            nearest_neighbor = 2
            while new in samples or new in shown:
                _, new = kdt.query(pp_summary[samples[-1]], [nearest_neighbor])
                new = new.item()
                nearest_neighbor += 1
            samples.append(new)

        shown.extend(samples[1:])

        return samples[1:], shown
    else:
        return [], shown


def plot_samples(pp_samples, references, kind="pdf", sharex=True, fig=None):
    row_colum = int(np.ceil(len(pp_samples_idxs) ** 0.5))

    if fig is None:
        fig, axes = plt.subplots(row_colum, row_colum, figsize=(8, 6))
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
            az.plot_kde(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member
        elif kind == "hist":
            bins, *_ = ax.hist(
                sample, color="C0", bins="auto", alpha=0.5, density=True
            )  # pylint:disable=no-member
            ax.set_ylim(-bins.max() * 0.05, None)

        elif kind == "ecdf":
            az.plot_ecdf(sample, ax=ax, plot_kwargs={"color": "C0"})  # pylint:disable=no-member

        plot_pointinterval(sample, ax=ax)
        ax.set_title(idx)
        ax.set_yticks([])

    if sharex:
        for ax in axes:
            ax.set_xlim(np.floor(x_lims[0]), np.ceil(x_lims[1]))

    fig.canvas.draw()
    return fig, axes


def select_prior_samples(selected, prior_samples, model):
    """
    Given a selected set of prior predictive samples pick the corresponding
    prior samples.
    """
    rvs = model.unobserved_RVs  # we should exclude deterministics
    rv_names = [rv.name for rv in rvs]
    subsample = {rv: prior_samples[rv].sel(draw=selected).values.squeeze() for rv in rv_names}

    return subsample


def back_fitting(model, subset, string):
    """
    Use MLE to fit a subset of the prior samples to the individual prior distributions
    """
    pymc_to_preliz = get_pymc_to_preliz()
    for unobserved in model.unobserved_RVs:
        distribution = unobserved.owner.op.name
        dist = pymc_to_preliz[distribution]
        name = unobserved.name
        idx = unobserved.owner.nin - 3
        params = unobserved.owner.inputs[-idx:]
        is_fitable = any(param.name is None for param in params)
        if is_fitable:
            dist._fit_mle(subset[name])
            string += f"{repr_to_matplotlib(dist)}\n"
    return string
