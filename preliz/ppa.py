"""Prior predictive check assistant."""

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from preliz import distributions
from .utils.plot_utils import plot_pointinterval, repr_to_matplotlib

_log = logging.getLogger("preliz")

pymc_to_preliz = {
    "normal": "Normal",
    "halfnormal": "HalfNormal",
}


def ppa(idata, model, prepros="octiles", random_seed=None, backfitting=True):
    """
    Prior predictive check assistant.

    This is experimental

    Parameters
    ----------

    idata : InferenceData
        With samples from the prior and prior predictive distributions
    model : PyMC model
        Model associated to ``ìdata``.
    summary : str
        summary statistic using for clustering
    method : str
        clustering method
    random_seed : int
        random seed passed to the clustering method
    backfitting : bool
        This is not doing anything at this point
    """
    _log.info(
        """Enter at your own risk."""
        """This is highly experimental code and not recommended for regular use."""
    )
    obs_rv = model.observed_RVs[0].name  # only one observed variable for the moment
    pp_samples = idata.prior_predictive[obs_rv].squeeze().values
    prior_samples = idata.prior.squeeze()
    sample_size = pp_samples.shape[0]
    pp_summary, kdt = pre_processing(pp_samples, prepros)
    pp_samples_idxs, shown = initialize_subsamples(pp_summary, kdt)
    fig, _ = plot_clusters(pp_samples, pp_samples_idxs)

    clicked = []

    def onclick(event):
        if event.inaxes not in clicked:
            clicked.append(event.inaxes)
        else:
            clicked.remove(event.inaxes)
            plt.setp(event.inaxes.spines.values(), color="k", lw=1)

        for ax in clicked:
            plt.setp(ax.spines.values(), color="C1", lw=3)

    def on_leave_fig(event):  # pylint: disable=unused-argument
        if clicked:
            choices = [int(ax.get_title()) for ax in clicked]
            _ = keep_subsampling(choices, shown, kdt)
            # pps1 = resample(choices, prior_samples, model, db0)
            if backfitting:
                string = new_priors_back_fitting(model, pps1, sample_size)
            # else:
            #     string = new_priors_posterior(model, pps1, sample_size)

            fig.clf()
            plt.text(0.2, 0.5, string, fontsize=14)
            plt.yticks([])
            plt.xticks([])

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("figure_leave_event", on_leave_fig)


def pre_processing(pp_samples, prepros):
    if prepros is None:
        pp_summary = pp_samples
    else:
        if prepros == "octiles":
            pp_summary = np.quantile(
                pp_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1
            ).T
        elif prepros == "quartiles":
            pp_summary = np.quantile(
                pp_samples,
                [
                    0.25,
                    0.5,
                    0.75,
                ],
                axis=1,
            ).T

        elif prepros == "hexiles":
            pp_summary = np.quantile(pp_samples, [0.25, 0.375, 0.5, 0.625, 0.75], axis=1).T
        elif prepros == "sort":
            pp_summary = np.sort(pp_samples)
        elif prepros == "octi_sum":
            suma = np.quantile(smp, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1).T
            sa = suma[:, 3]
            sb = suma[:, 5] - suma[:, 1]
            sg = (suma[:, 5] + suma[:, 1] - 2 * suma[:, 3]) / sb
            sk = (suma[:, 6] - suma[:, 4] + suma[:, 2] - suma[:, 0]) / sb
            pp_summary = np.stack([sa, sb, sg, sk]).T

    kdt = KDTree(pp_summary)
    return pp_summary, kdt


def initialize_subsamples(pp_summary, kdt):
    shown = []
    new = 0  # np.random.choice(list(set(range(0, len(smp))) - set(shown)))
    samples = [new]

    for _ in range(8):
        length = pp_summary.shape[0]
        while new in samples:
            _, new = kdt.query(pp_summary[samples[-1]], [length])
            new = new.item()
            length -= 1
        samples.append(new)

    shown.extend(samples)

    return samples, shown


def keep_subsampling(choices, shown, kdt):
    # Select distributions that are similar
    # TODO Generalize to work with more than one choice
    new = choices[0]
    samples = [new]

    for i in range(8):
        nn = 2
        while new in samples or new in shown:
            _, new = kdt.query(pp_summary[samples[-1]], [nn])
            new = new.item()
            nn += 1
        samples.append(new)

    shown.extend(samples)

    for i in smp[samples]:
        az.plot_kde(i)
    shown


def plot_clusters(pp_samples, pp_samples_idxs):
    row_colum = int(np.ceil(len(pp_samples_idxs) ** 0.5))

    fig, axes = plt.subplots(row_colum, row_colum, figsize=(8, 6), sharex=True)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    for ax, idx in zip(axes, pp_samples_idxs):
        ax.axvline(0, ls="--", color="0.5")

        sample = pp_samples[idx]
        # for s in sample:
        #    az.plot_kde(s, ax=ax)
        az.plot_kde(sample, ax=ax, plot_kwargs={"color": "C0"})

        plot_pointinterval(sample, ax=ax)
        ax.set_title(idx)
        ax.set_yticks([])

    for idx in range(len(pp_samples_idxs), len(axes)):
        axes[idx].remove()
    return fig, axes


# def resample(choices, prior_samples, model, clust):
#     rvs = model.unobserved_RVs  # we should exclude deterministics
#     rvs_names = [rv.name for rv in rvs]
#     dado = {}
#     for rvn in rvs_names:
#         lista = []
#         for choice in choices:
#             lista.extend(prior_samples[rvn].values[clust.labels_ == choice])
#         dado[rvn] = np.array(lista)

#     # posti = pm.sample_posterior_predictive(az.from_dict(dado), model=model)
#     # dado.update(posti)
#     return dado


def new_priors_back_fitting(model, pps1, sample_size):
    subset = len(pps1[list(pps1.keys())[0]])

    string = f"You have selected {subset} out of {sample_size} prior predictive samples\n"
    string += "That selection is consistent with the following priors:\n"

    for unobserved in model.unobserved_RVs:
        distribution = unobserved.owner.op.name
        dist = getattr(distributions, pymc_to_preliz[distribution])()
        # pymc_name, sp_rv = pymc_to_scipy[distribution]
        name = unobserved.name
        idx = unobserved.owner.nin - 3
        params = unobserved.owner.inputs[-idx:]
        is_fitable = any(param.name is None for param in params)
        if is_fitable:
            # pymc_name, sp_rv = pymc_to_scipy[distribution]
            dist._fit_mle(pps1[name])
            string += f"{repr_to_matplotlib(dist)}\n"
            # string += f'{name} = pm.{pymc_name}("{name}", '
            # if distribution == "normal":
            #    string += f"{prior[0]:.2f}, {prior[1]:.2f})\n"
            # elif distribution == "halfnormal":
            #    string += f"{prior[1]:.2f})\n"

    return string
