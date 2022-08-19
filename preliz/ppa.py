"""Prior predictive check assistant."""

import logging

import matplotlib.pyplot as plt

import arviz as az
import numpy as np
import sklearn.cluster as clu
from sklearn.preprocessing import StandardScaler

from preliz import distributions
from .utils.plot_utils import plot_pointinterval, repr_to_matplotlib

_log = logging.getLogger("preliz")

pymc_to_preliz = {
    "normal": "Normal",
    "halfnormal": "HalfNormal",
}


def ppa(idata, model, prepros="octiles", method="affinity", random_seed=None, backfitting=True):
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
    db0 = clusterize(pp_samples, prepros, method, random_seed)
    fig, _ = plot_clusters(pp_samples, db0)

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
            pps1 = resample(choices, prior_samples, model, db0)
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


def clusterize(prior_predictive_samples, prepros="octiles", method="kmeans", random_seed=None):
    if prepros is None:
        pp_samples = prior_predictive_samples
    else:
        if prepros == "octiles":
            pp_samples = np.quantile(
                prior_predictive_samples, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], axis=1
            ).T
        elif prepros == "quartiles":
            pp_samples = np.quantile(
                prior_predictive_samples,
                [
                    0.25,
                    0.5,
                    0.75,
                ],
                axis=1,
            ).T

        elif prepros == "hexiles":
            pp_samples = np.quantile(
                prior_predictive_samples, [0.25, 0.375, 0.5, 0.625, 0.75], axis=1
            ).T
        elif prepros == "sort":
            pp_samples = np.sort(prior_predictive_samples)

    if method == "kmeans":
        clust = clu.KMeans(n_clusters=9, random_state=random_seed).fit(pp_samples)
    elif method == "spectral":
        pp_samples = StandardScaler().fit_transform(pp_samples)
        clust = clu.SpectralClustering(n_clusters=9, random_state=random_seed).fit(pp_samples)
    elif method == "affinity":
        pp_samples = StandardScaler().fit_transform(pp_samples)
        clust = clu.AffinityPropagation(random_state=random_seed).fit(pp_samples)
    else:
        raise NotImplementedError(f"The method {method} is not implemented")

    return clust


def plot_clusters(prior_predictive_samples, clust):
    clusters = np.unique(clust.labels_)
    clusters = clusters[clusters != -1]
    row_colum = int(np.ceil(len(clusters) ** 0.5))

    fig, axes = plt.subplots(row_colum, row_colum, figsize=(8, 6), sharex=True)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        ax.axvline(0, ls="--", color="0.5")

        sample = prior_predictive_samples[clust.labels_ == cluster]
        # for s in sample:
        #    az.plot_kde(s, ax=ax)
        az.plot_kde(sample, ax=ax, plot_kwargs={"color": "C0"})

        plot_pointinterval(sample, None, ax)
        ax.set_title(cluster)
        ax.set_yticks([])

    for idx in range(len(clusters), len(axes)):
        axes[idx].remove()
    return fig, axes


def resample(choices, prior_samples, model, clust):
    rvs = model.unobserved_RVs  # we should exclude deterministics
    rvs_names = [rv.name for rv in rvs]
    dado = {}
    for rvn in rvs_names:
        lista = []
        for choice in choices:
            lista.extend(prior_samples[rvn].values[clust.labels_ == choice])
        dado[rvn] = np.array(lista)

    # posti = pm.sample_posterior_predictive(az.from_dict(dado), model=model)
    # dado.update(posti)
    return dado


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
