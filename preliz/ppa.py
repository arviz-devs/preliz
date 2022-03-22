import logging

import matplotlib.pyplot as plt

from .utils.ppa_utils import (
    clusterize,
    plot_clusters,
    resample,
    new_priors_back_fitting,
)

_log = logging.getLogger("preliz")


def ppa(idata, model, prepros="octiles", method="affinity", random_seed=None, backfitting=True):
    _log.info(
        """Enter at your own risk."""
        """This is highly experimental code and not recommended for regular use."""
    )
    pp_samples = idata.prior_predictive["y"].squeeze().values
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
