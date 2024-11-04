import numpy as np


from preliz.internal.plot_helper import repr_to_matplotlib
from preliz.internal.distribution_helper import get_distributions
from preliz.unidimensional import mle


def back_fitting_ppa(model, subset, new_families=True):
    """
    Use MLE to fit a subset of the prior samples to the marginal prior distributions
    """
    string = "Your selection is consistent with the priors (original families):\n"

    for name, dist in model.items():
        dist._fit_mle(subset[name])
        string += f"{name} = {repr_to_matplotlib(dist)}\n"

    if new_families:
        string += "\nYour selection is consistent with the priors (new families):\n"

        # We should store this in a central place
        # So we use the same families for other functions
        common_cont = ["Gamma", "Exponential"]
        common_disc = ["Poisson", "NegativeBinomial"]
        for name, dist in model.items():
            if dist.kind == "continuous":
                distributions = get_distributions(set([dist.__class__.__name__] + common_cont))
            elif dist.kind == "discrete":
                distributions = get_distributions(set([dist.__class__.__name__] + common_disc))
            idx, _ = mle(distributions, subset[name], plot=False)
            string += f"{name} = {repr_to_matplotlib(distributions[idx[0]])}\n"

    return string, np.concatenate([dist.params for dist in model.values()])


def select_prior_samples(selected, prior_samples, model):
    """
    Given a selected set of prior predictive samples pick the corresponding
    prior samples.
    """
    subsample = {rv: prior_samples[rv][selected] for rv in model.keys()}

    return subsample
