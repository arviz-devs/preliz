from sys import modules

import numpy as np


from .plot_helper import (
    repr_to_matplotlib,
)

from ..unidimensional import mle


def back_fitting(model, subset, new_families=True):
    """
    Use MLE to fit a subset of the prior samples to the marginal prior distributions
    """
    string = "Your selection is consistent with the priors (original families):\n"

    for name, dist in model.items():
        dist._fit_mle(subset[name])
        string += f"{name} = {repr_to_matplotlib(dist)}\n"

    if new_families:
        string += "\nYour selection is consistent with the priors (new families):\n"

        exclude, distributions = get_distributions()
        for name, dist in model.items():
            if dist.__class__.__name__ in exclude:
                dist._fit_mle(subset[name])
            else:
                idx, _ = mle(distributions, subset[name], plot=False)
                dist = distributions[idx[0]]
            string += f"{name} = {repr_to_matplotlib(dist)}\n"

    return string, np.concatenate([dist.params for dist in model.values()])


def get_distributions():
    exclude = [
        "Beta",
        "BetaScaled",
        "Triangular",
        "TruncatedNormal",
        "Uniform",
        "VonMises",
        "Categorical",
        "DiscreteUniform",
        "HyperGeometric",
        "zeroInflatedBinomial",
        "ZeroInflatedNegativeBinomial",
        "ZeroInflatedPoisson",
        "MvNormal",
    ]
    all_distributions = modules["preliz.distributions"].__all__
    distributions = []
    for a_dist in all_distributions:
        dist = getattr(modules["preliz.distributions"], a_dist)()
        if dist.__class__.__name__ not in exclude:
            distributions.append(dist)
    return exclude, distributions


def select_prior_samples(selected, prior_samples, model):
    """
    Given a selected set of prior predictive samples pick the corresponding
    prior samples.
    """
    subsample = {rv: prior_samples[rv][selected] for rv in model.keys()}

    return subsample
