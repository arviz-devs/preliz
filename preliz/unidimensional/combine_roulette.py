import numpy as np

from preliz.internal.distribution_helper import get_distributions
from preliz.internal.distribution_helper import process_extra
from preliz.internal.optimization import fit_to_epdf


def combine_roulette(responses, weights=None, dist_names=None, params=None):
    """
    Combine multiple elicited distributions into a single distribution.

    Parameters
    ----------
    responses : list of tuples
        Typically, each tuple comes from the ``.inputs`` attribute of a ``Roulette`` object and
        represents a single elicited distribution.
    weights : array-like, optional
        Weights for each elicited distribution. Defaults to None, i.e. equal weights.
        The sum of the weights must be equal to 1, otherwise it will be normalized.
    dist_names: list
        List of distributions names to be used in the elicitation.
        Defaults to ["Normal", "BetaScaled", "Gamma", "LogNormal", "StudentT"].
    params : str, optional
        Extra parameters to be passed to the distributions. The format is a string with the
        PreliZ's distribution name followed by the argument to fix.
        For example: "TruncatedNormal(lower=0), StudentT(nu=8)".

    Returns
    -------
    PreliZ distribution
    """

    if params is not None:
        extra_pros = process_extra(params)
    else:
        extra_pros = []

    if weights is None:
        weights = np.full(len(responses), 1 / len(responses))
    else:
        weights = np.array(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("The weights must be positive.")

    weights /= weights.sum()

    if not all(records[3:] == responses[0][3:] for records in responses):
        raise ValueError(
            "To combine single elicitation instances, the grid should be the same for all of them."
        )

    if dist_names is None:
        dist_names = ["Normal", "BetaScaled", "Gamma", "LogNormal", "StudentT"]

    new_pdf = {}
    for records, weight in zip(responses, weights):
        chips = records[2]
        for x_i, pdf_i in zip(records[0], records[1]):
            if x_i in new_pdf:
                new_pdf[x_i] += pdf_i * weight * chips
            else:
                new_pdf[x_i] = pdf_i * weight * chips

    total = sum(new_pdf.values())
    mean = 0
    for x_i, pdf_i in new_pdf.items():
        val = pdf_i / total
        mean += x_i * val
        new_pdf[x_i] = val

    var = 0
    for x_i, pdf_i in new_pdf.items():
        var += pdf_i * (x_i - mean) ** 2
    std = var**0.5

    # Assuming all the elicited distributions have the same x_min and x_max
    x_min = responses[0][3]
    x_max = responses[0][4]

    fitted_dist = fit_to_epdf(
        get_distributions(dist_names),
        list(new_pdf.keys()),
        list(new_pdf.values()),
        mean,
        std,
        x_min,
        x_max,
        extra_pros,
    )

    return fitted_dist
