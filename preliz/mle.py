import logging
import numpy as np

from .utils.optimization import get_distributions, fit_to_sample

_log = logging.getLogger("preliz")


def mle(sample, distributions=None):
    """
    Return the distribution with the maximum likelihood given sample and a list of distributions

    Parameters
    ----------
    sample : list or 1D array-like
    distributions : list of Preliz distributions or strings

    Returns
    -------
    PreliZ distribution

    """
    sample = np.array(sample)
    x_min = sample.min()
    x_max = sample.max()

    selected_distributions = get_distributions(
        [dist().__class__.__name__ if not isinstance(dist, str) else dist for dist in distributions]
    )

    if len(distributions) != len(selected_distributions):
        _log.info(
            "One or more of the name you passed were not understood.\nUsing %s ",
            selected_distributions,
        )

    dist = fit_to_sample(selected_distributions, sample, x_min, x_max)

    return dist
