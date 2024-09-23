import numpy as np

from preliz.unidimensional.mle import mle
from preliz.internal.distribution_helper import get_distributions


def combine(
    distributions,
    weights=None,
    dist_names=None,
    sample_size=10_000,
    rng=0,
    plot=1,
    plot_kwargs=None,
    ax=None,
):
    """
    Combine a set of distributions into a single one.

    Fit a weighted sample from ``distributions`` into the distributions listed in ``dist_names`.
    The fit is done using maximum likelihood estimation, and the best match is plotted.
    Notice that the result is NOT a Mixture distribution, but a single distribution
    that best fits the weighted sample.

    Parameters
    ----------
    distributions : List of PreliZ distributions
        These are the distributions that we want to combine. Typically, these have been
        elicited from different individuals or instances.
    weights : array-like, optional
        Weights for each distribution. Defaults to None, i.e. equal weights.
        The sum of the weights must be equal to 1, otherwise it will be normalized.
    dist_names: list
        List of distributions to fit the weighted sample.
        Defaults to ``["Normal", "Gamma", "LogNormal", "StudentT"]``.
    sample_size : int
        Number of total samples to generate for the fit.
    rng : int or numpy.random.Generator, optional
        Random number generator or seed. Defaults to ``0``.
    plot : int
        Number of distributions to plots. Defaults to ``1`` (i.e. plot the best match)
        If larger than the number of passed distributions it will plot all of them.
        Use ``0`` or ``False`` to disable plotting.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()``.
    ax : matplotlib axes

    Returns
    -------
    fitted_distributions : list of PreliZ distributions.
        The distributions that best fit the weighted sample. Sorted from best to worst match.
    axes : matplotlib axes
    """

    if rng is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng)

    if weights is None:
        weights = np.full(len(distributions), 1 / len(distributions))
    else:
        weights = np.array(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("The weights must be positive.")

    weights /= weights.sum()

    n_size = (sample_size * weights).astype(int)

    if dist_names is None:
        dist_names = ["Normal", "Gamma", "LogNormal", "StudentT"]

    sample = []
    for dist, n in zip(distributions, n_size):
        sample.append(dist.rvs(n, random_state=rng))

    distributions = get_distributions(dist_names)

    idx, ax = mle(distributions, np.concatenate(sample), plot=plot, plot_kwargs=plot_kwargs, ax=ax)

    return np.array(distributions)[idx], ax
