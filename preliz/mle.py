import logging
import numpy as np

from .utils.optimization import fit_to_sample

_log = logging.getLogger("preliz")


def mle(
    distributions,
    sample,
    plot="first",
    plot_kwargs=None,
    ax=None,
):
    """
    Find the maximum likelihood distribution with given a list of distributions
    and one sample.

    Parameters
    ----------
    distributions : list of PreliZ distribution
        Instance of a PreliZ distribution
    sample : list or 1D array-like

    Returns
    -------
    PreliZ distribution

    """
    sample = np.array(sample)
    x_min = sample.min()
    x_max = sample.max()

    fitted = fit_to_sample(distributions, sample, x_min, x_max)

    if plot:
        idx = np.argsort(fitted.losses)
        if plot == "first":
            idx = idx[:1]
        for dist in fitted.distributions[idx]:
            if dist is not None:
                ax = dist.plot_pdf(plot_kwargs)
    return ax
