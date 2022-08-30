import logging
import numpy as np

from .utils.optimization import fit_to_sample

_log = logging.getLogger("preliz")


def mle(
    distributions,
    sample,
    plot=1,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the maximum likelihood distribution with given a list of distributions
    and one sample.

    Parameters
    ----------
    distributions : list of PreliZ distribution
        Instance of a PreliZ distribution. Notice that the distributions will be
        updated inplace.
    sample : list or 1D array-like
        Data used to estimate the distribution parameters.
    plot : int
        Number of distributions to plots. Defaults to ``1`` (i.e. plot the best match)
        If larger than the number of passed distributions it will plot all of them.
        Use ``0`` or ``False`` to disable plotting.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------
    axes: matplotlib axes
    """
    sample = np.array(sample)
    x_min = sample.min()
    x_max = sample.max()

    fitted = fit_to_sample(distributions, sample, x_min, x_max)

    if plot > len(distributions):
        plot = len(distributions)

    if plot:
        idx = np.argsort(fitted.losses)
        idx = idx[:plot]
        for dist in fitted.distributions[idx]:
            if dist is not None:
                ax = dist.plot_pdf(plot_kwargs)
    return ax
