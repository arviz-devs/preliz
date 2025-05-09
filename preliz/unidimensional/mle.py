import logging
import warnings

import numpy as np

from preliz.internal.distribution_helper import valid_distribution
from preliz.internal.optimization import fit_to_sample

_log = logging.getLogger("preliz")


def mle(
    distributions,
    sample,
    plot=1,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the maximum likelihood distribution given a list of distributions and one sample.

    AIC with a correction for small sample sizes is used to compare the fits.
    (See :footcite:t:`Burnham2004`)

    Parameters
    ----------
    distributions : list of PreliZ distribution
        Instance of a PreliZ distribution. Notice that the distributions will be
        updated inplace.
    sample : list or 1D array-like
        Data used to estimate the distribution parameters.
    ignore_support : bool

    plot : int
        Number of distributions to plots. Defaults to ``1`` (i.e. plot the best match)
        If larger than the number of passed distributions it will plot all of them.
        Use ``0`` or ``False`` to disable plotting.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------
    idx : array with the indexes to sort ``distributions`` from best to worst match
    axes : matplotlib axes

    References
    ----------

    .. footbibliography::

    """
    for dist in distributions:
        valid_distribution(dist)

    sample = np.array(sample)
    x_min = sample.min()
    x_max = sample.max()

    fitted = fit_to_sample(distributions, sample, x_min, x_max)

    plot = min(plot, len(distributions))

    idx = np.argsort(fitted.losses)

    if all(dist is None or not dist.is_frozen for dist in fitted.distributions):
        warnings.warn(
            """
                      No distribution was fitted. This is likely because the support of the
                      distributions is incompatible with the sampled values."""
        )

    if plot:
        plot_idx = idx[:plot]
        for dist in fitted.distributions[plot_idx]:
            if dist is not None and dist.is_frozen:
                ax = dist.plot_pdf(plot_kwargs)

    return idx, ax
