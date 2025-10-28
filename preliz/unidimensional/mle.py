import warnings

import numpy as np

from preliz.internal.distribution_helper import valid_distribution
from preliz.internal.optimization import fit_to_sample
from preliz.internal.rcparams import rcParams
from preliz.ppls.pymc_io import if_pymc_get_preliz


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
    distributions : list of PreliZ (or PyMC) distributions
        Instance of a distribution. Notice that the distributions will be
        updated inplace. If the distribution is from PyMC, it will be
        converted to a PreliZ distribution, use `.to_pymc()` to convert it back to PyMC.
        All parameters will be estimated from the data, no parameter can be fixed.
        For PreliZ distributions, pass uninitialized distributions.
        For some PyMC distributions, you may need to pass `np.nan` for parameters.
    sample : list or 1D array-like
        Data used to estimate the distribution parameters.
    plot : int
        Number of distributions to plots. Defaults to ``1`` (i.e. plot the best match)
        If larger than the number of passed distributions it will plot all of them.
        Use ``0`` or ``False`` to disable plotting.
        If you want to disable plotting globally you can also set
        ``rcParams["plots.show_plot"] = False``.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------
    idx : array with the indexes to sort ``distributions`` from best to worst match
    axes : matplotlib axes

    Examples
    --------
    Fit a Normal and a Gamma distribution to data sampled from a Moyal distribution.
    In "real life" instead of a sample from a known distribution you would have some observed data.

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> sample = pz.Moyal(1, 2).rvs(1000)  # some random data
        >>> pz.mle([pz.Normal(), pz.Gamma()], sample)

    References
    ----------

    .. footbibliography::

    """
    dists = []
    for dist in distributions:
        dist_ = if_pymc_get_preliz(dist)
        valid_distribution(dist_)
        dists.append(dist_)

    sample = np.array(sample)
    x_min = sample.min()
    x_max = sample.max()

    fitted = fit_to_sample(dists, sample, x_min, x_max)

    plot = min(plot, len(dists))

    idx = np.argsort(fitted.losses)

    if all(dist is None or not dist.is_frozen for dist in fitted.distributions):
        warnings.warn(
            """
                      No distribution was fitted. This is likely because the support of the
                      distributions is incompatible with the sampled values."""
        )

    if plot and rcParams["plots.show_plot"]:
        plot_idx = idx[:plot]
        for dist in fitted.distributions[plot_idx]:
            if dist is not None and dist.is_frozen:
                ax = dist.plot_pdf(plot_kwargs)

    return idx, ax
