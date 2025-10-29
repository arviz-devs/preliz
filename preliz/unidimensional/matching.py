import warnings

import numpy as np

from preliz.internal.distribution_helper import valid_distribution
from preliz.internal.optimization import get_fixed_params, optimize_moments, optimize_quantiles
from preliz.internal.rcparams import rcParams
from preliz.ppls.pymc_io import if_pymc_get_preliz


def match_moments(
    from_dist,
    to_dist,
    moments="mv",
    plot=None,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the distribution `to_dist` that matches the moments of `from_dist`.

    Parameters
    ----------
    from_dist : PreliZ distribution or PyMC distribution
        Instance of a fully parametrized PreliZ distribution. We will take the moments
        from this distribution.
    to_dist : PreliZ distribution or PyMC distribution
        Instance of a distribution to be fitted to match the moments of `from_dist`.
        If a PreliZ distribution then it can have some parameters fixed.
        PreliZ distributions are updated inplace.
    moments : str
        The type of moments to compute. Default is 'mv' (mean and variance).
        Valid combinations are any subset of 'mvdsk', where 'm' = mean,
        'v' = variance, 's' = skewness, and 'k' = kurtosis.
        'd' = standard deviation is also valid.
    plot : bool
        Whether to plot the distributions. Defaults to None, which results in
        the value of rcParams["plots.show_plot"] being used.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``from_dist`` and ``to_dist``.
    ax : matplotlib axes

    Returns
    -------
    dict: PreliZ distribution
    axes: matplotlib axes (only if `plot=True`)

    Notes
    -----
    After calling this function the attribute `opt` of the distribution will be updated with the
    OptimizeResult object from the optimization step.

    See Also
    --------
    match_quantiles : Match the distribution to the specified quantiles.

    Examples
    --------
    Moment matching between a known Normal and a Gamma distribution:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.match_moments(pz.Normal(14, 1), pz.Gamma())


    Moment matching between a known Normal and a StudentT distribution with nu fixed to 5:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.match_moments(pz.Normal(14, 1), pz.StudentT(nu=5))
    """
    from_dist = if_pymc_get_preliz(from_dist)
    to_dist = if_pymc_get_preliz(to_dist)

    valid_distribution(from_dist)
    valid_distribution(to_dist)

    if plot is None:
        plot = rcParams["plots.show_plot"]

    if plot_kwargs is None:
        plot_kwargs = {}

    none_idx, fixed = get_fixed_params(to_dist)

    target_values = np.array(from_dist.moments(moments))
    if not np.any(np.isfinite(target_values)):
        raise ValueError(
            f"At least one of the requested moments ({moments}) of `from_dist` is not finite."
        )

    # Initialize `to_dist` to a distribution matching the mean and standard deviation
    # of `from_dist`. The ``_fit_moments`` method is correct for some distributions,
    # but just an heuristic for others.
    to_dist._fit_moments(from_dist.mean(), from_dist.std())

    opt = optimize_moments(to_dist, moments, target_values, none_idx, fixed)
    to_dist.opt = opt
    requested_moments = to_dist.moments(moments)

    if not np.all(np.isfinite(requested_moments)):
        raise ValueError(
            f"At least one of the requested moments ({moments}) of `to_dist` is not finite."
        )

    _check_relative_error(moments, target_values, requested_moments)

    if plot:
        ax = from_dist.plot_pdf(**plot_kwargs)
        to_dist.plot_pdf(ax=ax, **plot_kwargs)
        return to_dist, ax

    return to_dist


def match_quantiles(
    from_dist,
    to_dist,
    quantiles=None,
    plot=None,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the distribution `to_dist` that matches the moments of `from_dist`.

    Parameters
    ----------
    from_dist : PreliZ distribution or PyMC distribution
        Instance of a fully parametrized PreliZ distribution. We will take the moments
        from this distribution.
    to_dist : PreliZ distribution or PyMC distribution
        Instance of a distribution to be fitted to match the quantiles of `from_dist`.
        If a PreliZ distribution then it can have some parameters fixed.
        PreliZ distributions are updated inplace.
    quantiles : array-like, optional
        Quantiles to match. Default is [0.25, 0.5, 0.75].
    plot : bool
        Whether to plot the distributions. Defaults to None, which results in
        the value of rcParams["plots.show_plot"] being used.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``from_dist`` and ``to_dist``.
    ax : matplotlib axes

    Returns
    -------
    dict: PreliZ distribution
    axes: matplotlib axes (only if `plot=True`)

    Notes
    -----
    After calling this function the attribute `opt` of the distribution will be updated with the
    OptimizeResult object from the optimization step.

    See Also
    --------
    match_moments : Match the distribution to the specified moments.

    Examples
    --------
    Moment matching between a known Normal and a Gamma distribution:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.match_quantiles(pz.Normal(14, 1), pz.Gamma())


    Moment matching between a known Normal and a StudentT distribution with nu fixed to 5:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.match_quantiles(pz.Normal(14, 1), pz.StudentT(nu=5))
    """
    from_dist = if_pymc_get_preliz(from_dist)
    to_dist = if_pymc_get_preliz(to_dist)

    valid_distribution(from_dist)
    valid_distribution(to_dist)

    if quantiles is None:
        quantiles = np.array([0.25, 0.5, 0.75])
    else:
        quantiles = np.asarray(quantiles)
        if np.any((quantiles <= 0) | (quantiles >= 1)):
            raise ValueError("Quantiles must be between 0 and 1.")

    if plot is None:
        plot = rcParams["plots.show_plot"]

    if plot_kwargs is None:
        plot_kwargs = {}

    none_idx, fixed = get_fixed_params(to_dist)

    target_values = np.array(from_dist.ppf(quantiles))

    # Initialize `to_dist` to a distribution matching the mean and standard deviation
    # of `from_dist`. The ``_fit_moments`` method is correct for some distributions,
    # but just an heuristic for others.
    to_dist._fit_moments(from_dist.mean(), from_dist.std())

    opt = optimize_quantiles(to_dist, quantiles, target_values, none_idx, fixed)
    to_dist.opt = opt
    requested_moments = to_dist.ppf(quantiles)

    _check_relative_error(quantiles, target_values, requested_moments, tol=0.1)

    if plot:
        ax = from_dist.plot_pdf(**plot_kwargs)
        to_dist.plot_pdf(ax=ax, **plot_kwargs)
        return to_dist, ax

    return to_dist


def _check_relative_error(values, target_values, requested_moments, tol=0.01):
    errors = abs((requested_moments - target_values) / (target_values + 1e-6) * 100)

    if np.any(errors > tol):
        msg = "There is a mismatch:"
        for idx, error in enumerate(errors):
            if error > tol:
                msg += (
                    f"\n - {values[idx]}: {target_values[idx]:.3g} vs {requested_moments[idx]:.3g}"
                )

        if msg:
            warnings.warn(
                msg,
                stacklevel=2,
            )
