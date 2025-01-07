import warnings

from preliz.distributions.normal import Normal
from preliz.internal.distribution_helper import valid_distribution
from preliz.internal.optimization import relative_error, optimize_max_ent, get_fixed_params
from preliz.internal.rcparams import rcParams


def maxent(
    distribution=None,
    lower=-1,
    upper=1,
    mass=None,
    mode=None,
    fixed_stat=None,
    plot=None,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the maximum entropy distribution with `mass` in the interval defined by the `lower` and
    `upper` end-points.

    Parameters
    ----------
    name : PreliZ distribution
        Instance of a PreliZ distribution. Notice that the distribution will be
        updated inplace.
    lower : float
        Lower end-point
    upper: float
        Upper end-point
    mass: float
        Probability mass between ``lower`` and ``upper`` bounds. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.
    mode: float
        Mode of the distribution. Pass a value to fix the mode of the distribution.
    fixed_stat: tuple
        Summary statistic to fix. The first element should be a name and the second a
        numerical value. Valid names are: "mean", "mode", "median", "variance", "std",
        "skewness", "kurtosis". Defaults to None.
    plot : bool
        Whether to plot the distribution, and lower and upper bounds. Defaults to None,
        which results in the value of rcParams["plots.show_plot"] being used.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------

    dict: dict with the parameters of the distribution
    axes: matplotlib axes (only if `plot=True`)

    Notes
    -----
    After calling this function the attribute `opt` of the distribution will be updated with the
    OptimizeResult object from the optimization step.

    See Also
    --------
    quartile : Find the distribution with the specified quartiles.

    Examples
    --------
    Calculate the maxent Gamma distribution with 90 % of the mass between 1 and 8:

    .. plot::
        :context: close-figs
        :include-source: true

        >>>
        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.maxent(pz.Gamma(), 1, 8, 0.9)

    Calculate the maxent HalfStudentT T distribution with 90 % of the mass between 0 and 12
    and a value of nu=4:

    .. plot::
        :context: close-figs
        :include-source: true

        >>>
        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.maxent(pz.HalfStudentT(nu=4), 0, 12, 0.9)

    """
    valid_distribution(distribution)

    if mode is not None:
        fixed_stat = ("mode", mode)
        warnings.warn(
            "The parameter `mode` is deprecated and will be removed in a future release. "
            "Use `fixed_stat=('mode', mode)` instead.",
            FutureWarning,
        )

    if mass is None:
        mass = rcParams["stats.ci_prob"]

    if plot is None:
        plot = rcParams["plots.show_plot"]

    if fixed_stat is None:
        fixed_stat = ()
    else:
        valid_stats = ["mean", "mode", "median", "variance", "std", "skewness", "kurtosis"]
        if fixed_stat[0] not in valid_stats:
            raise ValueError("fixed_stat should be one of the following: " + ", ".join(valid_stats))

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    if plot_kwargs is None:
        plot_kwargs = {}

    if distribution is None:
        distribution = Normal()

    if distribution.is_frozen:
        raise ValueError("All parameters are fixed, at least one should be free")

    distribution._check_endpoints(lower, upper)

    if distribution.kind == "discrete":
        if not end_points_ints(lower, upper):
            warnings.warn(
                f"\n{distribution.__class__.__name__} distribution is discrete, "
                "but the provided bounds are not integers"
            )

    # Find which parameters has been fixed
    none_idx, fixed_params = get_fixed_params(distribution)

    # Heuristic to provide an initial guess for the optimization step
    # We obtain those guesses by first approximating the mean and standard deviation
    # from intervals and mass and then use those values for moment matching
    if distribution.__class__.__name__ == "Uniform":
        distribution._fit_moments(  # pylint:disable=protected-access
            mean=(lower + upper) / 2, sigma=((upper - lower) / 3.4) / mass
        )
    else:
        distribution._fit_moments(  # pylint:disable=protected-access
            mean=(lower + upper) / 2, sigma=((upper - lower) / 4) / mass
        )

    if "mode" in fixed_stat:
        try:
            distribution.mode()
        except NotImplementedError as exc:
            raise ValueError(
                f"{distribution.__class__.__name__} does not have a mode method"
            ) from exc

    opt = optimize_max_ent(distribution, lower, upper, mass, none_idx, fixed_params, fixed_stat)
    distribution.opt = opt

    r_error, computed_mass = relative_error(distribution, lower, upper, mass)

    if r_error > 0.01:
        warnings.warn(
            f"\nThe requested mass is {mass:.3g},\n" f"but the computed one is {computed_mass:.3g}",
            stacklevel=2,
        )

    if plot:
        ax = distribution.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot([lower, upper], [0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
        return distribution, ax

    return distribution


def end_points_ints(lower, upper):
    return is_integer_num(lower) and is_integer_num(upper)


def is_integer_num(obj):
    if isinstance(obj, int):
        return True
    if isinstance(obj, float):
        return obj.is_integer()
    return False
