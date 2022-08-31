import logging

from .distributions import Normal
from .utils.optimization import relative_error, optimize_max_ent

_log = logging.getLogger("preliz")


def maxent(
    distribution=None,
    lower=-1,
    upper=1,
    mass=0.90,
    plot=True,
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
        Probability mass between ``lower`` and ``upper`` bounds. Defaults to 0.9
    plot : bool
        Whether to plot the distribution, and lower and upper bounds. Defaults to True.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------

    axes: matplotlib axes
    """
    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    if plot_kwargs is None:
        plot_kwargs = {}

    if distribution is None:
        distribution = Normal()

    distribution._check_endpoints(lower, upper)

    if distribution.kind == "discrete":
        if not end_points_ints(lower, upper):
            _log.info(
                "%s distribution is discrete, but the provided bounds are not integers",
                distribution.name.capitalize(),
            )

    # Heuristic to provide an initial guess for the optimization step
    # We obtain those guesses by first approximating the mean and standard deviation
    # from intervals and mass and then use those values for moment matching
    distribution._fit_moments(  # pylint:disable=protected-access
        mean=(lower + upper) / 2, sigma=((upper - lower) / 4) / mass
    )

    opt = optimize_max_ent(distribution, lower, upper, mass)

    r_error, computed_mass = relative_error(distribution, lower, upper, mass)

    if r_error > 0.01:
        _log.info(
            " The requested mass is %.3g, but the computed one is %.3g",
            mass,
            computed_mass,
        )

    if plot:
        ax = distribution.plot_pdf(plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot([lower, upper], [0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax, opt


def end_points_ints(lower, upper):
    return is_integer_num(lower) and is_integer_num(upper)


def is_integer_num(obj):
    if isinstance(obj, int):
        return True
    if isinstance(obj, float):
        return obj.is_integer()
    return False
