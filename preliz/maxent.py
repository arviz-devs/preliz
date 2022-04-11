import logging

from .distributions import Normal
from .utils.maxent_utils import relative_error, end_points_ints, optimize_max_ent

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
        Instance of a PreliZ distribution
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

    distribution._check_boundaries(lower, upper)

    if distribution.kind == "discrete":
        if not end_points_ints(lower, upper):
            _log.info(
                "%s distribution is discrete, but the provided bounds are not integers",
                distribution.name.capitalize(),
            )

    # Heuristic to approximate mean and standard deviation from intervals and mass
    mu_init = (lower + upper) / 2
    sigma_init = ((upper - lower) / 4) / mass
    normal_dist = Normal(mu_init, sigma_init)
    optimize_max_ent(normal_dist, lower, upper, mass)

    # I am doing one extra step for the normal!!!
    distribution.fit_moments(mean=normal_dist.mu, sigma=normal_dist.sigma)

    optimize_max_ent(distribution, lower, upper, mass)

    r_error = relative_error(distribution, lower, upper, mass)

    if r_error > 0.01:
        _log.info(
            " The relative error between the requested and computed interval is %.2f",
            r_error,
        )

    if plot:
        ax = distribution.plot_pdf(plot_kwargs)
        if plot_kwargs.get("box"):
            cid = -4
        else:
            cid = -1
        ax.plot([lower, upper], [0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax
