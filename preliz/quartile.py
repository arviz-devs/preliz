import logging

from .distributions import Normal
from .utils.optimization import relative_error, optimize_quartile

_log = logging.getLogger("preliz")


def quartile(
    distribution=None,
    q1=-1,
    q2=0,
    q3=1,
    plot=True,
    plot_kwargs=None,
    ax=None,
):
    """
    Find the maximum entropy distribution with `mass` in the interval defined by the `lower` and
    `upper` end-points.

    Parameters
    ----------
    distribution : PreliZ distribution
        Instance of a PreliZ distribution
    q1 : float
        First quartile, i.e 0.25 of the mass is below this point.
    q2 : float
        Second quartile, i.e 0.50 of the mass is below this point. This is also know
        as the median.
    q3 : float
        Third quartile, i.e 0.75 of the mass is below this point.
    plot : bool
        Whether to plot the distribution, and lower and upper bounds. Defaults to True.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------

    axes: matplotlib axes
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if not q1 < q2 < q3:
        raise ValueError("The order of the quartiles should be q1 < q2 < q3")

    if distribution is None:
        distribution = Normal()

    distribution._check_endpoints(q1, q3)

    # Heuristic to provide an initial guess for the optimization step
    # We obtain those guesses by first approximating the mean and standard deviation
    # from the quartiles and then use those values for moment matching
    distribution._fit_moments(mean=q2, sigma=(q3 - q1) * 1.5)  # pylint:disable=protected-access

    opt = optimize_quartile(distribution, (q1, q2, q3))

    r_error, computed_mass = relative_error(distribution, q1, q3, 0.5)

    if r_error > 0.01:
        _log.info(
            "The requested mass in the interval (q1=%.2g - q3=%.2g) is 0.5, "
            "but the computed one is %.2g",
            q1,
            q3,
            computed_mass,
        )

    if plot:
        ax = distribution.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot([q1, q2, q3], [0, 0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax, opt
