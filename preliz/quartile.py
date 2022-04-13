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

    distribution.check_boundaries(q1, q3)

    # Heuristic to approximate mean and standard deviation from quartiles
    mu_init = q2
    sigma_init = (q3 - q1) * 1.5
    normal_dist = Normal(mu_init, sigma_init)
    optimize_quartile(normal_dist, (q1, q2, q3))

    # I am doing one extra step for the normal!!!
    distribution.fit_moments(mean=normal_dist.mu, sigma=normal_dist.sigma)

    opt = optimize_quartile(distribution, (q1, q2, q3))

    r_error = relative_error(distribution, q1, q3, 0.5)

    if r_error > 0.01:
        _log.info(
            " The relative error between the requested and computed interval is %.2f",
            r_error,
        )

    if plot:
        ax = distribution.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("box"):
            cid = -4
        else:
            cid = -1
        ax.plot([q1, q2, q3], [0, 0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax, opt
