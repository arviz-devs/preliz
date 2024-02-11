import preliz as pz
import logging
from preliz.internal.optimization import optimize_one_iter

_log = logging.getLogger("preliz")


def one_iter(lower, upper, mode, mass=0.99, plot=False, plot_kwargs=None, ax=None):

    """ Fits Parameters to a Beta Distribution based on the mode, confidence intervals and mass of the distribution.

    Parameters
    -----------
    lower : float
        Lower end-point between 0 and upper.
    upper : float
        Upper end-point between lower and 1.
    mode : float
        Mode of the Beta distribution between lower and upper.
    mass : float
        Concentarion of the probabilty mass between lower and upper. Defaults to 0.99.
    plot : bool
        Whether to plot the distribution. Defaults to True.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    --------

    dist : Preliz Beta distribution.
        Beta distribution with fitted parameters alpha and beta for the given mass and intervals.

    """

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if mode < lower or mode > upper:
        raise ValueError("mode should be between lower and upper")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    dist = pz.Beta()
    dist._fit_moments(  # pylint:disable=protected-access
        mean=(upper + lower) / 2,
        sigma=((upper - lower) / 6) / mass
    )
    tau_a = (dist.alpha - 1) / mode
    tau_b = (dist.beta - 1) / (1 - mode)
    tau = (tau_a + tau_b) / 2
    prob = dist.cdf(upper) - dist.cdf(lower)

    prob, dist = optimize_one_iter(lower, upper, tau, mode, dist, mass, prob)

    relative_error = abs((prob - mass) / mass * 100)

    if relative_error > 0.005*100:
        _log.info(
            " The requested mass is %.3g, but the computed one is %.3g",
            mass,
            prob,
        )

    if plot:
        ax = dist.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot([lower, upper], [0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax, dist


