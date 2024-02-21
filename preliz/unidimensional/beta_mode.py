import logging

from ..distributions import Beta
from ..internal.optimization import optimize_beta_mode

_log = logging.getLogger("preliz")


def beta_mode(lower, upper, mode, mass=0.94, plot=True, plot_kwargs=None, ax=None):
    """Fits Parameters to a Beta Distribution based on the mode, confidence intervals
    and mass of the distribution.

    Parameters
    -----------
    lower : float
        Lower end-point between 0 and upper.
    upper : float
        Upper end-point between lower and 1.
    mode : float
        Mode of the Beta distribution between lower and upper.
    mass : float
        Probability mass between ``lower`` and ``upper`` bounds. Defaults to 0.94
    plot : bool
        Whether to plot the distribution. Defaults to True.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()``.
    ax : matplotlib axes

    Returns
    --------
    dist : Preliz Beta distribution.
        Beta distribution with fitted parameters alpha and beta for the given mass and intervals.

    See also
    --------
    maxent : maximum entropy distribution with mass in the lower-upper interval.

    References
    ----------
    Adapted from  Evans et al. (2017) see https://doi.org/10.3390/e19100564
    """

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if mode <= lower or mode >= upper:
        raise ValueError("mode should be between lower and upper")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    if plot_kwargs is None:
        plot_kwargs = {}

    dist = Beta()
    dist._fit_moments(  # pylint:disable=protected-access
        mean=(upper + lower) / 2, sigma=((upper - lower) / 4) / mass
    )
    tau_a = (dist.alpha - 1) / mode
    tau_b = (dist.beta - 1) / (1 - mode)
    tau = (tau_a + tau_b) / 2
    prob = dist.cdf(upper) - dist.cdf(lower)

    optimize_beta_mode(lower, upper, tau, mode, dist, mass, prob)

    if plot:
        ax = dist.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot([lower, upper], [0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)

    return ax, dist
