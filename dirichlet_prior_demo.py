import logging
from preliz.distributions import Dirichlet, Beta
import numpy as np
from preliz.internal.optimization import optimize_dirichlet_mode


_log = logging.getLogger("preliz")


def dirichlet_mode(mode, mass=0.90, bound=0.01, plot=True, plot_kwargs={}, ax=None):
    """

    This function returns a Dirichlet distribution that has the specified mass concentrated in the region of
    mode +- bound.

    Parameters
    ----------

    mode : list
        Mode of the Dirichlet distribution.

    mass : float
        Probability mass between ``lower`` and ``upper`` bounds. Defaults to 0.90.

    bound : float
        Defines upper and lower bounds for the mass as mode+-bound. Defaults to 0.01.

    plot : bool
        Whether to plot the distribution. Defaults to True.

    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()``.

    ax : matplotlib axes

    Returns
    -------

    ax : matplotlib axes

    dist : Preliz Dirichlet distribution.
        Dirichlet distribution with fitted parameters alpha for the given mass and intervals.

    References
    ----------
    Adapted from  Evans et al. (2017) see https://doi.org/10.3390/e19100564

    """

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if not all(i > 0 for i in mode):
        raise ValueError("mode should be larger than 0")

    if not abs(sum(mode) - 1) < 0.0001:
        _log.warning("The mode should sum to 1, normalising mode to sum to 1")
        sum_mode = sum(mode)
        mode = [i / sum_mode for i in mode]

    lower_bounds = np.clip(np.array(mode) - bound, 0, 1)
    target_mass = (1 - mass) / 2
    _dist = Beta()


    new_prob, alpha = optimize_dirichlet_mode(
        lower_bounds, mode, target_mass, _dist
    )

    alpha_np = np.array(alpha)
    calculated_mode = (alpha_np - 1) / (alpha_np.sum() - len(alpha_np))

    if np.linalg.norm(np.array(mode) - calculated_mode) > 0.01:
        _log.warning(
            f"The requested mode {mode} is different from the calculated mode {calculated_mode}."
        )

    dirichlet_distribution = Dirichlet(alpha)

    if plot:
        ax = dirichlet_distribution.plot_pdf(**plot_kwargs)

    return ax, dirichlet_distribution


