import warnings

import numpy as np

from preliz.distributions import Beta, Dirichlet
from preliz.internal.optimization import optimize_dirichlet_mode
from preliz.internal.rcparams import rcParams


def dirichlet_mode(mode, mass=None, bound=0.01, plot=None, plot_kwargs=None, ax=None):
    """
    Elicitate a Dirichlet distribution with a given mode and mass.

    Computes a Dirichlet distribution where the marginals have the specified mode
    and mass and their masses lie within the range mode ± bound
    (Adapted from :footcite:t:`Michael2017`).

    Parameters
    ----------
    mode : list
        Mode of the Dirichlet distribution.
    mass : float
        Probability mass between within mode +- bounds. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.
    bound : float
        Defines upper and lower bounds for the mass as mode +- bound. Defaults to 0.01.
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

    .. footbibliography::

    """
    if mass is None:
        mass = rcParams["stats.ci_prob"]

    if plot is None:
        plot = rcParams["plots.show_plot"]

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if not all(i > 0 for i in mode):
        raise ValueError("mode should be larger than 0")

    if not abs(sum(mode) - 1) < 0.0001:
        warnings.warn("The mode should sum to 1, normalising mode to sum to 1")
        sum_mode = sum(mode)
        mode = [i / sum_mode for i in mode]

    if plot_kwargs is None:
        plot_kwargs = {}

    lower_bounds = np.clip(np.array(mode) - bound, 0, 1)
    target_mass = (1 - mass) / 2
    _dist = Beta()

    _, alpha = optimize_dirichlet_mode(lower_bounds, mode, target_mass, _dist)

    alpha_np = np.array(alpha)
    calculated_mode = (alpha_np - 1) / (alpha_np.sum() - len(alpha_np))

    if np.any((np.array(mode) - calculated_mode) > 0.01):
        warnings.warn(
            f"The requested mode {mode} is different from the calculated mode {calculated_mode}."
        )

    dirichlet_distribution = Dirichlet(alpha)

    if plot:
        ax = dirichlet_distribution.plot_pdf(**plot_kwargs, ax=ax)

    return ax, dirichlet_distribution
