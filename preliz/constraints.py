
import matplotlib.pyplot as plt

from .distributions import Normal
from .utils.constraints_utils import relative_error


def constraints(
    distribution=None,
    lower=-1,
    upper=1,
    mass=0.90,
    #parametrization="pymc",
    plot=True,
    figsize=(10, 4),
    ax=None,
):
    """
    Find parameters for a given distribution with `mass` in the interval defined by `lower` and
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
        Whether to plot the distribution, and lower and upper bounds. Defautls to True.
    figsize : tuple
        size of the figure when ``plot=True``
    ax : matplotlib axes

    Returns
    -------

    axes: matplotlib axes
    rv_frozen : scipy.stats.distributions.rv_frozen
        Notice that the returned rv_frozen object always use the scipy parametrization,
        irrespective of the value of `parametrization` argument.
        Unlike standard rv_frozen objects this one has a name attribute
    opt : scipy.optimize.OptimizeResult
        Represents the optimization result.
    """
    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    if distribution is None:
        distribution = Normal()

    distribution._check_boundaries(lower, upper)

    # Use least squares assuming a Gaussian
    mu_init = (lower + upper) / 2
    sigma_init = ((upper - lower) / 4) / mass
    normal_dist = Normal(mu_init, sigma_init)
    normal_dist._optimize(lower, upper, mass)

    # I am doing one extra step for the normal!!!
    distribution.fit_moments(mean=normal_dist.mu, sigma=normal_dist.sigma)
    distribution._optimize(lower, upper, mass)

    if plot:
        r_error = relative_error(distribution.rv_frozen, upper, lower, mass)
        x = distribution._xvals()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = next(ax._get_lines.prop_cycler)["color"]  # pylint: disable=protected-access
        ax.plot([lower, upper], [0, 0], "o", color=color, alpha=0.5)
        title = distribution.__repr__()
        subtitle = f"relative error = {r_error:.2f}"
        ax.plot(x, distribution.rv_frozen.pdf(x), label=title + "\n" + subtitle, color=color)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_yticks([])

    return ax
