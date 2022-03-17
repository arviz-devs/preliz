import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .utils.constraints_utils import (
    get_parametrization,
    check_boundaries,
    relative_error,
    sane_scipy,
    compute_xvals,
    optimize,
    method_of_moments,
)


def constraints(
    name="normal",
    lower=-1,
    upper=1,
    mass=0.90,
    extra=None,
    parametrization="pymc",
    plot=True,
    figsize=(10, 4),
    ax=None,
):
    """
    Find parameters for a given distribution with `mass` between `lower` and `upper`.

    Parameters
    ----------
    name : str
        Name of the distribution to use as prior
    lower : float
        Lower bound
    upper: float
        Upper bound
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

    check_boundaries(name, lower, upper)

    # Use least squares assuming a Gaussian
    opt = optimize(lower=lower, upper=upper, mass=mass)
    a, b = opt["x"]
    if name == "normal":
        dist = stats.norm
    else:
        # Use mu (a) and sigma (b) values to obtain parameters for distribution other than normal
        a, b, dist = method_of_moments(name, a, b)
        # Use least squares given a distribution
        opt = optimize(lower=lower, upper=upper, mass=mass, dist=dist, a=a, b=b, extra=extra)
        a, b = opt["x"]
    rv_frozen = sane_scipy(dist, a, b, extra)

    if plot:
        r_error = relative_error(rv_frozen, upper, lower, mass)
        x = compute_xvals(rv_frozen)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        color = next(ax._get_lines.prop_cycler)["color"]  # pylint: disable=protected-access
        ax.plot([lower, upper], [0, 0], "o", color=color, alpha=0.5)
        title = get_parametrization(name, a, b, extra, parametrization)
        subtitle = f"relative error = {r_error:.2f}"
        ax.plot(x, rv_frozen.pdf(x), label=title + "\n" + subtitle, color=color)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_yticks([])

    return ax, rv_frozen, opt
