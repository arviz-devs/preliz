import logging

import numpy as np

from ..distributions import Normal
from ..internal.distribution_helper import valid_distribution
from ..internal.optimization import relative_error, optimize_quartile, get_fixed_params

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
    Find the distribution with the specified quartiles.

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

    See Also
    --------
    maxent : Find the maximum entropy distribution with a given mass inside a user defined interval.

    Examples
    --------
    Calculate the Gamma distribution with quartiles 3, 6 and 8:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import arviz as az
        >>> import preliz as pz
        >>> az.style.use('arviz-doc')
        >>> pz.quartile(pz.Gamma(), 3, 6, 8)

    Calculate the HalfStudentT T distribution with quartiles 2, 9 and 12
    and a value of nu=7:

    .. plot::
        :context: close-figs
        :include-source: true

        >>> import arviz as az
        >>> import preliz as pz
        >>> az.style.use('arviz-doc')
        >>> pz.quartile(pz.HalfStudentT(nu=7), 2, 9, 12)

    """
    valid_distribution(distribution)

    if plot_kwargs is None:
        plot_kwargs = {}

    if not q1 < q2 < q3:
        raise ValueError("The order of the quartiles should be q1 < q2 < q3")

    quartiles = np.array([q1, q2, q3])

    if distribution is None:
        distribution = Normal()

    distribution._check_endpoints(q1, q3)

    # Find which parameters has been fixed
    none_idx, fixed = get_fixed_params(distribution)

    # Heuristic to provide an initial guess for the optimization step
    # We obtain those guesses by first approximating the mean and standard deviation
    # from the quartiles and then use those values for moment matching
    distribution._fit_moments(mean=q2, sigma=(q3 - q1) / 1.35)  # pylint:disable=protected-access

    opt = optimize_quartile(distribution, quartiles, none_idx, fixed)

    r_error, _ = relative_error(distribution, q1, q3, 0.5)

    if r_error > 0.01:
        _log.info(
            "The expected masses are 0.25, 0.5, 0.75\n The computed ones are: %.2g, %.2g, %.2g",
            *distribution.cdf(quartiles)
        )

    if plot:
        ax = distribution.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot(quartiles, [0, 0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
    return ax, opt
