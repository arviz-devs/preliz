import warnings
import numpy as np

from preliz.distributions.normal import Normal
from preliz.internal.distribution_helper import valid_distribution
from preliz.internal.optimization import relative_error, optimize_quartile, get_fixed_params
from preliz.internal.rcparams import rcParams


def quartile(
    distribution=None,
    q1=-1,
    q2=0,
    q3=1,
    plot=None,
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
        Whether to plot the distribution, and lower and upper bounds. Defaults to None,
        which results in the value of rcParams["plots.show_plot"] being used.
    plot_kwargs : dict
        Dictionary passed to the method ``plot_pdf()`` of ``distribution``.
    ax : matplotlib axes

    Returns
    -------

    dict: dict with the parameters of the distribution
    axes: matplotlib axes (only if `plot=True`)

    Notes
    -----
    After calling this function the attribute `opt` of the distribution will be updated with the
    OptimizeResult object from the optimization step.

    See Also
    --------
    maxent : Find the maximum entropy distribution with a given mass inside a user defined interval.

    Examples
    --------
    Calculate the Gamma distribution with quartiles 3, 6 and 8:

    .. plot::
        :context: close-figs
        :include-source: true

        >>>
        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.quartile(pz.Gamma(), 3, 6, 8)

    Calculate the HalfStudentT T distribution with quartiles 2, 9 and 12
    and a value of nu=7:

    .. plot::
        :context: close-figs
        :include-source: true

        >>>
        >>> import preliz as pz
        >>> pz.style.use('preliz-doc')
        >>> pz.quartile(pz.HalfStudentT(nu=7), 2, 9, 12)

    """
    valid_distribution(distribution)

    if plot is None:
        plot = rcParams["plots.show_plot"]

    if plot_kwargs is None:
        plot_kwargs = {}

    if not q1 < q2 < q3:
        raise ValueError("The order of the quartiles should be q1 < q2 < q3")

    quartiles = np.array([q1, q2, q3])

    if distribution is None:
        distribution = Normal()

    if distribution.is_frozen:
        raise ValueError("All parameters are fixed, at least one should be free")

    distribution._check_endpoints(q1, q3)

    # Find which parameters has been fixed
    none_idx, fixed = get_fixed_params(distribution)

    # Heuristic to provide an initial guess for the optimization step
    # We obtain those guesses by first approximating the mean and standard deviation
    # from the quartiles and then use those values for moment matching
    distribution._fit_moments(mean=q2, sigma=(q3 - q1) / 1.35)  # pylint:disable=protected-access

    opt = optimize_quartile(distribution, quartiles, none_idx, fixed)

    distribution.opt = opt

    r_error, _ = relative_error(distribution, q1, q3, 0.5)

    if r_error > 0.01:
        computed_masses = distribution.cdf(quartiles).astype(float)
        warnings.warn(
            f"\nThe expected masses are 0.25, 0.5, 0.75\n"
            f"The computed ones are: {computed_masses[0]:.2g}, "
            f"{computed_masses[1]:.2g}, {computed_masses[2]:.2g}",
            stacklevel=2,
        )

    if plot:
        ax = distribution.plot_pdf(**plot_kwargs)
        if plot_kwargs.get("pointinterval"):
            cid = -4
        else:
            cid = -1
        ax.plot(quartiles, [0, 0, 0], "o", color=ax.get_lines()[cid].get_c(), alpha=0.5)
        return distribution, ax

    return distribution
