from preliz.ppls.pymc_io import if_pymc_get_preliz


def plot(
    dist,
    kind="pdf",
    moments=None,
    marginals=True,
    pointinterval=False,
    interval=None,
    levels=None,
    support="restricted",
    baseline=True,
    legend="legend",
    figsize=None,
    ax=None,
    **kwargs,
):
    """
    Plot distribution functions, including pdf, cdf, ppf, sf and isf.

    Parameters
    ----------
    kind : str
        Specify which distribution function to plot. Defaults to ``pdf``, i.e. the probability
        density function for continuos distributions or probability mass function for discrete
        distributions.
        Others possible values are:
        * ``cdf`` cumulative distribution function
        * ``ppf`` percent point function, also know as inverse of the cdf
        * ``sf`` survival function (1-cdf)
        * ``isf`` inverse survival function
    moments : str
        Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
        for the mean (μ), standard deviation (σ), variance (σ²), skew (γ) or kurtosis (κ)
        respectively. Other strings will be ignored. Defaults to None. Ignored for multivariate
        distributions.
    marginals : bool
        Whether to plot the marginals of the distribution. Only used when plotting multivariate
        distributions and ``kind="pdf"``. Defaults to True. If False the joint distribution will be
        plotted when possible.
    pointinterval : bool
        Whether to include a plot of the quantiles. Defaults to False. If True the default is to
        plot the median and two interquantiles ranges.
    interval : str
        Type of interval. Available options are highest density interval `"hdi"` (default),
        equal tailed interval `"eti"` or intervals defined by arbitrary `"quantiles"`.
        Defaults to the value in rcParams["stats.ci_kind"].
    levels : list
        Mass of the intervals. For hdi or eti the number of elements should be 2 or 1.
        For quantiles the number of elements should be 5, 3, 1 or 0
        (in this last case nothing will be plotted).
    support : str:
        If ``full`` use the finite end-points to set the limits of the plot. For unbounded
        end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        Ignored when ``kind="ppf"``, ``kind="isf"``.
    baseline : bool
        Whether to include a horizontal line at y=0. Only used when ``kind="pdf"``, ignored
        otherwise.
    legend : str
        Whether to include a string with the distribution and its parameter as a ``"legend"`` a
        ``"title"`` or not include them ``None``.
    figsize : tuple
        Size of the figure
    ax : matplotlib axes
    kwargs : keyword arguments
        Additional keyword arguments passed to matplotlib plot function.
        For example, ``color``, ``alpha``, ``linewidth``, etc.
    """
    if kind not in ["pdf", "cdf", "ppf", "sf", "isf"]:
        raise ValueError(f"kind should be one of {['pdf', 'cdf', 'ppf', 'sf', 'isf']}")

    dist = if_pymc_get_preliz(dist)

    mandatory_kwargs = {
        "moments": moments,
        "pointinterval": pointinterval,
        "interval": interval,
        "levels": levels,
        "support": support,
        "baseline": baseline,
        "legend": legend,
        "figsize": figsize,
        "ax": ax,
    }

    if kind != "pdf":
        mandatory_kwargs.pop("baseline")

    if kind in ["ppf", "isf"]:
        mandatory_kwargs.pop("support")

    if dist.__class__.__name__ in ["MvNormal", "Dirichlet"]:
        mandatory_kwargs.pop("moments")
        if kind == "pdf":
            mandatory_kwargs["marginals"] = marginals

    return getattr(dist, f"plot_{kind}")(**mandatory_kwargs, **kwargs)
