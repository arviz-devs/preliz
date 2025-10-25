from preliz.ppls.pymc_io import from_pymc


def plot(
    dist,
    kind="pdf",
    moments=None,
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
    if kind not in ["pdf", "cdf", "ppf", "sf", "isf"]:
        raise ValueError(f"kind should be one of {['pdf', 'cdf', 'ppf', 'sf', 'isf']}")

    if dist.__class__.__name__ == "TensorVariable":
        dist = from_pymc(dist)

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

    return getattr(dist, f"plot_{kind}")(**mandatory_kwargs, **kwargs)
