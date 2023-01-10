import inspect

from ipywidgets import interactive
from preliz.internal.parser import parse_function
from preliz.internal.plot_helper import get_sliders, plot_decorator


def predictive_sliders(fmodel, samples=50, kind_plot="hist"):
    """
    Create sliders and plot a set of samples returned by a function relating one or more
    PreliZ distributions.

    Use this function to interactively explore how a prior predictive distribution changes when the
    priors are changed.

    Parameters
    ----------
    fmodel : callable
        A function with PreliZ distributions. The distributions should call their rvs methods.
    samples : int, optional
        The number of samples to draw from the prior predictive distribution (default is 50).
    kind_plot : str, optional
        The type of plot to display. Defaults to "hist". Options are "hist" (histogram),
        "kde" (kernel density estimate), "ecdf" (empirical cumulative distribution function),
        or None (no plot).
    """
    signature = inspect.signature(fmodel)
    source = inspect.getsource(fmodel)

    model = parse_function(source, signature)
    print(model)
    sliders = get_sliders(signature, model)

    if kind_plot is None:
        new_fmodel = fmodel
    else:
        new_fmodel = plot_decorator(fmodel, samples, kind_plot)

    return interactive(new_fmodel, **sliders)
