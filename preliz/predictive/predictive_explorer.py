from IPython.display import display

try:
    from ipywidgets import VBox, HBox, interactive_output
except ImportError:
    pass
from preliz.internal.parser import inspect_source, parse_function_for_pred_textboxes
from preliz.internal.plot_helper import get_textboxes, plot_decorator


def predictive_explorer(fmodel, samples=50, kind_plot="ecdf", references=None, plot_func=None):
    """
    Create textboxes and plot a set of samples returned by a function relating one or more
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
        The type of plot to display. Defaults to "kde". Options are "hist" (histogram),
        "kde" (kernel density estimate), "ecdf" (empirical cumulative distribution function).
    references : int, float, list, tuple or dictionary
        Value(s) used as reference points representing prior knowledge. For example expected
        values or values that are considered extreme. Use a dictionary for labeled references.
    plot_func : function
        Custom matplotlib code. Defaults to None. ``kind_plot`` and ``references`` are ignored
        if ``plot_func`` is specified.
    """
    source, signature = inspect_source(fmodel)

    model = parse_function_for_pred_textboxes(source, signature)
    textboxes = get_textboxes(signature, model)

    new_fmodel = plot_decorator(fmodel, samples, kind_plot, references, plot_func)

    out = interactive_output(new_fmodel, textboxes)
    default_names = ["__set_xlim__", "__x_min__", "__x_max__", "__resample__"]
    default_controls = [textboxes[name] for name in default_names]
    params_controls = [v for k, v in textboxes.items() if k not in default_names]

    params_plot = VBox(params_controls + [out])

    display(HBox([params_plot, VBox(default_controls)]))
