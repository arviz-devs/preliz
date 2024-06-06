try:
    import ipywidgets as widgets
except ImportError:
    pass

from preliz.internal.optimization import fit_to_quartile
from ..internal.plot_helper import (
    create_figure,
    check_inside_notebook,
    representations,
    reset_dist_panel,
)
from ..internal.distribution_helper import process_extra


def quartile_int(q1=1, q2=2, q3=3, dist_names=None, figsize=None):
    """
    Prior elicitation for 1D distributions from quartiles.

    Parameters
    ----------
    q1 : float
        First quartile, i.e 0.25 of the mass is below this point.
    q2 : float
        Second quartile, i.e 0.50 of the mass is below this point. This is also know
        as the median.
    q3 : float
        Third quartile, i.e 0.75 of the mass is below this point.
    dist_names: list
        List of distributions names to be used in the elicitation. If None, almost all 1D
        distributions available in PreliZ will be used. Some distributions like Uniform or
        Cauchy are omitted by default.
    figsize: Optional[Tuple[int, int]]
        Figure size. If None it will be defined automatically.

    Note
    ----
    Use the `params` text box to parametrize distributions, for instance write
    `BetaScaled(lower=-1, upper=10)` to specify the upper and lower bounds of BetaScaled
    distribution. To parametrize more that one distribution use commas for example
    `StudentT(nu=3), TruncatedNormal(lower=-2, upper=inf)`

    References
    ----------
    * Morris D.E. et al. (2014) see https://doi.org/10.1016/j.envsoft.2013.10.010
    * See quartile mode http://optics.eee.nottingham.ac.uk/match/uncertainty.php
    """

    check_inside_notebook(need_widget=True)

    w_q1, w_q2, w_q3, w_extra, w_repr, w_distributions = get_widgets(q1, q2, q3, dist_names)

    output = widgets.Output()

    with output:

        if figsize is None:
            figsize = (8, 6)

        fig, ax_fit = create_figure(figsize)

        def match_distribution_(_):
            match_distribution(
                fig.canvas,
                w_distributions.value,
                w_repr.value,
                w_q1.value,
                w_q2.value,
                w_q3.value,
                w_extra.value,
                ax_fit,
            )

        w_repr.observe(match_distribution_)
        w_distributions.observe(match_distribution_)
        w_q1.observe(match_distribution_)
        w_q2.observe(match_distribution_)
        w_q3.observe(match_distribution_)

        fig.canvas.mpl_connect(
            "button_release_event",
            lambda event: match_distribution(
                fig.canvas,
                w_distributions.value,
                w_repr.value,
                w_q1.value,
                w_q2.value,
                w_q3.value,
                w_extra.value,
                ax_fit,
            ),
        )

    controls = widgets.VBox([w_q1, w_q2, w_q3, w_extra])

    display(widgets.HBox([controls, w_repr, w_distributions]))  # pylint:disable=undefined-variable


def match_distribution(canvas, dist_names, kind_plot, q1, q2, q3, extra, ax):
    q1 = float(q1)
    q2 = float(q2)
    q3 = float(q3)
    extra_pros = process_extra(extra)
    fitted_dist = None

    if q1 < q2 < q3:
        reset_dist_panel(ax, yticks=False)

        fitted_dist = fit_to_quartile(dist_names, q1, q2, q3, extra_pros)

        if fitted_dist is None:
            ax.set_title("domain error")
        else:
            representations(fitted_dist, kind_plot, ax)
    else:
        reset_dist_panel(ax, yticks=True)
        ax.set_title("quantiles must follow the order: q1 < q2 < q3 ")

    canvas.draw()

    return fitted_dist


def get_widgets(q1, q2, q3, dist_names=None):

    width_entry_text = widgets.Layout(width="150px")
    width_repr_text = widgets.Layout(width="250px")
    width_distribution_text = widgets.Layout(width="150px", height="125px")

    w_q1 = widgets.FloatText(
        value=q1,
        step=0.1,
        description="q1",
        disabled=False,
        layout=width_entry_text,
    )

    w_q2 = widgets.FloatText(
        value=q2,
        step=0.1,
        description="q2",
        disabled=False,
        layout=width_entry_text,
    )

    w_q3 = widgets.FloatText(
        value=q3,
        step=0.1,
        description="q3",
        disabled=False,
        layout=width_entry_text,
    )

    w_extra = widgets.Textarea(
        value="",
        placeholder="Pass extra parameters",
        description="params:",
        disabled=False,
        layout=width_repr_text,
    )

    w_repr = widgets.RadioButtons(
        options=["pdf", "cdf", "ppf"],
        value="pdf",
        description="",
        disabled=False,
        layout=width_repr_text,
    )

    if dist_names is None:

        default_dist = ["Normal", "BetaScaled", "Gamma", "LogNormal", "StudentT"]

        dist_names = [
            "AsymmetricLaplace",
            "BetaScaled",
            "ChiSquared",
            "ExGaussian",
            "Exponential",
            "Gamma",
            "Gumbel",
            "HalfNormal",
            "HalfStudentT",
            "InverseGamma",
            "Laplace",
            "LogNormal",
            "Logistic",
            "LogitNormal",
            "Moyal",
            "Normal",
            "Pareto",
            "Rice",
            "SkewNormal",
            "StudentT",
            "Triangular",
            "VonMises",
            "Wald",
            "Weibull",
            "BetaBinomial",
            "DiscreteWeibull",
            "Geometric",
            "NegativeBinomial",
            "Poisson",
        ]
    else:
        default_dist = dist_names

    w_distributions = widgets.SelectMultiple(
        options=dist_names,
        value=default_dist,
        description="",
        disabled=False,
        layout=width_distribution_text,
    )

    return w_q1, w_q2, w_q3, w_extra, w_repr, w_distributions
