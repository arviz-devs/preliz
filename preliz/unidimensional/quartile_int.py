try:
    import ipywidgets as widgets
except ImportError:
    pass

from preliz.internal.optimization import fit_to_quartile
from preliz.internal.plot_helper import (
    create_figure,
    check_inside_notebook,
    representations,
    reset_dist_panel,
)
from preliz.internal.distribution_helper import process_extra, get_distributions


class QuartileInt:
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

    def __init__(self, q1=1, q2=2, q3=3, dist_names=None, figsize=None):
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3
        self.dist = None
        self._dist_names = dist_names
        self._figsize = figsize

        check_inside_notebook(need_widget=True)
        self._widgets = self._get_widgets()

        self._output = widgets.Output()

        with self._output:
            if self._figsize is None:
                self._figsize = (8, 6)

            self._fig, self._ax_fit = create_figure(self._figsize)
            self._setup_observers()

            self._fig.canvas.mpl_connect(
                "button_release_event",
                lambda event: self._match_distribution(),
            )

        self._match_distribution()
        controls = widgets.VBox(
            [
                self._widgets["w_q1"],
                self._widgets["w_q2"],
                self._widgets["w_q3"],
                self._widgets["w_extra"],
            ]
        )
        display(  # pylint:disable=undefined-variable
            widgets.HBox([controls, self._widgets["w_repr"], self._widgets["w_distributions"]])
        )

    def _get_widgets(self):
        width_entry_text = widgets.Layout(width="150px")
        width_repr_text = widgets.Layout(width="250px")
        width_distribution_text = widgets.Layout(width="150px", height="125px")

        w_q1 = widgets.FloatText(
            value=self._q1,
            step=0.1,
            description="q1",
            disabled=False,
            layout=width_entry_text,
        )

        w_q2 = widgets.FloatText(
            value=self._q2,
            step=0.1,
            description="q2",
            disabled=False,
            layout=width_entry_text,
        )

        w_q3 = widgets.FloatText(
            value=self._q3,
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

        if self._dist_names is None:

            default_dist = ["Normal", "BetaScaled", "Gamma", "LogNormal", "StudentT"]

            self._dist_names = [
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
            default_dist = self._dist_names

        w_distributions = widgets.SelectMultiple(
            options=self._dist_names,
            value=default_dist,
            description="",
            disabled=False,
            layout=width_distribution_text,
        )

        return {
            "w_q1": w_q1,
            "w_q2": w_q2,
            "w_q3": w_q3,
            "w_extra": w_extra,
            "w_repr": w_repr,
            "w_distributions": w_distributions,
        }

    def _match_distribution(self):
        q1 = float(self._widgets["w_q1"].value)
        q2 = float(self._widgets["w_q2"].value)
        q3 = float(self._widgets["w_q3"].value)
        extra_pros = process_extra(self._widgets["w_extra"].value)
        fitted_dist = None

        if q1 < q2 < q3:
            reset_dist_panel(self._ax_fit, yticks=False)

            fitted_dist = fit_to_quartile(
                get_distributions(self._widgets["w_distributions"].value), q1, q2, q3, extra_pros
            )

            if fitted_dist is None:
                self._ax_fit.set_title("domain error")
            else:
                representations(fitted_dist, self._widgets["w_repr"].value, self._ax_fit)
        else:
            reset_dist_panel(self._ax_fit, yticks=True)
            self._ax_fit.set_title("quantiles must follow the order: q1 < q2 < q3 ")

        self._fig.canvas.draw()

        self.dist = fitted_dist

    def _setup_observers(self):
        def _match_distribution_(_):
            self._match_distribution()

        self._widgets["w_repr"].observe(_match_distribution_)
        self._widgets["w_distributions"].observe(_match_distribution_)
        self._widgets["w_q1"].observe(_match_distribution_)
        self._widgets["w_q2"].observe(_match_distribution_)
        self._widgets["w_q3"].observe(_match_distribution_)
