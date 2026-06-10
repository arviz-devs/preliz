"""Distribution catalog for exploring and querying PreliZ distributions."""

from sys import modules

import numpy as np

from preliz.distributions import (
    all_continuous,
    all_continuous_multivariate,
    all_discrete,
    all_modifiers,
)
from preliz.distributions.distributions import _format_support

_GROUPS = {
    "continuous": all_continuous,
    "discrete": all_discrete,
    "continuous_multivariate": all_continuous_multivariate,
    "unbounded": [
        "AsymmetricLaplace",
        "Cauchy",
        "ExGaussian",
        "Gumbel",
        "Laplace",
        "Logistic",
        "Moyal",
        "Normal",
        "SkewNormal",
        "SkewStudentT",
        "StudentT",
    ],
    "positive": [
        "ChiSquared",
        "Exponential",
        "Gamma",
        "HalfCauchy",
        "HalfNormal",
        "HalfStudentT",
        "InverseGamma",
        "LogLogistic",
        "LogNormal",
        "Pareto",
        "Rice",
        "Wald",
        "Weibull",
    ],
    "bounded": [
        "Beta",
        "BetaScaled",
        "Kumaraswamy",
        "LogitNormal",
        "Triangular",
        "TruncatedNormal",
        "Uniform",
        "VonMises",
    ],
    "non_negative_continuous": [
        "ChiSquared",
        "Exponential",
        "Gamma",
        "HalfCauchy",
        "HalfNormal",
        "HalfStudentT",
        "InverseGamma",
        "LogLogistic",
        "LogitNormal",
        "LogNormal",
        "Pareto",
        "Rice",
        "ScaledInverseChiSquared",
        "Wald",
        "Weibull",
    ],
    "non_negative_discrete": [
        "Bernoulli",
        "Binomial",
        "DiscreteWeibull",
        "Geometric",
        "NegativeBinomial",
        "Poisson",
        "ZeroInflatedBinomial",
        "ZeroInflatedNegativeBinomial",
        "ZeroInflatedPoisson",
    ],
    "bounded_discrete": [
        "BetaBinomial",
        "Binomial",
        "Categorical",
        "DiscreteUniform",
        "Hypergeometric",
        "ZeroInflatedBinomial",
    ],
    "multivariate": [
        "Dirichlet",
        "MultivariateNormal",
    ],
    "symmetric": [
        "Beta",
        "BetaScaled",
        "Cauchy",
        "DiscreteUniform",
        "Laplace",
        "Logistic",
        "MultivariateNormal",
        "Normal",
        "StudentT",
        "Uniform",
        "VonMises",
    ],
    "asymmetric": [
        "AsymmetricLaplace",
        "DiscreteWeibull",
        "ExGaussian",
        "Exponential",
        "Gamma",
        "Geometric",
        "Gumbel",
        "InverseGamma",
        "Kumaraswamy",
        "LogLogistic",
        "LogitNormal",
        "Moyal",
        "Pareto",
        "Rice",
        "ScaledInverseChiSquared",
        "SkewNormal",
        "SkewStudentT",
        "Wald",
        "Weibull",
    ],
    "heavy_tailed": [
        "Cauchy",
        "HalfCauchy",
        "HalfStudentT",
        "InverseGamma",
        "LogLogistic",
        "LogNormal",
        "Pareto",
        "SkewStudentT",
        "StudentT",
    ],
    "light_tailed": [
        "AsymmetricLaplace",
        "ChiSquared",
        "ExGaussian",
        "Exponential",
        "Gamma",
        "HalfNormal",
        "Laplace",
        "Logistic",
        "Moyal",
        "Normal",
        "Rice",
        "SkewNormal",
        "Triangular",
        "TruncatedNormal",
        "Wald",
        "Weibull",
    ],
    "zero_inflated": [
        "ZeroInflatedBinomial",
        "ZeroInflatedNegativeBinomial",
        "ZeroInflatedPoisson",
    ],
    "extreme_value": [
        "Gumbel",
        "LogLogistic",
    ],
    "circular": [
        "VonMises",
    ],
    "binary": [
        "Bernoulli",
        "Binomial",
    ],
    "count": [
        "DiscreteWeibull",
        "Geometric",
        "NegativeBinomial",
        "Poisson",
        "ZeroInflatedNegativeBinomial",
        "ZeroInflatedPoisson",
    ],
}


def _get_dist_class(name):
    return getattr(modules["preliz.distributions"], name)


class DistributionCatalog:
    """Registry for accessing PreliZ distributions.

    Provides methods to list, filter, and inspect distributions.

    Examples
    --------
    List all distributions::

        >>> pz.dists

    Get instances by category::

        >>> pz.catalog.get("continuous")
        >>> pz.catalog.get("positive")

    Get distribution names::

        >>> pz.catalog.names()
        >>> pz.catalog.names("discrete")

    Get info about a specific distribution::

        >>> pz.catalog.info("Gamma")

    Filter distributions by properties::

        >>> pz.catalog.find(kind="continuous", num_params=2)
    """

    def __repr__(self):
        continuous_names = [d.__name__ for d in all_continuous]
        discrete_names = [d.__name__ for d in all_discrete]
        multivariate_names = [d.__name__ for d in all_continuous_multivariate]
        modifiers = [d.__name__ for d in all_modifiers]

        lines = ["PreliZ Distributions", "=" * 50]
        lines.append(f"Continuous ({len(continuous_names)}):")
        lines.append("  " + ", ".join(continuous_names))
        lines.append(f"\nDiscrete ({len(discrete_names)}):")
        lines.append("  " + ", ".join(discrete_names))
        lines.append(f"\nMultivariate ({len(multivariate_names)}):")
        lines.append("  " + ", ".join(multivariate_names))
        lines.append(f"\nModifiers ({len(modifiers)}):")
        lines.append("  " + ", ".join(modifiers))
        return "\n".join(lines)

    def _repr_html_(self):
        continuous_names = [d.__name__ for d in all_continuous]
        discrete_names = [d.__name__ for d in all_discrete]
        multivariate_names = [d.__name__ for d in all_continuous_multivariate]
        modifiers = [d.__name__ for d in all_modifiers]

        html = ["<div style='font-family: monospace'>"]
        html.append("<b>PreliZ Distributions</b><br><br>")
        html.append(f"<b>Continuous</b> ({len(continuous_names)}): ")
        html.append(", ".join(continuous_names) + "<br><br>")
        html.append(f"<b>Discrete</b> ({len(discrete_names)}): ")
        html.append(", ".join(discrete_names) + "<br><br>")
        html.append(f"<b>Multivariate</b> ({len(multivariate_names)}): ")
        html.append(", ".join(multivariate_names) + "<br><br>")
        html.append(f"<b>Modifiers</b> ({len(modifiers)}): ")
        html.append(", ".join(modifiers))
        html.append("</div>")

        return "".join(html)

    def get(self, category="continuous", output="instances"):
        """Return a list of uninitialized PreliZ distribution instances by category.

        Parameters
        ----------
        category : str
            Category of distributions to return. One of:
            - ``"continuous"``: All univariate continuous distributions.
            - ``"discrete"``: All discrete distributions.
            - ``"continuous_multivariate"``: All continuous multivariate distributions.
            - ``"positive"``: Continuous distributions on the positive reals.
            - ``"unbounded"``: Continuous distributions on the full real line.
            - ``"bounded"``: Continuous distributions on a finite interval.
            - ``"non_negative"``: All non-negative distributions.
            - ``"non_negative_continuous"``: Continuous distributions on [0, inf).
            - ``"non_negative_discrete"``: Discrete distributions on {0, 1, 2, ...}.
            - ``"bounded_discrete"``: Discrete distributions on a finite interval.
            - ``"multivariate"``: All multivariate distributions.
            - ``"symmetric"``: Distributions that are symmetric.
            - ``"asymmetric"``: Distributions with skewed shapes.
            - ``"heavy_tailed"``: Distributions with slowly decaying tails.
            - ``"light_tailed"``: Distributions with quickly decaying tails.
            - ``"zero_inflated"``: Discrete distributions with extra zeros.
            - ``"extreme_value"``: Distributions for extreme events.
            - ``"circular"``: Distributions on a circular domain.
            - ``"binary"``: Distributions for binary outcomes.
            - ``"count"``: Discrete distributions for count data.

        output : str
            Whether to return distribution instances ("instances") or names ("names").
            Defaults to "instances".

        Returns
        -------
        list of PreliZ distribution instances
        """
        if output not in ["instances", "names"]:
            raise ValueError("Invalid value for 'output'. Must be 'instances' or 'names'.")

        if output == "instances":
            group = _GROUPS.get(category)
            if group is None:
                raise ValueError(
                    f"Unknown category '{category}'. "
                    f"Must be one of: {', '.join(repr(k) for k in _GROUPS)}"
                )
            if category in ("continuous", "discrete", "continuous_multivariate"):
                return [d() for d in group]
            return [_get_dist_class(name)() for name in group]

        else:
            group = _GROUPS.get(category)
            if group is None:
                raise ValueError(
                    f"Unknown category '{category}'. "
                    f"Must be one of: {', '.join(repr(k) for k in _GROUPS)}"
                )
            if category in ("continuous", "discrete", "continuous_multivariate"):
                return [d.__name__ for d in group]
            return list(group)

    def info(self, name):
        """Return metadata about a distribution.

        Parameters
        ----------
        name : str
            Name of the distribution (e.g., "Gamma", "Normal").

        Returns
        -------
        dict
            Dictionary with keys: name, kind, param_names, params_support, support,
            parametrizations.
        """
        dist_cls = _get_dist_class(name)
        dist = dist_cls()
        result = {
            "name": name,
            "kind": dist.kind,
            "param_names": dist.param_names,
            "params_support": _format_support(dist.params_support),
            "support": _format_support(dist.support),
        }
        parametrizations = getattr(dist_cls, "parametrizations", None)
        if parametrizations is not None:
            result["parametrizations"] = parametrizations
        return result

    def find(self, kind=None, num_params=None, support=None):
        """Find distributions matching given criteria.

        Parameters
        ----------
        kind : str, optional
            Filter by kind: "continuous", "discrete".
        num_params : int, optional
            Filter by number of parameters.
        support : str, optional
            Filter by support type: "positive", "bounded", "unbounded",
            "non_negative".

        Returns
        -------
        list of PreliZ distribution instances
        """
        results = []
        candidates = []

        if kind == "continuous":
            candidates = all_continuous
        elif kind == "discrete":
            candidates = all_discrete
        else:
            candidates = all_continuous + all_discrete

        for dist_cls in candidates:
            dist = dist_cls()

            if num_params is not None and len(dist.param_names) != num_params:
                continue

            if support is not None:
                if not self._matches_support(dist, support):
                    continue

            results.append(dist)

        return results

    def _matches_support(self, dist, support_type):
        lower, upper = dist.support
        if lower is None or upper is None:
            return False
        if support_type == "positive":
            return lower >= 0 and upper == np.inf
        elif support_type == "bounded":
            return lower != -np.inf and upper != np.inf
        elif support_type == "unbounded":
            return lower == -np.inf and upper == np.inf
        elif support_type == "non_negative":
            return lower >= 0
        return True


catalog = DistributionCatalog()
