# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np

from preliz import Gamma, distributions, style
from preliz.internal.distribution_helper import init_vals

style.use("preliz-doc")
rng = np.random.default_rng(247)

init_vals["Hurdle"] = None
init_vals["Mixture"] = None
init_vals["SkewStudentT"] = {"mu": 0.0, "sigma": 1, "a": 2.5, "b": 1.5}
for name, params in init_vals.items():
    color = f"C{rng.integers(0, 4)}"
    _, ax = plt.subplots(figsize=(3.5, 2.3))
    dist = getattr(distributions, name)
    if name in ["Truncated", "Censored"]:
        dist(Gamma(mu=2, sigma=1), -np.inf, 2).plot_pdf(legend=False, ax=ax, color=color)
        ax.get_lines()[0].set_alpha(0)
        ax.get_lines()[1].set_linewidth(4)
    elif name == "Hurdle":
        dist(Gamma(mu=2, sigma=1), 0.8).plot_pdf(legend=False, ax=ax, color=color)
        ax.get_lines()[0].set_alpha(0)
        ax.get_lines()[1].set_linewidth(4)
    elif name == "Mixture":
        dist([Gamma(mu=1, sigma=0.5), Gamma(mu=3, sigma=1)], [0.5, 0.5]).plot_pdf(
            legend=False, ax=ax, color=color
        )
        ax.get_lines()[0].set_alpha(0)
        ax.get_lines()[1].set_linewidth(4)
    else:
        dist = getattr(distributions, name)
        dist(**params).plot_pdf(legend=False, ax=ax, color=color)

        if dist().kind == "discrete":
            if name not in ["Bernoulli", "Categorical"]:
                ax.get_lines()[0].set_linewidth(3)
                ax.get_lines()[1].set_markersize(10)
                ax.get_lines()[2].set_alpha(0)
            else:
                ax.get_lines()[0].set_markersize(10)
                ax.get_lines()[1].set_alpha(0)
        else:
            ax.get_lines()[0].set_alpha(0)
            ax.get_lines()[1].set_linewidth(4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)

    if name in ["Categorical"]:
        ax.set_xticks([0, 1, 2], labels=["♣", "♥", "♦"])
    else:
        if name in ["BetaScaled", "Truncated", "Censored", "Pareto", "Mixture"]:
            l_b, u_b = (-np.inf, np.inf)
        elif name == "Hurdle":
            l_b, u_b = (0, np.inf)
        else:
            l_b, u_b = dist().support

        if l_b == -np.inf:
            l_b = None
        if u_b == np.inf:
            u_b = None

        pos = ax.get_ylim()[0]

        # The boundaries depends on the parameterization
        if name in [
            "BetaScaled",
            "TruncatedNormal",
            "Triangular",
            "Uniform",
            "Binomial",
            "BetaBinomial",
            "DiscreteUniform",
            "HyperGeometric",
            "Censored",
            "Hurdle",
            "Mixture",
            "Truncated",
        ]:
            marker_0 = marker_1 = "*"
        elif name == "Pareto":
            marker_0 = "*"
            marker_1 = ">"
        else:
            marker_0 = "<"
            marker_1 = ">"

        if l_b is None:
            ax.plot(
                0,
                pos,
                color="k",
                marker=marker_0,
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=3,
            )
        if u_b is None:
            ax.plot(
                1,
                pos,
                color="k",
                marker=marker_1,
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=3,
            )

        if name == "VonMises":
            ax.set_xticks([l_b, u_b], labels=[r"$-\pi$", r"$\pi$"])

        if l_b is not None and u_b is not None:
            ax.set_xticks([l_b, u_b])
        elif l_b is not None and u_b is None:
            ax.set_xticks([l_b])
        elif l_b is None and u_b is not None:
            ax.set_xticks([u_b])

        ax.spines["bottom"].set_position(("data", pos))

    plt.savefig(f"examples/img/{name}.png")
