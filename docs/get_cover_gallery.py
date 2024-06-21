# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
from preliz import Gamma
from preliz import distributions
from preliz.internal.distribution_helper import init_vals


az.style.use("arviz-doc")
rng = np.random.default_rng(247)

init_vals["Hurdle"] = None
for name, params in init_vals.items():
    color = f"C{rng.integers(0, 4)}"
    _, ax = plt.subplots(figsize=(6, 4))
    dist = getattr(distributions, name)
    if name in ["Truncated", "Censored"]:
        dist(Gamma(mu=2, sigma=1), -np.inf, 2).plot_pdf(legend=False, ax=ax, color=color)
        ax.get_lines()[0].set_alpha(0)
        ax.get_lines()[1].set_linewidth(4)
    elif name in ["Hurdle"]:
        dist(Gamma(mu=2, sigma=1), 0.8).plot_pdf(legend=False, ax=ax, color=color)
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
    ax.spines[["left", "bottom"]].set_visible(False)
    plt.savefig(f"examples/img/{name}.png")
