"""Tests plot_utils functions."""
# pylint: disable=redefined-outer-name
import pytest
import matplotlib.pyplot as plt
import numpy as np

import preliz as pz


@pytest.fixture(scope="function")
def two_dist():
    return pz.Beta(2, 6), pz.Poisson(4.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"moments": "mdsk"},
        {"moments": "mdsk", "legend": "title"},
        {"pointinterval": True},
        {"pointinterval": True, "levels": [0.1, 0.9]},
        {"pointinterval": True, "interval": "eti", "levels": [0.9]},
        {"pointinterval": True, "interval": "quantiles"},
        {"pointinterval": True, "interval": "quantiles", "levels": [0.1, 0.5, 0.9]},
        {"support": "restricted"},
        {"color": "C1", "alpha": 0.1},
        {"figsize": (4, 4)},
        {"ax": plt.subplots()[1]},
    ],
)
def test_continuous_plot_pdf_cdf_ppf(two_dist, kwargs):
    for a_dist in two_dist:
        a_dist.plot_pdf(**kwargs)
        a_dist.plot_cdf(**kwargs)
        kwargs.pop("support", None)
        a_dist.plot_ppf(**kwargs)


def test_plot_interactive():
    for idx, distribution in enumerate(pz.distributions.__all__):
        if distribution not in ["Dirichlet", "MvNormal"]:
            dist = getattr(pz.distributions, distribution)
            kind = ["pdf", "cdf", "ppf"][idx % 3]
            fixed_lim = ["auto", "both"][idx % 2]
            dist().plot_interactive(kind=kind, fixed_lim=fixed_lim)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"marginals": True},
        {"pointinterval": True},
        {"pointinterval": True, "levels": [0.1, 0.9]},
        {"pointinterval": True, "interval": "eti", "levels": [0.9]},
        {"pointinterval": True, "interval": "quantiles"},
        {"pointinterval": True, "interval": "quantiles", "levels": [0.1, 0.5, 0.9]},
        {"support": "restricted"},
        {"figsize": (4, 4)},
    ],
)
def test_dirichlet_plot(kwargs):
    a_dist = pz.Dirichlet([2, 1, 2])
    a_dist.plot_pdf(**kwargs)
    kwargs.pop("marginals", None)
    a_dist.plot_cdf(**kwargs)
    kwargs.pop("support", None)
    a_dist.plot_ppf(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"marginals": True},
        {"pointinterval": True},
        {"pointinterval": True, "levels": [0.1, 0.9]},
        {"pointinterval": True, "interval": "eti", "levels": [0.9]},
        {"pointinterval": True, "interval": "quantiles"},
        {"pointinterval": True, "interval": "quantiles", "levels": [0.1, 0.5, 0.9]},
        {"support": "restricted"},
        {"figsize": (4, 4)},
    ],
)
def test_mvnormal_plot(kwargs):
    a_dist = pz.MvNormal(np.zeros(2), np.eye(2))
    a_dist.plot_pdf(**kwargs)
    kwargs.pop("marginals", None)
    a_dist.plot_cdf(**kwargs)
    kwargs.pop("support", None)
    a_dist.plot_ppf(**kwargs)
