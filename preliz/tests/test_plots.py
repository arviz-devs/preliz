"""Tests plot_utils functions."""
# pylint: disable=redefined-outer-name
import pytest
import matplotlib.pyplot as plt

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
        dist = getattr(pz.distributions, distribution)
        kind = ["pdf", "cdf", "ppf"][idx % 3]
        fixed_lim = ["auto", "both"][idx % 2]
        dist().plot_interactive(kind=kind, fixed_lim=fixed_lim)
