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
        {"moments": "mdsk", "legend": False},
        {"moments": "mdsk", "legend": True},
        {"pointinterval": True},
        {"quantiles": [0.25, 0.5, 0.75]},
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
