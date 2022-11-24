"""Tests plot_utils functions."""
# pylint: disable=redefined-outer-name
import pytest
import matplotlib.pyplot as plt

import preliz as pz


@pytest.fixture(scope="function")
def a_dist():
    return pz.Beta(2, 6)


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
def test_plot_pdf_cdf_ppf(a_dist, kwargs):
    a_dist.plot_pdf(**kwargs)
    a_dist.plot_cdf(**kwargs)
    kwargs.pop("support", None)
    a_dist.plot_ppf(**kwargs)
