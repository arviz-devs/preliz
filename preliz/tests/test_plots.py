"""Tests plot_utils functions."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_helper import run_notebook

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


def test_plot_interactive(capsys):
    pz.Poisson(4.5).plot_interactive()
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out
    run_notebook("plot_interactive.ipynb")


@pytest.fixture
def sample_ax():
    return plt.subplot()


def test_plot_references(sample_ax):
    # Test with a dictionary of references
    references_dict = {"Ref1": 0.5, "Ref2": 1.0, "Ref3": 1.5}
    pz.internal.plot_helper.plot_references(references_dict, sample_ax)

    lines = sample_ax.lines
    texts = sample_ax.texts

    assert len(lines) == len(texts) == len(references_dict)

    # Test with a list of references
    sample_ax.clear()
    references_list = [0.5, 1.0, 1.5]
    pz.internal.plot_helper.plot_references(references_list, sample_ax)

    lines = sample_ax.lines

    assert len(lines) == len(references_list)

    # Test with a single reference value
    sample_ax.clear()
    reference_single = 0.5
    pz.internal.plot_helper.plot_references(reference_single, sample_ax)

    lines = sample_ax.lines

    assert len(lines) == 1

    # Test with None input
    sample_ax.clear()
    references_none = None
    pz.internal.plot_helper.plot_references(references_none, sample_ax)

    lines = sample_ax.lines
    texts = sample_ax.texts

    assert len(lines) == len(texts) == 0
