import inspect
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_helper import run_notebook

from preliz import predictive_explorer
from preliz.internal.plot_helper import get_textboxes, plot_repr
from preliz.ppls.agnostic import _extract_var, _get_default_var_name


class _FakeDist:
    param_names = ["p"]
    params_support = [(np.finfo(float).eps, 1 - np.finfo(float).eps)]
    p = 0.5


class _FakeRV:
    def __init__(self, name):
        self.name = name


class _FakeModel:
    def __init__(self, free, observed):
        self.free_RVs = [_FakeRV(name) for name in free]
        self.observed_RVs = [_FakeRV(name) for name in observed]


def _make_signature(params):
    return inspect.Signature(
        [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
            )
            for name, default in params.items()
        ]
    )


def test_predictive_explorer():
    run_notebook("predictive_explorer.ipynb")


def test_get_textboxes_expit_upper_bound():
    signature = _make_signature({"p": 0.5})

    textboxes = get_textboxes(signature, {"p": (_FakeDist(), 0, "expit")}, "ecdf")

    assert "-inf" in textboxes["p"].description
    assert "inf" in textboxes["p"].description


def test_get_textboxes_skips_unused_parameter_with_default():
    signature = _make_signature({"p": 0.5, "unused": 1.0})

    textboxes = get_textboxes(signature, {"p": (_FakeDist(), 0, None)}, "ecdf")

    assert "p" in textboxes
    assert "unused" not in textboxes


def test_get_textboxes_unused_required_parameter_raises():
    signature = _make_signature({"unused": inspect.Parameter.empty, "p": 0.5})

    with pytest.raises(ValueError, match="unused"):
        get_textboxes(signature, {"p": (_FakeDist(), 0, None)}, "ecdf")


def test_plot_repr_does_not_mutate_stats_kwargs():
    stats_kwargs = {}
    _, ax = plt.subplots()
    results = np.array([[1.0, 2.0], [3.0, 4.0]])

    plot_repr(results, "hist", None, 2, stats_kwargs, ax)
    plot_repr(results, "ecdf", None, 2, stats_kwargs, ax)

    assert stats_kwargs == {}
    plt.close("all")


def test_plot_repr_empty_results():
    _, ax = plt.subplots()

    plot_repr(np.array([]), "ecdf", None, 1, {}, ax)

    assert len(ax.texts) == 1
    plt.close("all")


@pytest.mark.parametrize("kind", ["hist", "kde", "ecdf"])
def test_plot_repr_scalar_results(kind):
    _, ax = plt.subplots()

    plot_repr(np.array([1.0, 2.0, 3.0]), kind, None, 3, {}, ax)

    plt.close("all")


def test_plot_repr_invalid_kind():
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="kind_plot"):
        plot_repr(np.array([1.0]), "invalid", None, 1, {}, ax)

    plt.close("all")


def test_predictive_explorer_invalid_kind():
    def model(a_mu):
        return a_mu

    with pytest.raises(ValueError, match="kind_plot"):
        predictive_explorer(model, kind_plot="invalid")


def test_predictive_explorer_invalid_samples():
    def model(a_mu):
        return a_mu

    with pytest.raises(ValueError, match="samples"):
        predictive_explorer(model, samples=0)


def test_predictive_explorer_scalar_output():
    pytest.importorskip("ipywidgets")

    def scalar_model(mu):
        from preliz.distributions import Normal

        return Normal(mu, 1).rvs()

    with patch("preliz.predictive.predictive_explorer.display") as mock_display:
        predictive_explorer(scalar_model, samples=5)

    mock_display.assert_called_once()


def test_default_var_name_for_prior_group():
    model = _FakeModel(free=["a", "b"], observed=["c"])

    assert _get_default_var_name(model, "prior") == "b"
    assert _get_default_var_name(model, "prior_predictive") == "c"


def test_extract_var_missing_group():
    class _FakeIdata:
        children = ["posterior"]

        def __getitem__(self, key):
            raise KeyError(key)

    with pytest.raises(ValueError, match="posterior_predictive"):
        _extract_var(_FakeIdata(), "posterior_predictive", "y")


def test_extract_var_missing_variable():
    class _FakeDataset:
        data_vars = ["x"]

        def __contains__(self, key):
            return key in self.data_vars

    class _FakeNode:
        dataset = _FakeDataset()

    class _FakeIdata:
        children = ["posterior"]

        def __getitem__(self, key):
            return _FakeNode()

    with pytest.raises(ValueError, match="'y'"):
        _extract_var(_FakeIdata(), "posterior", "y")
