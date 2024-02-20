import pytest
import numpy as np
from preliz import beta_mode


@pytest.mark.parametrize(
    "lower, upper, mode, mass, expected_alpha, expected_beta",
    [
        (0.25, 0.75, 0.5, 0.9, 4.94, 4.94),  # Example test case
        # Add more test cases here
    ],
)
def test_beta_mode(lower, upper, mode, mass, expected_alpha, expected_beta):
    _, dist = beta_mode(lower, upper, mode, mass)

    assert np.isclose(dist.alpha, expected_alpha, atol=0.01)
    assert np.isclose(dist.beta, expected_beta, atol=0.01)


def test_invalid_mass():
    with pytest.raises(ValueError):
        beta_mode(0.25, 0.75, 0.5, mass=1.1)


def test_invalid_mode():
    with pytest.raises(ValueError):
        beta_mode(0.25, 0.75, 1.5)


def test_invalid_bounds():
    with pytest.raises(ValueError):
        beta_mode(0.75, 0.25, 0.5)


def test_plot_beta_mode():
    _, _ = beta_mode(0.25, 0.75, 0.5, 0.9, plot=True, plot_kwargs={"pointinterval": True})
