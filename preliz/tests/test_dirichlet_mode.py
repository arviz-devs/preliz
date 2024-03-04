import pytest
import numpy as np
from preliz import dirichlet_mode


def test_dirichlet_mode():
    _, dist = dirichlet_mode([0.22, 0.22, 0.32, 0.22], 0.99, bound=0.02)
    for alpha, expected in zip(dist.alpha, [675, 675, 981.37, 675.0]):
        assert np.isclose(alpha, expected, atol=0.01)


def test_invalid_mass():
    with pytest.raises(ValueError):
        dirichlet_mode([0.22, 0.22, 0.32, 0.22], 1.1, bound=0.02)


def test_invalid_mode():
    with pytest.raises(ValueError):
        dirichlet_mode([-0.2, 0.22, 0.32, 1.22], 0.99, bound=0.02)


def test_plot_beta_mode():
    _, _ = dirichlet_mode(
        mode=[0.22, 0.22, 0.32, 0.22],
        mass=0.99,
        bound=0.02,
        plot=True,
        plot_kwargs={"pointinterval": True},
    )
