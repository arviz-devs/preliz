import numpy as np
import pytest

import preliz as pz
from preliz.unidimensional.matching import match_moments, match_quantiles


@pytest.fixture
def normal_dist():
    return pz.Normal(15, 1)


@pytest.fixture
def normal_array():
    return pz.Normal(15, 1).rvs(size=1000, random_state=247)


@pytest.fixture
def gamma_dist():
    return pz.Gamma()


@pytest.mark.parametrize("dist", ["normal_dist", "normal_array"])
def test_match_moments_basic_parametrized(dist, gamma_dist, request):
    dist = request.getfixturevalue(dist)
    match_moments(dist, gamma_dist)
    assert np.isclose(gamma_dist.mean(), dist.mean(), atol=1e-2)
    assert np.isclose(gamma_dist.std(), dist.std(), atol=1e-2)


@pytest.mark.parametrize("dist", ["normal_dist", "normal_array"])
def test_match_moments_custom_moments(dist, gamma_dist, request):
    dist = request.getfixturevalue(dist)
    match_moments(dist, gamma_dist, moments="mdk")
    assert np.isclose(gamma_dist.mean(), dist.mean(), atol=1e-2)
    assert np.isclose(gamma_dist.std(), dist.std(), atol=1e-2)


def test_match_moments_invalid_moments(normal_dist, gamma_dist):
    with pytest.raises(ValueError):
        match_moments(pz.Cauchy(), gamma_dist)

    with pytest.raises(ValueError):
        match_moments(normal_dist, pz.Cauchy())


def test_match_quantiles_basic(normal_dist, gamma_dist):
    match_quantiles(normal_dist, gamma_dist)
    q = np.array([0.25, 0.5, 0.5])
    assert np.allclose(gamma_dist.ppf(q), normal_dist.ppf(q), atol=1e-2)


def test_match_quantiles_basic_array(normal_array):
    dist = pz.Normal()
    match_quantiles(normal_array, dist)
    q = np.array([0.25, 0.5, 0.5])
    assert np.allclose(dist.ppf(q), np.quantile(normal_array, q), atol=1e-2)


def test_match_quantiles_custom_quantiles(normal_dist):
    to_dist = pz.StudentT(nu=5)
    quantiles = [0.1, 0.5, 0.9]
    match_quantiles(normal_dist, to_dist, quantiles=quantiles)
    assert np.allclose(to_dist.ppf(quantiles), normal_dist.ppf(quantiles), atol=1e-2)


def test_match_quantiles_invalid_quantiles(normal_dist):
    to_dist = pz.Gamma()
    with pytest.raises(ValueError):
        match_quantiles(normal_dist, to_dist, quantiles=[-0.1, 1.1])


def test_match_moments_plot(normal_dist, gamma_dist):
    result = match_moments(normal_dist, gamma_dist, plot=True)
    assert isinstance(result, tuple)
    assert hasattr(result[0], "mean")
    assert hasattr(result[1], "plot")


def test_match_quantiles_plot(normal_dist, gamma_dist):
    result = match_quantiles(normal_dist, gamma_dist, plot=True)
    assert isinstance(result, tuple)
    assert hasattr(result[0], "mean")
    assert hasattr(result[1], "plot")
