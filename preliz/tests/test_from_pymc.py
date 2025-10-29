import numpy as np
import pymc as pm
import pytest
from numpy.testing import assert_allclose

import preliz as pz
from preliz.internal.distribution_helper import init_vals
from preliz.ppls.pymc_io import from_pymc


def test_from_pymc():
    for dist_name, params in init_vals.items():
        if dist_name in [
            "BetaScaled",
            "LogLogistic",
            "ScaledInverseChiSquared",
            "ChiSquared",
            "Truncated",
            "Censored",
            "Mixture",
            "Dirichlet",
            "LogitNormal",  # Remove after PyMC 5.26.1 is out
            "DiscreteWeibull",  # Remove after PyMC 5.26.1 is out
        ]:
            continue
        pymc_dist = getattr(pm, dist_name).dist(**params)
        preliz_dist = from_pymc(pymc_dist)
        PrelizDistClass = getattr(pz, dist_name)
        assert isinstance(preliz_dist, PrelizDistClass)
        for param_name, param_value in params.items():
            preliz_param_value = getattr(preliz_dist, param_name)
            assert_allclose(preliz_param_value, param_value)


def test_from_pymc_model():
    with pm.Model() as model:
        a = pm.Beta("x", alpha=np.array([2, 5]), beta=5)
        b = pm.Gamma("y", mu=3, sigma=1)
        c = pm.Truncated("c", pm.Laplace.dist(0, 2), lower=0)
        d = pm.Censored("d", pm.HalfNormal.dist(2), lower=2)
        e = pm.ZeroInflatedNegativeBinomial("e", mu=4, alpha=1, psi=0.3)
        f = pm.HurdleGamma("f", mu=2, sigma=1, psi=0.5)
        g = pm.Mixture("g", w=[0.3, 0.7], comp_dists=[pm.Normal.dist(0, 1), pm.Normal.dist(5, 1)])

    for rv in model.free_RVs:
        pz.plot(rv)

    a_preliz = from_pymc(a)
    assert isinstance(a_preliz, pz.Beta)
    assert_allclose(a_preliz.alpha, [2, 5])
    assert a_preliz.beta == 5

    b_preliz = from_pymc(b)
    assert isinstance(b_preliz, pz.Gamma)
    assert_allclose(b_preliz.mu, 3)
    assert_allclose(b_preliz.sigma, 1)

    c_preliz = from_pymc(c)
    assert isinstance(c_preliz, pz.Truncated)
    assert isinstance(c_preliz.dist, pz.Laplace)
    assert_allclose(c_preliz.dist.mu, 0)
    assert_allclose(c_preliz.dist.b, 2)
    assert c_preliz.lower == 0
    assert c_preliz.upper == np.inf

    d_preliz = from_pymc(d)
    assert isinstance(d_preliz, pz.Censored)
    assert isinstance(d_preliz.dist, pz.HalfNormal)
    assert d_preliz.dist.sigma == 2
    assert d_preliz.lower == 2
    assert d_preliz.upper == np.inf

    e_preliz = from_pymc(e)
    assert isinstance(e_preliz, pz.ZeroInflatedNegativeBinomial)
    assert e_preliz.mu == 4
    assert e_preliz.alpha == 1
    assert e_preliz.psi == 0.3

    f_preliz = from_pymc(f)
    assert isinstance(f_preliz, pz.Hurdle)
    assert isinstance(f_preliz.dist, pz.Gamma)
    assert f_preliz.dist.mu == 2
    assert f_preliz.dist.sigma == 1
    assert f_preliz.psi == 0.5

    g_preliz = from_pymc(g)
    assert isinstance(g_preliz, pz.Mixture)
    assert np.allclose(g_preliz.weights, [0.3, 0.7])
    assert isinstance(g_preliz.dist[0], pz.Normal)
    assert g_preliz.dist[0].mu == 0
    assert g_preliz.dist[0].sigma == 1
    assert isinstance(g_preliz.dist[1], pz.Normal)
    assert g_preliz.dist[1].mu == 5
    assert g_preliz.dist[1].sigma == 1


@pytest.mark.parametrize(
    "preliz_dist, pymc_dist, lower, upper, fixed_params",
    [
        (pz.Beta(), pm.Beta.dist(alpha=np.nan, beta=np.nan), 0.4, 0.7, None),
        (pz.Gamma(), pm.Gamma.dist(np.nan, np.nan), 1, 5, None),
        (pz.Gamma(mu=3), pm.Gamma.dist(np.nan, np.nan), 1, 5, {"mu": 3}),
        (
            pz.Truncated(pz.Laplace(), lower=0),
            pm.Truncated.dist(pm.Laplace.dist(np.nan, np.nan), lower=0),
            1,
            10,
            None,
        ),
        (
            pz.Censored(pz.HalfNormal(), lower=2),
            pm.Censored.dist(pm.HalfNormal.dist(np.nan), lower=2),
            2,
            7,
            None,
        ),
        (
            pz.ZeroInflatedNegativeBinomial(),
            pm.ZeroInflatedNegativeBinomial.dist(mu=np.nan, alpha=np.nan, psi=np.nan),
            1,
            10,
            None,
        ),
        (
            pz.Hurdle(pz.Gamma(), psi=0.5),
            pm.HurdleGamma.dist(mu=np.nan, sigma=np.nan, psi=0.5),
            0,
            10,
            None,
        ),
    ],
)
def test_from_pymc_maxent(preliz_dist, pymc_dist, lower, upper, fixed_params):
    original_dist, _ = pz.maxent(preliz_dist, lower, upper)
    converted_dist, _ = pz.maxent(pymc_dist, lower, upper, fixed_params=fixed_params)
    assert_allclose(converted_dist.params, original_dist.params)


@pytest.mark.parametrize(
    "preliz_dist, pymc_dist, q1, q2, q3",
    [
        (pz.Beta(), pm.Beta.dist(alpha=np.nan, beta=np.nan), 0.3, 0.5, 0.7),
        (pz.Gamma(), pm.Gamma.dist(np.nan, np.nan), 0.5, 1, 2.5),
        (
            pz.Truncated(pz.Laplace(), lower=0),
            pm.Truncated.dist(pm.Laplace.dist(np.nan, np.nan), lower=0),
            0.45,
            1,
            2,
        ),
        (
            pz.Censored(pz.HalfNormal(), lower=2),
            pm.Censored.dist(pm.HalfNormal.dist(np.nan), lower=2),
            2,
            3,
            4,
        ),
        (
            pz.ZeroInflatedNegativeBinomial(),
            pm.ZeroInflatedNegativeBinomial.dist(mu=np.nan, alpha=np.nan, psi=np.nan),
            1,
            3,
            4,
        ),
        (
            pz.Hurdle(pz.Gamma(), psi=0.75),
            pm.HurdleGamma.dist(mu=np.nan, sigma=np.nan, psi=0.75),
            0.5,
            3,
            4,
        ),
    ],
)
def test_from_pymc_quartile(preliz_dist, pymc_dist, q1, q2, q3):
    original_dist, _ = pz.quartile(preliz_dist, q1, q2, q3)
    converted_dist, _ = pz.quartile(pymc_dist, q1, q2, q3)
    assert_allclose(converted_dist.params, original_dist.params)
