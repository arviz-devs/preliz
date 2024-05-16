import numpy as np
import preliz as pz
import pymc as pm
import pytest
from numpy.testing import assert_allclose

np.random.seed(42)
rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "mu_x, sigma_x, sigma_z, target, X, new_mu_x, new_sigma_x, new_sigma_z",
    [
        (0, 100, 100, pz.Normal(mu=175.0, sigma=9.12), None, 175.01249486, 0.965982, 9.043413),
        (
            [0, 17],
            [10, 10],
            10,
            pz.Normal(mu=175, sigma=9.12),
            np.random.normal(0, 10, 120),
            (175.096086, 174.918323),
            (1.192426, 1.223783),
            9.01539,
        )
    ],
)
def test_ppe_pymc(mu_x, sigma_x, sigma_z, target, X, new_mu_x, new_sigma_x, new_sigma_z):
    Y = (
        np.zeros(100)
        if X is None
        else 2
        + np.random.normal(
            X,
            1,
        )
    )
    with pm.Model() as model:
        x = pm.Normal("x", mu=mu_x, sigma=sigma_x)
        z = pm.HalfNormal("z", sigma_z)
        x_idx = (
            x
            if max(np.asarray(mu_x).size, np.asarray(sigma_x).size) == 1
            else x[np.repeat(np.arange(2), Y.size / 2)]
        )
        y = pm.Normal("y", x_idx, z, observed=Y)
    new_prior_vals = pz.ppe(model, target)[1]
    assert_allclose(new_prior_vals["x"].mu, new_mu_x, rtol=1e-6)
    assert_allclose(new_prior_vals["x"].sigma, new_sigma_x, rtol=1e-6)
    assert_allclose(new_prior_vals["z"].sigma, new_sigma_z, rtol=1e-6)
