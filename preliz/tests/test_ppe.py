import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytest
from numpy.testing import assert_allclose

np.random.seed(42)


@pytest.mark.parametrize(
    "params",
    [
        {
            "mu_x": 0,
            "sigma_x": 10,
            "sigma_z": 10,
            "target": pz.Normal(mu=174, sigma=20),
            "X": np.random.normal(0, 10, 120),
            "new_x": 174,
            "new_z": 2.9,
        },
        {
            "mu_x": [0, 1],
            "sigma_x": [10, 10],
            "sigma_z": 10,
            "target": pz.Normal(mu=174, sigma=20),
            "X": np.random.normal(0, 10, 120),
            "new_x": [174, 174],
            "new_z": 2.9,
        },
        {
            "mu_x": 0,
            "sigma_x": 10,
            "sigma_z": 10,
            "target": [(pz.Normal(mu=174, sigma=20), 0.5), (pz.Normal(mu=176, sigma=19.5), 0.5)],
            "X": np.random.normal(0, 10, 120),
            "new_x": 175,
            "new_z": 2.9,
        },
        {
            "mu_x": [0, 1],
            "sigma_x": 10,
            "sigma_z": 10,
            "target": [
                (pz.Normal(mu=174, sigma=20), 0.5),
                (pz.Normal(mu=176, sigma=19.5), 0.4),
                (pz.StudentT(mu=174, sigma=20, nu=3), 0.1),
            ],
            "X": np.random.normal(0, 10, 120),
            "new_x": [175, 175],
            "new_z": 3,
        },
    ],
)
def test_ppe(params):
    Y = np.zeros_like(params["X"])
    target = params.pop("target")
    with pm.Model() as model:
        x = pm.Normal("x", shape=1 if isinstance(params["mu_x"], int) else 2)
        z = pm.HalfNormal("z")
        x_idx = (
            x if np.asarray(params["mu_x"]).size == 1 else x[np.repeat(np.arange(2), Y.size / 2)]
        )
        pm.Normal("y", x_idx, z, observed=Y)

    new_prior = (
        pz.ppe(model, target, method="projective").replace("\x1b[1m", "").replace("\x1b[0m", "")
    )
    exec_context = {}
    exec(new_prior, globals(), exec_context)
    model = exec_context.get("model")
    initial = model.initial_point()
    assert_allclose(initial["x"], params["new_x"])
    assert_allclose(initial["z_log__"], params["new_z"], atol=0.1)
