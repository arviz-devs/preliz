import pandas as pd
import numpy as np
import pymc as pm
import bambi as bmb
import preliz as pz


data = pz.Normal(0, 1).rvs(200)

with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.HalfNormal("b", sigma=1)
    y = pm.Normal("y", mu=a, sigma=b, observed=data)
    idata = pm.sample(tune=200, draws=500, random_seed=2945)


def test_p2p_pymc():
    pz.posterior_to_prior(model, idata)
    assert 'Gamma\x1b[0m("b", alpha=' in pz.posterior_to_prior(model, idata, alternative="auto")
    pz.posterior_to_prior(model, idata, alternative=[pz.LogNormal()])
    assert 'Gamma\x1b[0m("b", mu=' in pz.posterior_to_prior(
        model, idata, alternative={"b": [pz.Gamma(mu=0)]}
    )


bmb_data = pd.DataFrame(
    {
        "y": np.random.normal(size=117),
        "x": np.random.normal(size=117),
        "x1": np.random.normal(size=117),
    }
)
bmb_prior = {"Intercept": bmb.Prior("HalfStudentT", nu=1)}
bmb_model = bmb.Model("y ~ x + x1", bmb_data, priors=bmb_prior)
bmb_idata = bmb_model.fit(tune=200, draws=200, random_seed=2945)


def test_p2p_bambi():
    pz.posterior_to_prior(bmb_model, bmb_idata)
    assert 'Gamma\x1b[0m", alpha=' in pz.posterior_to_prior(
        bmb_model, bmb_idata, alternative="auto"
    )
    pz.posterior_to_prior(bmb_model, bmb_idata, alternative=[pz.LogNormal()])
    assert 'Normal\x1b[0m", mu=' in pz.posterior_to_prior(
        bmb_model, bmb_idata, alternative={"Intercept": [pz.Normal(mu=1, sigma=1)]}
    )
