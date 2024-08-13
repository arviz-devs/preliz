import re

import preliz as pz
import pymc as pm

data = pz.Normal(0, 1).rvs(200)

with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.HalfNormal("b", sigma=1)
    y = pm.Normal("y", mu=a, sigma=b, observed=data)
    idata = pm.sample(tune=200, draws=200)


def test_p2p():
    pz.posterior_to_prior(model, idata)
    assert 'Gamma\x1b[0m("b", alpha=' in pz.posterior_to_prior(model, idata, alternative="auto")
    pz.posterior_to_prior(model, idata, alternative=[pz.LogNormal()])
    assert 'Gamma\x1b[0m("b", mu=' in pz.posterior_to_prior(
        model, idata, alternative={"b": [pz.Gamma(mu=0)]}
    )
