import pandas as pd
import pymc as pm
import bambi as bmb
import preliz as pz


SEED = 2945

data = pz.Normal(0, 1).rvs(200, random_state=SEED)

with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.HalfNormal("b", sigma=[1, 1], shape=2)
    y = pm.Normal("y", mu=a, sigma=b[0], observed=data)  # pylint:disable = unsubscriptable-object
    idata = pm.sample(tune=200, draws=500, random_seed=SEED)


def test_p2p_pymc():
    pz.posterior_to_prior(model, idata)
    assert 'Gamma\x1b[0m("b", alpha=' in pz.posterior_to_prior(model, idata, new_families="auto")
    pz.posterior_to_prior(model, idata, new_families=[pz.LogNormal()])
    assert 'Gamma\x1b[0m("b", mu=' in pz.posterior_to_prior(
        model, idata, new_families={"b": [pz.Gamma(mu=0)]}
    )


bmb_data = pd.DataFrame(
    {
        "y": pz.Normal(0, 1).rvs(117, random_state=SEED + 1),
        "x": pz.Normal(0, 1).rvs(117, random_state=SEED + 2),
        "x1": pz.Normal(0, 1).rvs(117, random_state=SEED + 3),
    }
)
bmb_prior = {"Intercept": bmb.Prior("Normal", mu=0, sigma=1)}
bmb_model = bmb.Model("y ~ x + x1", bmb_data, priors=bmb_prior)
bmb_idata = bmb_model.fit(tune=200, draws=200, random_seed=SEED)


def test_p2p_bambi():
    pz.posterior_to_prior(bmb_model, bmb_idata)
    assert 'Gamma\x1b[0m", alpha=' in pz.posterior_to_prior(
        bmb_model, bmb_idata, new_families="auto"
    )
    pz.posterior_to_prior(bmb_model, bmb_idata, new_families=[pz.LogNormal()])
    assert 'Normal\x1b[0m", mu=' in pz.posterior_to_prior(
        bmb_model, bmb_idata, new_families={"Intercept": [pz.Normal(mu=1, sigma=1)]}
    )
