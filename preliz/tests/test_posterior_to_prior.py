import pandas as pd
import pymc as pm
import bambi as bmb
from preliz.distributions import Normal, LogNormal, Gamma
from preliz.ppls.agnostic import posterior_to_prior


SEED = 2945

data = Normal(0, 1).rvs(200, random_state=SEED)

with pm.Model() as model:
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.HalfNormal("b", sigma=[1, 1], shape=2)
    y = pm.Normal("y", mu=a, sigma=b[0], observed=data)  # pylint:disable = unsubscriptable-object
    idata = pm.sample(tune=200, draws=500, random_seed=SEED)


def test_p2p_pymc():
    posterior_to_prior(model, idata)
    assert 'Gamma\x1b[0m("b", alpha=' in posterior_to_prior(model, idata, new_families="auto")
    posterior_to_prior(model, idata, new_families=[LogNormal()])
    assert 'Gamma\x1b[0m("b", mu=' in posterior_to_prior(
        model, idata, new_families={"b": [Gamma(mu=0)]}
    )


bmb_data = pd.DataFrame(
    {
        "y": Normal(0, 1).rvs(117, random_state=SEED + 1),
        "x": Normal(0, 1).rvs(117, random_state=SEED + 2),
        "x1": Normal(0, 1).rvs(117, random_state=SEED + 3),
    }
)
bmb_prior = {"Intercept": bmb.Prior("Normal", mu=0, sigma=1)}
bmb_model = bmb.Model("y ~ x + x1", bmb_data, priors=bmb_prior)
bmb_idata = bmb_model.fit(tune=200, draws=200, random_seed=SEED)


def test_p2p_bambi():
    posterior_to_prior(bmb_model, bmb_idata)
    assert 'Gamma\x1b[0m", alpha=' in posterior_to_prior(bmb_model, bmb_idata, new_families="auto")
    posterior_to_prior(bmb_model, bmb_idata, new_families=[LogNormal()])
    assert 'Normal\x1b[0m", mu=' in posterior_to_prior(
        bmb_model, bmb_idata, new_families={"Intercept": [Normal(mu=1, sigma=1)]}
    )
