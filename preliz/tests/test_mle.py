import pytest
from numpy.testing import assert_allclose

import preliz as pz


@pytest.mark.parametrize(
    "distribution, params",
    [
        (pz.Beta, (2, 5)),
        (pz.BetaScaled, (2, 5, -1, 4)),
        (pz.Exponential, (5,)),
        (pz.Gamma, (2, 5)),
        (pz.HalfNormal, (2,)),
        (pz.Laplace, (0, 2)),
        (pz.LogNormal, (0, 1)),
        (pz.Normal, (0, 1)),
        (pz.SkewNormal, (0, 1, 6)),
        (pz.Student, (4, 0, 1)),
        (pz.Uniform, (2, 5)),
        (pz.Weibull, (2, 1)),
        (pz.Binomial, (5, 0.5)),
        (pz.DiscreteUniform, (-2, 2)),
        (pz.NegativeBinomial, (10, 0.5)),
        (pz.Poisson, (4.2,)),
    ],
)
def test_auto_recover(distribution, params):
    sample = distribution(*params).rvs(10000)
    dist = pz.mle(sample, [distribution])
    assert_allclose(dist.params, params, atol=1)


def test_recover_right():
    sample = pz.Normal(0, 1).rvs(10000)
    dist = pz.mle(sample, ["Normal", pz.Gamma, pz.Poisson])
    assert dist.name == "normal"

    sample = pz.Gamma(2, 10).rvs(10000)
    dist = pz.mle(sample, ["Normal", pz.Gamma, pz.Poisson])
    assert dist.name == "gamma"

    sample = pz.Poisson(10).rvs(10000)
    dist = pz.mle(sample, ["Normal", pz.Gamma, pz.Poisson])
    assert dist.name == "poisson"
