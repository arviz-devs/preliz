import pytest
from numpy.testing import assert_allclose

import preliz as pz


@pytest.mark.parametrize(
    "distribution, params",
    [
        (pz.Beta, (2, 5)),
        (pz.BetaScaled, (2, 5, -1, 4)),
        (pz.Cauchy, (0, 1)),
        (pz.Exponential, (5,)),
        (pz.Gamma, (2, 5)),
        (pz.HalfCauchy, (1,)),
        (pz.HalfNormal, (1,)),
        (pz.HalfStudent, (3, 1)),
        (pz.HalfStudent, (1000000, 1)),
        (pz.HalfNormal, (2,)),
        (pz.InverseGamma, (2, 5)),
        (pz.Laplace, (0, 2)),
        (pz.LogNormal, (0, 1)),
        (pz.Normal, (0, 1)),
        (pz.Pareto, (5, 1)),
        (pz.SkewNormal, (0, 1, 6)),
        (pz.Student, (4, 0, 1)),
        (pz.TruncatedNormal, (0, 1, -1, 1)),
        (pz.Uniform, (2, 5)),
        (pz.Wald, (2, 1)),
        (pz.Weibull, (2, 1)),
        (pz.Binomial, (5, 0.5)),
        (pz.DiscreteUniform, (-2, 2)),
        (pz.NegativeBinomial, (10, 0.5)),
        (pz.Poisson, (4.2,)),
    ],
)
def test_auto_recover(distribution, params):
    dist = distribution(*params)
    sample = dist.rvs(10000)
    pz.mle([dist], sample)
    assert_allclose(dist.params, params, atol=1)


def test_recover_right():
    dists = [pz.Normal(), pz.Gamma(), pz.Poisson()]
    sample = pz.Normal(0, 1).rvs(10000)
    pz.mle(dists, sample)
    assert dists[0].name == "normal"

    sample = pz.Gamma(2, 10).rvs(10000)
    pz.mle(dists, sample)
    assert dists[1].name == "gamma"

    sample = pz.Poisson(10).rvs(10000)
    pz.mle(dists, sample)
    assert dists[2].name == "poisson"
