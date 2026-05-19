import numpy as np

import preliz as pz
from preliz.internal.distribution_helper import init_vals

dists_with_lmoments = [
    "AsymmetricLaplace",
    "Beta",
    "Bernoulli",
    "Binomial",
    "Cauchy",
    "DiscreteUniform",
    "ExGaussian",
    "Gamma",
    "Gumbel",
    "HalfNormal",
    "HyperGeometric",
    "Kumaraswamy",
    "Laplace",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "TruncatedNormal",
    "Uniform",
    "Wald",
    "Weibull",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]


def test_lmoments():
    for name_dist, params in init_vals.items():
        if name_dist not in dists_with_lmoments:
            continue
        dist = getattr(pz, name_dist)(**params)
        lm1, lm2, lm3, lm4 = dist.lmoments()
        mean = dist.mean()

        if np.isnan(mean):
            assert np.isnan(lm1)
        else:
            assert np.isclose(lm1, mean)

        assert lm2 >= 0 or np.isnan(lm2)

        assert np.isnan(lm3) or -1 <= lm3 <= 1
        assert np.isnan(lm4) or -1 <= lm4 <= 1
