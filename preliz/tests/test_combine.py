from preliz.unidimensional.combine import combine
from preliz.distributions import Normal, Gamma


def test_combine():
    # Create some sample distributions
    distributions = [Normal(0, 1), Gamma(2, 1)]

    # Call the combine function
    fit_dists, ax = combine(distributions, rng=413)

    assert [dist.__class__.__name__ for dist in fit_dists[:2]] == ["StudentT", "Normal"]
    assert [dist.is_frozen for dist in fit_dists] == [True, True, False, False]

    fit_dists, ax = combine(distributions, weights=[0, 1], rng=None, plot=0)

    assert [dist.__class__.__name__ for dist in fit_dists[:2]] == ["Gamma", "LogNormal"]
    assert [dist.is_frozen for dist in fit_dists] == [True, True, True, True]
    assert ax is None

    fit_dists, _ = combine(distributions, dist_names=["Moyal"])
    assert len(fit_dists) == 1
    assert fit_dists[0].__class__.__name__ == "Moyal"
