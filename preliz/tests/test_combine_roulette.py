import pytest
from numpy.testing import assert_almost_equal
from preliz import combine_roulette
from preliz.distributions import BetaScaled, LogNormal, StudentT

response0 = (
    [1.5, 2.5, 3.5],
    [0.32142857142857145, 0.35714285714285715, 0.32142857142857145],
    28,
    0,
    10,
    10,
    11,
)
response1 = (
    [7.5, 8.5, 9.5],
    [0.32142857142857145, 0.35714285714285715, 0.32142857142857145],
    28,
    0,
    10,
    10,
    11,
)
response2 = ([9.5], [1], 10, 0, 10, 10, 11)
response3 = ([9.5], [1], 10, 0, 10, 10, 14)


@pytest.mark.parametrize(
    "responses, weights, dist_names, params, result",
    [
        ([response0, response1], [0.5, 0.5], None, None, BetaScaled(1.2, 1, 0, 10)),
        (
            [response0, response1],
            [0.5, 0.5],
            ["Beta", "StudentT"],
            "TruncatedNormal(lower=0), StudentT(nu=1000)",
            StudentT(1000, 5.5, 3.1),
        ),
        ([response0, response2], [1, 1], None, None, LogNormal(1.1, 0.6)),
    ],
)
def test_combine_roulette(responses, weights, dist_names, params, result):
    dist = combine_roulette(responses, weights, dist_names, params)
    assert_almost_equal(dist.params, result.params, decimal=1)


def test_combine_roulette_error():
    with pytest.raises(ValueError):
        combine_roulette([response0, response3])
