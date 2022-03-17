import pytest
from preliz.utils import check_boundaries


DOMAIN_ERROR = "The provided boundaries are outside the domain of the beta distribution"


@pytest.mark.parametrize(
    "name, params",
    [
        ("beta", (0, 2)),
        ("beta", (-1, 1)),
        ("beta", (-1, 2)),
        ("gamma", (-1, 2)),
        ("lognormal", (-1, 2)),
        ("exponential", (-1, 2)),
    ],
)
def test_domain_error(name, params):
    with pytest.raises(ValueError) as err:
        check_boundaries(name, *params)
        assert str(err.value) == DOMAIN_ERROR


def test_beta_boundaries():
    with pytest.raises(ValueError) as err:
        check_boundaries("beta", 0, 1)
        assert (
            str(err.value)
            == "Given the provided boundaries, mass will be always 1. Please provide other values"
        )
