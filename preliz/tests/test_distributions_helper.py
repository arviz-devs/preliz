from preliz.internal.distribution_helper import process_extra


def test_process_extra():
    ref0 = {"TruncatedNormal": {"lower": -3, "upper": 3}}
    ref1 = {"StudentT": {"nu": 3.4}, "Normal": {"mu": 3.0}}

    assert process_extra("TruncatedNormal(lower=-3, upper=3)") == ref0
    assert process_extra("StudentT(nu=3.4),Normal(mu=3)") == ref1
