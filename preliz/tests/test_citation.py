import pytest

import preliz as pz


@pytest.mark.parametrize(
    "methods, filepath, format_type",
    [
        (None, None, "bibtex"),
        ([pz.Roulette, pz.dirichlet_mode], None, "bibtex"),
        (["all", None, "bibtex"]),
    ],
)
def test_citations(methods, filepath, format_type):
    pz.citations(methods, filepath, format_type)
