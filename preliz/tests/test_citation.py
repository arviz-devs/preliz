import pytest

import preliz as pz


@pytest.mark.parametrize(
    "methods, show_as", [(None, "bibtex"), ([pz.Roulette, pz.dirichlet_mode], "bibtex")]
)
def test_citations(methods, show_as):
    pz.citations(methods, show_as)
