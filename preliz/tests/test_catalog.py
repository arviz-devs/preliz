import pytest

from preliz.distributions.catalog import catalog


class TestCatalogGet:
    def test_get_continuous_instances(self):
        dists = catalog.get("continuous")
        assert len(dists) > 0
        assert all(hasattr(d, "pdf") for d in dists)
        assert all(hasattr(d, "kind") for d in dists)

    def test_get_discrete_instances(self):
        dists = catalog.get("discrete")
        assert len(dists) > 0
        assert all(hasattr(d, "pmf") or hasattr(d, "pdf") for d in dists)

    def test_get_continuous_names(self):
        names = catalog.get("continuous", output="names")
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
        assert "Normal" in names
        assert "Gamma" in names

    def test_get_discrete_names(self):
        names = catalog.get("discrete", output="names")
        assert len(names) > 0
        assert "Poisson" in names
        assert "Binomial" in names

    def test_get_by_support_category(self):
        positive = catalog.get("positive")
        assert len(positive) > 0

        bounded = catalog.get("bounded")
        assert len(bounded) > 0

        unbounded = catalog.get("unbounded")
        assert len(unbounded) > 0

    def test_get_invalid_category(self):
        with pytest.raises(ValueError, match="Unknown category"):
            catalog.get("invalid_category")

    def test_get_invalid_output(self):
        with pytest.raises(ValueError, match="Invalid value for 'output'"):
            catalog.get("continuous", output="invalid")


class TestCatalogInfo:
    def test_info_returns_dict(self):
        info = catalog.info("Gamma")
        assert isinstance(info, dict)

    def test_info_has_required_keys(self):
        info = catalog.info("Gamma")
        assert "name" in info
        assert "kind" in info
        assert "param_names" in info
        assert "params_support" in info
        assert "support" in info

    def test_info_gamma(self):
        info = catalog.info("Gamma")
        assert info["name"] == "Gamma"
        assert info["kind"] == "continuous"
        assert info["param_names"] == ("alpha", "beta")
        assert "parametrizations" in info
        assert ("alpha", "beta") in info["parametrizations"]
        assert ("mu", "sigma") in info["parametrizations"]

    def test_info_normal(self):
        info = catalog.info("Normal")
        assert info["name"] == "Normal"
        assert info["kind"] == "continuous"
        assert "parametrizations" in info

    def test_info_poisson(self):
        info = catalog.info("Poisson")
        assert info["name"] == "Poisson"
        assert info["kind"] == "discrete"
        assert info["param_names"] == ("mu",)

    def test_info_support_formatting(self):
        info = catalog.info("Gamma")
        assert info["support"] == (0, "inf")
        assert info["params_support"] == ((0, "inf"), (0, "inf"))

        info = catalog.info("Normal")
        assert info["support"] == ("-inf", "inf")

    def test_info_invalid_name(self):
        with pytest.raises(AttributeError):
            catalog.info("NonExistentDistribution")


class TestCatalogFind:
    def test_find_by_kind_continuous(self):
        dists = catalog.find(kind="continuous")
        assert len(dists) > 0
        assert all(d.kind == "continuous" for d in dists)

    def test_find_by_kind_discrete(self):
        dists = catalog.find(kind="discrete")
        assert len(dists) > 0
        assert all(d.kind == "discrete" for d in dists)

    def test_find_by_num_params(self):
        dists = catalog.find(num_params=1)
        assert len(dists) > 0
        assert all(len(d.param_names) == 1 for d in dists)

        dists = catalog.find(num_params=2)
        assert len(dists) > 0
        assert all(len(d.param_names) == 2 for d in dists)

    def test_find_by_support_positive(self):
        dists = catalog.find(support="positive")
        assert len(dists) > 0
        for d in dists:
            lower, upper = d.support
            assert lower >= 0
            assert upper == float("inf")

    def test_find_by_support_bounded(self):
        dists = catalog.find(support="bounded")
        assert len(dists) > 0
        for d in dists:
            lower, upper = d.support
            assert lower != float("-inf")
            assert upper != float("inf")

    def test_find_combined_filters(self):
        dists = catalog.find(kind="continuous", num_params=2)
        assert len(dists) > 0
        assert all(d.kind == "continuous" for d in dists)
        assert all(len(d.param_names) == 2 for d in dists)

    def test_find_no_filters(self):
        dists = catalog.find()
        assert len(dists) > 0


class TestCatalogRepr:
    def test_repr_contains_distributions(self):
        repr_str = repr(catalog)
        assert "PreliZ Distributions" in repr_str
        assert "Continuous" in repr_str
        assert "Discrete" in repr_str
        assert "Normal" in repr_str
        assert "Poisson" in repr_str

    def test_repr_html_contains_distributions(self):
        html = catalog._repr_html_()
        assert "<div" in html
        assert "PreliZ Distributions" in html
        assert "Normal" in html
