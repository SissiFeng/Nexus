"""Tests for baseline adapters and domain configuration switching.

Validates that:
1. BaselineAdapter registry and factory work correctly for all strategies.
2. DomainConfig switching works correctly across all three experimental domains.
3. Baselines integrate properly with case study infrastructure.
"""

from __future__ import annotations

import copy
import unittest

from optimization_copilot.domain_knowledge.loader import DomainConfig
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.case_studies.baselines.adapters import (
    BaselineAdapter,
    get_default_baselines,
    get_all_baselines,
    get_baseline_capabilities,
    select_baselines_for_space,
)


# ---------------------------------------------------------------------------
# Baseline Adapter Tests
# ---------------------------------------------------------------------------


class TestBaselineAdapter(unittest.TestCase):
    """Tests for BaselineAdapter registry and factory."""

    def setUp(self) -> None:
        """Save original registry so tests can safely mutate it."""
        self._original_registry = dict(BaselineAdapter.REGISTRY)

    def tearDown(self) -> None:
        """Restore original registry after each test."""
        BaselineAdapter.REGISTRY = self._original_registry

    # -- get() ---------------------------------------------------------------

    def test_get_returns_algorithm_plugin(self) -> None:
        """get() should return an AlgorithmPlugin instance."""
        plugin = BaselineAdapter.get("random")
        self.assertIsInstance(plugin, AlgorithmPlugin)

    def test_get_unknown_raises_value_error(self) -> None:
        """get() should raise ValueError for unknown baseline name."""
        with self.assertRaises(ValueError) as ctx:
            BaselineAdapter.get("nonexistent_algo")
        self.assertIn("nonexistent_algo", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))

    def test_get_returns_fresh_instances(self) -> None:
        """get() should return a new instance each time."""
        a = BaselineAdapter.get("random")
        b = BaselineAdapter.get("random")
        self.assertIsNot(a, b)

    # -- get_all() -----------------------------------------------------------

    def test_get_all_returns_dict_with_all_baselines(self) -> None:
        """get_all() should return a dict with all registered baselines."""
        baselines = BaselineAdapter.get_all()
        self.assertIsInstance(baselines, dict)
        self.assertEqual(set(baselines.keys()), set(BaselineAdapter.REGISTRY.keys()))
        for name, plugin in baselines.items():
            self.assertIsInstance(plugin, AlgorithmPlugin, f"{name} is not AlgorithmPlugin")

    # -- get_subset() --------------------------------------------------------

    def test_get_subset_returns_only_requested(self) -> None:
        """get_subset() should return only the requested baselines."""
        subset = BaselineAdapter.get_subset(["random", "sobol"])
        self.assertEqual(set(subset.keys()), {"random", "sobol"})
        for plugin in subset.values():
            self.assertIsInstance(plugin, AlgorithmPlugin)

    def test_get_subset_unknown_raises_value_error(self) -> None:
        """get_subset() should raise ValueError for unknown names."""
        with self.assertRaises(ValueError):
            BaselineAdapter.get_subset(["random", "fake_algo"])

    # -- available() ---------------------------------------------------------

    def test_available_lists_all_names(self) -> None:
        """available() should list all registered baseline names."""
        names = BaselineAdapter.available()
        self.assertIsInstance(names, list)
        expected = {"random", "sobol", "gp_bo", "cma_es", "het_gp"}
        self.assertEqual(set(names), expected)

    # -- name() and capabilities() per baseline ------------------------------

    def test_each_baseline_has_name(self) -> None:
        """Every registered baseline should have a name() method returning str."""
        for baseline_key in BaselineAdapter.available():
            plugin = BaselineAdapter.get(baseline_key)
            name = plugin.name()
            self.assertIsInstance(name, str, f"Baseline {baseline_key!r} name is not str")
            self.assertTrue(len(name) > 0, f"Baseline {baseline_key!r} name is empty")

    def test_each_baseline_has_capabilities(self) -> None:
        """Every registered baseline should have a capabilities() method returning dict."""
        for baseline_key in BaselineAdapter.available():
            plugin = BaselineAdapter.get(baseline_key)
            caps = plugin.capabilities()
            self.assertIsInstance(caps, dict, f"Baseline {baseline_key!r} capabilities is not dict")

    def test_random_baseline_capabilities(self) -> None:
        """Random sampler should support all types and not require observations."""
        caps = BaselineAdapter.get("random").capabilities()
        self.assertTrue(caps["supports_categorical"])
        self.assertTrue(caps["supports_continuous"])
        self.assertTrue(caps["supports_discrete"])
        self.assertFalse(caps["requires_observations"])

    def test_sobol_baseline_capabilities(self) -> None:
        """Sobol sampler should support all types and have max_dimensions=21."""
        caps = BaselineAdapter.get("sobol").capabilities()
        self.assertTrue(caps["supports_categorical"])
        self.assertTrue(caps["supports_continuous"])
        self.assertFalse(caps["requires_observations"])
        self.assertEqual(caps["max_dimensions"], 21)

    def test_gp_bo_baseline_capabilities(self) -> None:
        """GP BO should not support categorical and should require observations."""
        caps = BaselineAdapter.get("gp_bo").capabilities()
        self.assertFalse(caps["supports_categorical"])
        self.assertTrue(caps["supports_continuous"])
        self.assertTrue(caps["requires_observations"])

    def test_cma_es_baseline_capabilities(self) -> None:
        """CMA-ES should not support categorical and should require observations."""
        caps = BaselineAdapter.get("cma_es").capabilities()
        self.assertFalse(caps["supports_categorical"])
        self.assertTrue(caps["supports_continuous"])
        self.assertTrue(caps["requires_observations"])

    def test_het_gp_baseline_capabilities(self) -> None:
        """Heteroscedastic GP should support heteroscedastic noise."""
        caps = BaselineAdapter.get("het_gp").capabilities()
        self.assertTrue(caps["supports_heteroscedastic_noise"])
        self.assertFalse(caps["supports_categorical"])
        self.assertTrue(caps["requires_observations"])

    # -- register() ----------------------------------------------------------

    def test_register_custom_baseline(self) -> None:
        """register() should add a new baseline to the registry."""
        from optimization_copilot.backends.builtin import RandomSampler

        BaselineAdapter.register("custom_random", lambda: RandomSampler())
        self.assertIn("custom_random", BaselineAdapter.available())
        plugin = BaselineAdapter.get("custom_random")
        self.assertIsInstance(plugin, AlgorithmPlugin)

    def test_register_overwrites_existing(self) -> None:
        """register() with existing name should overwrite the factory."""
        from optimization_copilot.backends.builtin import SobolSampler

        BaselineAdapter.register("random", lambda: SobolSampler())
        plugin = BaselineAdapter.get("random")
        self.assertEqual(plugin.name(), "sobol_sampler")

    # -- unregister() --------------------------------------------------------

    def test_unregister_removes_baseline(self) -> None:
        """unregister() should remove a baseline from the registry."""
        BaselineAdapter.register("temp_baseline", lambda: BaselineAdapter.get("random"))
        BaselineAdapter.unregister("temp_baseline")
        self.assertNotIn("temp_baseline", BaselineAdapter.available())

    def test_unregister_unknown_raises_key_error(self) -> None:
        """unregister() should raise KeyError for unknown name."""
        with self.assertRaises(KeyError):
            BaselineAdapter.unregister("nonexistent_algo")

    # -- convenience functions -----------------------------------------------

    def test_get_default_baselines_returns_three(self) -> None:
        """get_default_baselines() should return 3 strategies."""
        defaults = get_default_baselines()
        self.assertEqual(len(defaults), 3)
        self.assertEqual(set(defaults.keys()), {"random", "sobol", "gp_bo"})
        for plugin in defaults.values():
            self.assertIsInstance(plugin, AlgorithmPlugin)

    def test_get_all_baselines_returns_five(self) -> None:
        """get_all_baselines() should return 5 strategies."""
        all_baselines = get_all_baselines()
        self.assertEqual(len(all_baselines), 5)
        expected = {"random", "sobol", "gp_bo", "cma_es", "het_gp"}
        self.assertEqual(set(all_baselines.keys()), expected)

    def test_get_baseline_capabilities_returns_all(self) -> None:
        """get_baseline_capabilities() should return capabilities for all baselines."""
        caps = get_baseline_capabilities()
        self.assertEqual(len(caps), 5)
        for name, c in caps.items():
            self.assertIsInstance(c, dict, f"{name} capabilities not a dict")
            self.assertIn("supports_continuous", c)

    def test_select_baselines_for_categorical_space(self) -> None:
        """select_baselines_for_space with categorical should exclude GP and CMA-ES."""
        selected = select_baselines_for_space(has_categorical=True)
        for name, plugin in selected.items():
            caps = plugin.capabilities()
            self.assertTrue(
                caps.get("supports_categorical", False),
                f"{name} does not support categorical but was selected",
            )
        # GP BO, CMA-ES, and het_gp should be excluded
        self.assertNotIn("gp_bo", selected)
        self.assertNotIn("cma_es", selected)
        self.assertNotIn("het_gp", selected)

    def test_select_baselines_no_observations(self) -> None:
        """select_baselines_for_space with no observations should exclude model-based."""
        selected = select_baselines_for_space(requires_no_observations=True)
        for name, plugin in selected.items():
            caps = plugin.capabilities()
            self.assertFalse(
                caps.get("requires_observations", True),
                f"{name} requires observations but was selected",
            )
        self.assertIn("random", selected)
        self.assertIn("sobol", selected)


# ---------------------------------------------------------------------------
# Domain Switch Tests
# ---------------------------------------------------------------------------


class TestDomainSwitch(unittest.TestCase):
    """Test switching between domain configs works for case studies."""

    # -- individual domain loading -------------------------------------------

    def test_electrochemistry_domain_loads(self) -> None:
        """Electrochemistry domain config should load without error."""
        cfg = DomainConfig("electrochemistry")
        self.assertEqual(cfg.domain_name, "electrochemistry")

    def test_catalysis_domain_loads(self) -> None:
        """Catalysis domain config should load without error."""
        cfg = DomainConfig("catalysis")
        self.assertEqual(cfg.domain_name, "catalysis")

    def test_perovskite_domain_loads(self) -> None:
        """Perovskite domain config should load without error."""
        cfg = DomainConfig("perovskite")
        self.assertEqual(cfg.domain_name, "perovskite")

    # -- domain switching ----------------------------------------------------

    def test_switch_electrochemistry_to_catalysis(self) -> None:
        """Switching from electrochemistry to catalysis should work cleanly."""
        cfg_ec = DomainConfig("electrochemistry")
        cfg_cat = DomainConfig("catalysis")
        # They should be independent configs
        self.assertNotEqual(cfg_ec.domain_name, cfg_cat.domain_name)
        self.assertNotEqual(cfg_ec.get_instruments(), cfg_cat.get_instruments())

    def test_switch_catalysis_to_perovskite(self) -> None:
        """Switching from catalysis to perovskite should work cleanly."""
        cfg_cat = DomainConfig("catalysis")
        cfg_pv = DomainConfig("perovskite")
        self.assertNotEqual(cfg_cat.domain_name, cfg_pv.domain_name)
        self.assertNotEqual(cfg_cat.get_instruments(), cfg_pv.get_instruments())

    # -- instruments ---------------------------------------------------------

    def test_each_domain_has_instruments(self) -> None:
        """All three domains should have non-empty instruments."""
        for domain in DomainConfig.SUPPORTED_DOMAINS:
            cfg = DomainConfig(domain)
            instruments = cfg.get_instruments()
            self.assertIsInstance(instruments, dict, f"{domain} instruments not dict")
            self.assertTrue(len(instruments) > 0, f"{domain} has no instruments")

    # -- constraints ---------------------------------------------------------

    def test_each_domain_has_constraints(self) -> None:
        """All three domains should have non-empty constraints."""
        for domain in DomainConfig.SUPPORTED_DOMAINS:
            cfg = DomainConfig(domain)
            constraints = cfg.get_constraints()
            self.assertIsInstance(constraints, dict, f"{domain} constraints not dict")
            self.assertTrue(len(constraints) > 0, f"{domain} has no constraints")

    # -- quality thresholds --------------------------------------------------

    def test_each_domain_has_quality_thresholds(self) -> None:
        """All three domains should have non-empty quality thresholds."""
        for domain in DomainConfig.SUPPORTED_DOMAINS:
            cfg = DomainConfig(domain)
            thresholds = cfg.get_quality_thresholds()
            self.assertIsInstance(thresholds, dict, f"{domain} thresholds not dict")
            self.assertTrue(len(thresholds) > 0, f"{domain} has no quality thresholds")

    # -- domain-specific features --------------------------------------------

    def test_catalysis_has_known_incompatibilities(self) -> None:
        """Catalysis domain should have known_incompatibilities."""
        cfg = DomainConfig("catalysis")
        incompatibilities = cfg.get_known_incompatibilities()
        self.assertIsInstance(incompatibilities, list)
        self.assertTrue(len(incompatibilities) > 0)

    def test_perovskite_has_phase_stability_rules(self) -> None:
        """Perovskite domain config dict should include phase_stability_rules."""
        cfg = DomainConfig("perovskite")
        raw = cfg.config
        self.assertIn("phase_stability_rules", raw)
        rules = raw["phase_stability_rules"]
        self.assertIsInstance(rules, (dict, list))
        self.assertTrue(len(rules) > 0)

    # -- repr ----------------------------------------------------------------

    def test_domain_config_repr_includes_domain_name(self) -> None:
        """repr(DomainConfig) should include the domain name."""
        for domain in DomainConfig.SUPPORTED_DOMAINS:
            cfg = DomainConfig(domain)
            r = repr(cfg)
            self.assertIn(domain, r)
            self.assertIn("DomainConfig", r)

    # -- error handling ------------------------------------------------------

    def test_invalid_domain_raises_value_error(self) -> None:
        """DomainConfig with invalid domain name should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            DomainConfig("invalid_domain")
        self.assertIn("invalid_domain", str(ctx.exception))

    # -- independence --------------------------------------------------------

    def test_domain_configs_are_independent(self) -> None:
        """Modifying one domain config should not affect another."""
        cfg1 = DomainConfig("electrochemistry")
        cfg2 = DomainConfig("electrochemistry")

        # Mutate the internal config of cfg1
        cfg1.config["instruments"]["test_key"] = "test_value"

        # cfg2 should be unaffected
        self.assertNotIn("test_key", cfg2.get_instruments())

    def test_all_three_domains_can_coexist(self) -> None:
        """All three domain configs should be loadable simultaneously."""
        configs = {
            domain: DomainConfig(domain)
            for domain in DomainConfig.SUPPORTED_DOMAINS
        }
        self.assertEqual(len(configs), 3)

        # Each should have its own instruments
        instrument_sets = [
            set(cfg.get_instruments().keys())
            for cfg in configs.values()
        ]
        # At least some instruments should differ between domains
        self.assertNotEqual(instrument_sets[0], instrument_sets[1])

    # -- supported domains ---------------------------------------------------

    def test_supported_domains_tuple(self) -> None:
        """SUPPORTED_DOMAINS should contain exactly three domains."""
        self.assertEqual(
            set(DomainConfig.SUPPORTED_DOMAINS),
            {"electrochemistry", "catalysis", "perovskite"},
        )


if __name__ == "__main__":
    unittest.main()
