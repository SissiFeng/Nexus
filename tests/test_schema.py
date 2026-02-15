"""Tests for Decision Schema Hardening (schema.rules)."""

from optimization_copilot.schema.rules import (
    DecisionRule,
    DecisionRuleEngine,
    RuleDiff,
    RuleSignature,
    RuleTrace,
    _compute_signature,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_diagnostics(**overrides: float) -> dict[str, float]:
    """Diagnostic vector with safe defaults — mirrors test_meta_controller."""
    d: dict[str, float] = {
        "convergence_trend": 0.1,
        "improvement_velocity": 0.1,
        "variance_contraction": 0.8,
        "noise_estimate": 0.1,
        "failure_rate": 0.0,
        "failure_clustering": 0.0,
        "feasibility_shrinkage": 0.0,
        "parameter_drift": 0.05,
        "model_uncertainty": 0.5,
        "exploration_coverage": 0.3,
        "kpi_plateau_length": 0,
        "best_kpi_value": 5.0,
        "data_efficiency": 0.5,
        "constraint_violation_rate": 0.0,
    }
    d.update(overrides)
    return d


# ── Test: default rules loaded ────────────────────────────────────────────────

class TestDefaultRulesLoaded:
    def test_engine_has_ten_default_rules(self):
        engine = DecisionRuleEngine()
        assert len(engine.rules) == 10

    def test_default_rule_ids(self):
        engine = DecisionRuleEngine()
        ids = {r.rule_id for r in engine.rules}
        expected = {
            "phase_cold_start_v1",
            "phase_stagnation_plateau_v1",
            "phase_stagnation_failure_v1",
            "phase_exploitation_convergence_v1",
            "phase_exploitation_coverage_v1",
            "phase_learning_v1",
            "risk_conservative_phase_v1",
            "risk_conservative_failure_v1",
            "risk_aggressive_v1",
            "risk_moderate_v1",
        }
        assert ids == expected

    def test_rules_sorted_by_priority_descending(self):
        engine = DecisionRuleEngine()
        priorities = [r.priority for r in engine.rules]
        assert priorities == sorted(priorities, reverse=True)


# ── Test: cold-start rule fires ───────────────────────────────────────────────

class TestColdStartRule:
    def test_cold_start_fires_with_few_observations(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics()
        trace = engine.evaluate(diag, snapshot_size=3)

        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_cold_start_v1" in fired_ids

    def test_cold_start_action_is_set_phase(self):
        engine = DecisionRuleEngine()
        trace = engine.evaluate(_base_diagnostics(), snapshot_size=3)
        cold_sig = next(s for s in trace.signatures if s.rule_id == "phase_cold_start_v1")
        assert cold_sig.action_taken == "set_phase=cold_start"

    def test_cold_start_does_not_fire_with_enough_observations(self):
        engine = DecisionRuleEngine()
        trace = engine.evaluate(_base_diagnostics(), snapshot_size=15)
        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_cold_start_v1" not in fired_ids


# ── Test: stagnation rule fires ───────────────────────────────────────────────

class TestStagnationRule:
    def test_plateau_stagnation(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics(kpi_plateau_length=15)
        trace = engine.evaluate(diag, snapshot_size=20)
        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_stagnation_plateau_v1" in fired_ids

    def test_failure_clustering_stagnation(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics(failure_clustering=0.8)
        trace = engine.evaluate(diag, snapshot_size=15)
        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_stagnation_failure_v1" in fired_ids


# ── Test: exploitation rule fires ─────────────────────────────────────────────

class TestExploitationRule:
    def test_convergence_exploitation(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1)
        trace = engine.evaluate(diag, snapshot_size=15)
        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_exploitation_convergence_v1" in fired_ids

    def test_coverage_exploitation(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics(exploration_coverage=0.9, model_uncertainty=0.05)
        trace = engine.evaluate(diag, snapshot_size=15)
        fired_ids = [s.rule_id for s in trace.signatures]
        assert "phase_exploitation_coverage_v1" in fired_ids


# ── Test: rule trace correctness ──────────────────────────────────────────────

class TestRuleTrace:
    def test_trace_contains_correct_signatures(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics()
        trace = engine.evaluate(diag, snapshot_size=3)

        # Cold-start should fire; its risk derivative should also fire
        assert len(trace.signatures) > 0
        assert len(trace.decision_path) == len(trace.signatures)

        for sig in trace.signatures:
            assert sig.rule_id in trace.rule_versions
            assert trace.rule_versions[sig.rule_id] == sig.version
            assert len(sig.signature_hash) == 16
            assert len(sig.trigger_set) > 0

    def test_decision_path_order_matches_evaluation(self):
        engine = DecisionRuleEngine()
        trace = engine.evaluate(_base_diagnostics(), snapshot_size=3)
        assert trace.decision_path == [s.rule_id for s in trace.signatures]

    def test_cold_start_triggers_conservative_risk(self):
        """Cold-start phase should also fire the conservative-phase risk rule."""
        engine = DecisionRuleEngine()
        trace = engine.evaluate(_base_diagnostics(), snapshot_size=3)
        fired_ids = set(trace.decision_path)
        assert "risk_conservative_phase_v1" in fired_ids

    def test_learning_phase_gets_moderate_risk(self):
        engine = DecisionRuleEngine()
        diag = _base_diagnostics()
        trace = engine.evaluate(diag, snapshot_size=15)
        fired_ids = set(trace.decision_path)
        assert "phase_learning_v1" in fired_ids
        assert "risk_moderate_v1" in fired_ids


# ── Test: diff_traces ─────────────────────────────────────────────────────────

class TestDiffTraces:
    def test_identical_traces_have_no_diff(self):
        engine = DecisionRuleEngine()
        trace = engine.evaluate(_base_diagnostics(), snapshot_size=3)
        diff = engine.diff_traces(trace, trace)
        assert diff["added_rules"] == []
        assert diff["removed_rules"] == []
        assert diff["changed_versions"] == {}
        assert diff["changed_actions"] == {}
        assert diff["path_changed"] is False

    def test_detects_added_and_removed_rules(self):
        engine = DecisionRuleEngine()
        trace_a = engine.evaluate(_base_diagnostics(), snapshot_size=3)
        trace_b = engine.evaluate(
            _base_diagnostics(kpi_plateau_length=15),
            snapshot_size=20,
        )
        diff = engine.diff_traces(trace_a, trace_b)
        # Different phases fire — guaranteed to have changes
        assert diff["path_changed"] is True

    def test_detects_version_change(self):
        sig_a = RuleSignature(
            rule_id="r1", version="1.0.0",
            trigger_set=frozenset(["cond"]), action_taken="act",
            signature_hash="abc",
        )
        sig_b = RuleSignature(
            rule_id="r1", version="2.0.0",
            trigger_set=frozenset(["cond"]), action_taken="act",
            signature_hash="def",
        )
        trace_a = RuleTrace(
            signatures=[sig_a], decision_path=["r1"],
            rule_versions={"r1": "1.0.0"},
        )
        trace_b = RuleTrace(
            signatures=[sig_b], decision_path=["r1"],
            rule_versions={"r1": "2.0.0"},
        )
        diff = DecisionRuleEngine.diff_traces(trace_a, trace_b)
        assert "r1" in diff["changed_versions"]
        assert diff["changed_versions"]["r1"] == ("1.0.0", "2.0.0")


# ── Test: add / remove rules ─────────────────────────────────────────────────

class TestRuleCRUD:
    def test_add_rule(self):
        engine = DecisionRuleEngine()
        custom = DecisionRule(
            rule_id="custom_v1",
            version="0.1.0",
            description="Custom test rule",
            trigger_conditions=("default",),
            action="custom_action",
            priority=999,
        )
        engine.add_rule(custom)
        assert engine.get_rule("custom_v1") is custom
        assert len(engine.rules) == 11
        # Highest priority should be first
        assert engine.rules[0].rule_id == "custom_v1"

    def test_remove_rule(self):
        engine = DecisionRuleEngine()
        engine.remove_rule("risk_moderate_v1")
        assert len(engine.rules) == 9
        ids = {r.rule_id for r in engine.rules}
        assert "risk_moderate_v1" not in ids

    def test_remove_nonexistent_raises(self):
        engine = DecisionRuleEngine()
        try:
            engine.remove_rule("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_get_nonexistent_raises(self):
        engine = DecisionRuleEngine()
        try:
            engine.get_rule("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


# ── Test: export / import round-trip ──────────────────────────────────────────

class TestExportImport:
    def test_round_trip_preserves_rules(self):
        engine = DecisionRuleEngine()
        exported = engine.export_rules()

        engine2 = DecisionRuleEngine()
        engine2.import_rules(exported)

        assert len(engine2.rules) == len(engine.rules)
        for r1, r2 in zip(engine.rules, engine2.rules):
            assert r1.rule_id == r2.rule_id
            assert r1.version == r2.version
            assert r1.priority == r2.priority
            assert r1.action == r2.action
            assert r1.trigger_conditions == r2.trigger_conditions

    def test_import_replaces_existing(self):
        engine = DecisionRuleEngine()
        engine.import_rules([
            {
                "rule_id": "only_rule",
                "version": "1.0.0",
                "description": "The only rule",
                "trigger_conditions": ["default"],
                "action": "do_something",
                "priority": 1,
            }
        ])
        assert len(engine.rules) == 1
        assert engine.rules[0].rule_id == "only_rule"

    def test_exported_format_is_json_serialisable(self):
        import json
        engine = DecisionRuleEngine()
        exported = engine.export_rules()
        # Must not raise
        text = json.dumps(exported, sort_keys=True)
        restored = json.loads(text)
        assert len(restored) == 10


# ── Test: signature hashing determinism ───────────────────────────────────────

class TestSignatureHashing:
    def test_deterministic(self):
        h1 = _compute_signature("r1", "1.0.0", frozenset(["a", "b"]), "act")
        h2 = _compute_signature("r1", "1.0.0", frozenset(["a", "b"]), "act")
        assert h1 == h2

    def test_order_independent_triggers(self):
        h1 = _compute_signature("r1", "1.0.0", frozenset(["a", "b"]), "act")
        h2 = _compute_signature("r1", "1.0.0", frozenset(["b", "a"]), "act")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = _compute_signature("r1", "1.0.0", frozenset(["a"]), "act")
        h2 = _compute_signature("r2", "1.0.0", frozenset(["a"]), "act")
        assert h1 != h2

    def test_hash_length(self):
        h = _compute_signature("rule", "1.0.0", frozenset(["cond"]), "action")
        assert len(h) == 16  # matches _sha256 truncation

    def test_engine_signatures_are_deterministic(self):
        """Full integration: same inputs produce same signature hashes."""
        engine = DecisionRuleEngine()
        diag = _base_diagnostics()
        trace1 = engine.evaluate(diag, snapshot_size=3)
        trace2 = engine.evaluate(diag, snapshot_size=3)
        assert len(trace1.signatures) == len(trace2.signatures)
        for s1, s2 in zip(trace1.signatures, trace2.signatures):
            assert s1.signature_hash == s2.signature_hash


# ── Test: diff_rule_sets ────────────────────────────────────────────────────

class TestDiffRuleSets:
    """Tests for DecisionRuleEngine.diff_rule_sets (RuleDiff output)."""

    def _rule(self, rule_id: str, **overrides) -> DecisionRule:
        defaults = dict(
            rule_id=rule_id,
            version="1.0.0",
            description="test",
            trigger_conditions=("default",),
            action="do_something",
            priority=50,
        )
        defaults.update(overrides)
        return DecisionRule(**defaults)

    def test_identical_sets_no_changes(self):
        rules = [self._rule("r1"), self._rule("r2")]
        diff = DecisionRuleEngine.diff_rule_sets(rules, rules)
        assert diff.added_rules == []
        assert diff.removed_rules == []
        assert diff.modified_rules == {}
        assert diff.n_unchanged == 2

    def test_added_rule_detected(self):
        before = [self._rule("r1")]
        after = [self._rule("r1"), self._rule("r2")]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert diff.added_rules == ["r2"]
        assert diff.removed_rules == []
        assert diff.n_unchanged == 1

    def test_removed_rule_detected(self):
        before = [self._rule("r1"), self._rule("r2")]
        after = [self._rule("r1")]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert diff.removed_rules == ["r2"]
        assert diff.added_rules == []
        assert diff.n_unchanged == 1

    def test_modified_version_detected(self):
        before = [self._rule("r1", version="1.0.0")]
        after = [self._rule("r1", version="2.0.0")]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert "r1" in diff.modified_rules
        assert "version" in diff.modified_rules["r1"]
        assert diff.modified_rules["r1"]["version"] == ("1.0.0", "2.0.0")
        assert diff.n_unchanged == 0

    def test_modified_action_detected(self):
        before = [self._rule("r1", action="old_action")]
        after = [self._rule("r1", action="new_action")]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert "action" in diff.modified_rules["r1"]
        assert diff.modified_rules["r1"]["action"] == ("old_action", "new_action")

    def test_modified_trigger_conditions_detected(self):
        before = [self._rule("r1", trigger_conditions=("cond_a",))]
        after = [self._rule("r1", trigger_conditions=("cond_a", "cond_b"))]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert "trigger_conditions" in diff.modified_rules["r1"]

    def test_modified_priority_detected(self):
        before = [self._rule("r1", priority=10)]
        after = [self._rule("r1", priority=99)]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert diff.modified_rules["r1"]["priority"] == (10, 99)

    def test_complex_diff_add_remove_modify(self):
        before = [
            self._rule("r1"),
            self._rule("r2", version="1.0.0"),
            self._rule("r3"),
        ]
        after = [
            self._rule("r2", version="2.0.0"),
            self._rule("r4"),
        ]
        diff = DecisionRuleEngine.diff_rule_sets(before, after)
        assert diff.added_rules == ["r4"]
        assert sorted(diff.removed_rules) == ["r1", "r3"]
        assert "r2" in diff.modified_rules
        assert diff.n_unchanged == 0

    def test_empty_sets(self):
        diff = DecisionRuleEngine.diff_rule_sets([], [])
        assert diff.added_rules == []
        assert diff.removed_rules == []
        assert diff.modified_rules == {}
        assert diff.n_unchanged == 0

    def test_diff_with_default_rule_set(self):
        """Diff the default engine rules against themselves."""
        engine = DecisionRuleEngine()
        rules = engine.rules
        diff = DecisionRuleEngine.diff_rule_sets(rules, rules)
        assert diff.n_unchanged == 10
        assert diff.added_rules == []
