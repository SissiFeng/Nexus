"""Tests for the optimization_copilot.hitl package.

Covers:
- PriorType enum values
- ExpertPrior creation and fields
- PriorRegistry.add_prior and get_priors
- PriorRegistry.apply_to_suggestions re-ranks candidates based on priors
- AutonomyLevel enum (MANUAL=0..AUTONOMOUS=3)
- AutonomyPolicy.escalate / de_escalate
- AutonomyPolicy.should_consult_human at different levels
- TrustTracker.record_outcome updates trust_score
- SteeringAction enum values
- SteeringDirective creation
- SteeringEngine.apply_directive with FOCUS_REGION
- SteeringEngine.apply_directive with AVOID_REGION filtering
- SteeringEngine.directive_history tracking
"""

from __future__ import annotations

from optimization_copilot.core.models import ParameterSpec, VariableType
from optimization_copilot.hitl import (
    AutonomyLevel,
    AutonomyPolicy,
    ExpertPrior,
    PriorRegistry,
    PriorType,
    SteeringAction,
    SteeringDirective,
    SteeringEngine,
    TrustTracker,
)


# ---------------------------------------------------------------------------
# PriorType tests
# ---------------------------------------------------------------------------


class TestPriorType:
    def test_enum_values(self):
        assert PriorType.GAUSSIAN.value == "gaussian"
        assert PriorType.UNIFORM.value == "uniform"
        assert PriorType.PREFERENCE.value == "preference"
        assert PriorType.CONSTRAINT.value == "constraint"
        assert PriorType.RANKING.value == "ranking"

    def test_enum_count(self):
        assert len(PriorType) == 5


# ---------------------------------------------------------------------------
# ExpertPrior tests
# ---------------------------------------------------------------------------


class TestExpertPrior:
    def test_creation_and_fields(self):
        prior = ExpertPrior(
            parameter_name="temperature",
            prior_type=PriorType.GAUSSIAN,
            mean=300.0,
            std=50.0,
            confidence=0.9,
            source="domain_expert",
        )
        assert prior.parameter_name == "temperature"
        assert prior.prior_type == PriorType.GAUSSIAN
        assert prior.mean == 300.0
        assert prior.std == 50.0
        assert prior.confidence == 0.9
        assert prior.source == "domain_expert"
        assert prior.lower is None
        assert prior.upper is None

    def test_default_values(self):
        prior = ExpertPrior(
            parameter_name="x",
            prior_type=PriorType.UNIFORM,
        )
        assert prior.confidence == 0.5
        assert prior.source == ""
        assert prior.mean is None
        assert prior.std is None


# ---------------------------------------------------------------------------
# PriorRegistry tests
# ---------------------------------------------------------------------------


class TestPriorRegistry:
    def test_add_prior_and_get_priors(self):
        registry = PriorRegistry()
        p1 = ExpertPrior(parameter_name="x", prior_type=PriorType.GAUSSIAN, mean=1.0, std=0.1)
        p2 = ExpertPrior(parameter_name="y", prior_type=PriorType.UNIFORM, lower=0, upper=10)
        p3 = ExpertPrior(parameter_name="x", prior_type=PriorType.CONSTRAINT, lower=0, upper=5)

        registry.add_prior(p1)
        registry.add_prior(p2)
        registry.add_prior(p3)

        x_priors = registry.get_priors("x")
        assert len(x_priors) == 2
        assert all(p.parameter_name == "x" for p in x_priors)

        y_priors = registry.get_priors("y")
        assert len(y_priors) == 1

        z_priors = registry.get_priors("z")
        assert len(z_priors) == 0

        assert registry.n_priors == 3

    def test_apply_to_suggestions_reranks_candidates(self):
        """A Gaussian prior centered at x=5.0 should rank x=5.0 above x=0.0."""
        registry = PriorRegistry()
        registry.add_prior(ExpertPrior(
            parameter_name="x",
            prior_type=PriorType.GAUSSIAN,
            mean=5.0,
            std=1.0,
            confidence=1.0,
        ))

        # Candidate far from mean listed first, close to mean listed second
        candidates = [
            {"x": 0.0},
            {"x": 5.0},
            {"x": 3.0},
        ]

        specs = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0, upper=10)]
        ranked = registry.apply_to_suggestions(candidates, specs)

        # x=5.0 (highest Gaussian score) should be first
        assert ranked[0]["x"] == 5.0
        # x=3.0 should rank above x=0.0 (closer to mean)
        assert ranked[1]["x"] == 3.0
        assert ranked[2]["x"] == 0.0

    def test_apply_to_suggestions_empty_priors_preserves_order(self):
        registry = PriorRegistry()
        candidates = [{"x": 1.0}, {"x": 2.0}]
        ranked = registry.apply_to_suggestions(candidates)
        assert ranked == candidates

    def test_apply_to_suggestions_with_constraint_prior(self):
        """CONSTRAINT prior penalizes out-of-bounds candidates."""
        registry = PriorRegistry()
        registry.add_prior(ExpertPrior(
            parameter_name="x",
            prior_type=PriorType.CONSTRAINT,
            lower=2.0,
            upper=8.0,
            confidence=1.0,
        ))

        candidates = [
            {"x": 1.0},   # out of bounds -> negative score
            {"x": 5.0},   # in bounds -> positive score
        ]
        ranked = registry.apply_to_suggestions(candidates)

        # In-bounds candidate should rank first
        assert ranked[0]["x"] == 5.0
        assert ranked[1]["x"] == 1.0


# ---------------------------------------------------------------------------
# AutonomyLevel tests
# ---------------------------------------------------------------------------


class TestAutonomyLevel:
    def test_enum_values(self):
        assert AutonomyLevel.MANUAL == 0
        assert AutonomyLevel.SUPERVISED == 1
        assert AutonomyLevel.COLLABORATIVE == 2
        assert AutonomyLevel.AUTONOMOUS == 3

    def test_ordering(self):
        assert AutonomyLevel.MANUAL < AutonomyLevel.SUPERVISED
        assert AutonomyLevel.SUPERVISED < AutonomyLevel.COLLABORATIVE
        assert AutonomyLevel.COLLABORATIVE < AutonomyLevel.AUTONOMOUS


# ---------------------------------------------------------------------------
# AutonomyPolicy tests
# ---------------------------------------------------------------------------


class TestAutonomyPolicy:
    def test_escalate(self):
        policy = AutonomyPolicy(initial_level=AutonomyLevel.MANUAL)
        assert policy.current_level == AutonomyLevel.MANUAL

        new_level = policy.escalate()
        assert new_level == AutonomyLevel.SUPERVISED
        assert policy.current_level == AutonomyLevel.SUPERVISED

        policy.escalate()
        assert policy.current_level == AutonomyLevel.COLLABORATIVE

        policy.escalate()
        assert policy.current_level == AutonomyLevel.AUTONOMOUS

        # Should be capped at AUTONOMOUS
        policy.escalate()
        assert policy.current_level == AutonomyLevel.AUTONOMOUS

    def test_de_escalate(self):
        policy = AutonomyPolicy(initial_level=AutonomyLevel.AUTONOMOUS)
        assert policy.current_level == AutonomyLevel.AUTONOMOUS

        new_level = policy.de_escalate()
        assert new_level == AutonomyLevel.COLLABORATIVE

        policy.de_escalate()
        assert policy.current_level == AutonomyLevel.SUPERVISED

        policy.de_escalate()
        assert policy.current_level == AutonomyLevel.MANUAL

        # Should be capped at MANUAL
        policy.de_escalate()
        assert policy.current_level == AutonomyLevel.MANUAL

    def test_should_consult_human_at_different_levels(self):
        policy = AutonomyPolicy(initial_level=AutonomyLevel.MANUAL)
        assert policy.should_consult_human("suggest") is True

        policy.set_level(AutonomyLevel.SUPERVISED)
        assert policy.should_consult_human("suggest") is True

        policy.set_level(AutonomyLevel.COLLABORATIVE)
        # Normal decisions -> no consultation needed
        assert policy.should_consult_human("suggest") is False
        # Critical decisions -> still need consultation
        assert policy.should_consult_human("critical") is True
        assert policy.should_consult_human("switch") is True

        policy.set_level(AutonomyLevel.AUTONOMOUS)
        assert policy.should_consult_human("suggest") is False
        assert policy.should_consult_human("critical") is False


# ---------------------------------------------------------------------------
# TrustTracker tests
# ---------------------------------------------------------------------------


class TestTrustTracker:
    def test_initial_trust_score(self):
        tracker = TrustTracker()
        assert tracker.trust_score == 0.5
        assert tracker.n_records == 0

    def test_record_outcome_updates_trust_score(self):
        tracker = TrustTracker(decay_rate=0.9)
        initial_trust = tracker.trust_score

        # Good outcome: AI suggested, human approved, high quality
        tracker.record_outcome(ai_suggested=True, human_approved=True, result_quality=1.0)
        assert tracker.trust_score > initial_trust
        assert tracker.n_records == 1
        assert tracker.consecutive_good == 1

    def test_bad_outcome_decreases_trust(self):
        tracker = TrustTracker(decay_rate=0.9)
        # Start with a good outcome to raise trust
        tracker.record_outcome(ai_suggested=True, human_approved=True, result_quality=0.8)
        trust_after_good = tracker.trust_score

        # Human rejects AI suggestion -> trust goes down
        tracker.record_outcome(ai_suggested=True, human_approved=False, result_quality=0.0)
        assert tracker.trust_score < trust_after_good
        assert tracker.consecutive_good == 0

    def test_should_escalate(self):
        tracker = TrustTracker(decay_rate=0.5)
        # Record many good outcomes to push trust up
        for _ in range(10):
            tracker.record_outcome(ai_suggested=True, human_approved=True, result_quality=1.0)

        assert tracker.trust_score > 0.8
        assert tracker.consecutive_good >= 5
        assert tracker.should_escalate(threshold=0.8, min_consecutive=5) is True

    def test_should_de_escalate(self):
        tracker = TrustTracker(decay_rate=0.5)
        # Record many bad outcomes to push trust down
        for _ in range(10):
            tracker.record_outcome(ai_suggested=True, human_approved=False, result_quality=0.0)

        assert tracker.trust_score < 0.3
        assert tracker.should_de_escalate(threshold=0.3) is True


# ---------------------------------------------------------------------------
# SteeringAction tests
# ---------------------------------------------------------------------------


class TestSteeringAction:
    def test_enum_values(self):
        assert SteeringAction.ACCEPT.value == "accept"
        assert SteeringAction.REJECT.value == "reject"
        assert SteeringAction.MODIFY.value == "modify"
        assert SteeringAction.FOCUS_REGION.value == "focus_region"
        assert SteeringAction.AVOID_REGION.value == "avoid_region"
        assert SteeringAction.CHANGE_OBJECTIVE.value == "change_objective"

    def test_enum_count(self):
        assert len(SteeringAction) == 6


# ---------------------------------------------------------------------------
# SteeringDirective tests
# ---------------------------------------------------------------------------


class TestSteeringDirective:
    def test_creation(self):
        directive = SteeringDirective(
            action=SteeringAction.FOCUS_REGION,
            region_bounds={"x": (1.0, 5.0)},
            reason="Expert believes optimal region is near x=3",
        )
        assert directive.action == SteeringAction.FOCUS_REGION
        assert directive.region_bounds == {"x": (1.0, 5.0)}
        assert "optimal region" in directive.reason

    def test_default_values(self):
        directive = SteeringDirective(action=SteeringAction.ACCEPT)
        assert directive.parameters == {}
        assert directive.region_bounds is None
        assert directive.reason == ""
        assert directive.timestamp == 0.0


# ---------------------------------------------------------------------------
# SteeringEngine tests
# ---------------------------------------------------------------------------


class TestSteeringEngine:
    def _make_candidates(self) -> list[dict[str, float]]:
        return [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
            {"x": 5.0, "y": 6.0},
            {"x": 7.0, "y": 8.0},
        ]

    def test_apply_directive_focus_region(self):
        engine = SteeringEngine()
        candidates = self._make_candidates()

        directive = SteeringDirective(
            action=SteeringAction.FOCUS_REGION,
            region_bounds={"x": (2.0, 6.0)},
        )
        result = engine.apply_directive(directive, candidates)

        # Only candidates with x in [2.0, 6.0] should remain
        assert len(result) == 2
        assert all(2.0 <= c["x"] <= 6.0 for c in result)

    def test_apply_directive_avoid_region(self):
        engine = SteeringEngine()
        candidates = self._make_candidates()

        directive = SteeringDirective(
            action=SteeringAction.AVOID_REGION,
            region_bounds={"x": (2.0, 6.0)},
        )
        result = engine.apply_directive(directive, candidates)

        # Candidates with x in [2.0, 6.0] should be filtered out
        assert len(result) == 2
        assert all(c["x"] < 2.0 or c["x"] > 6.0 for c in result)

    def test_apply_directive_accept_returns_all(self):
        engine = SteeringEngine()
        candidates = self._make_candidates()

        directive = SteeringDirective(action=SteeringAction.ACCEPT)
        result = engine.apply_directive(directive, candidates)
        assert len(result) == len(candidates)

    def test_apply_directive_reject_returns_empty(self):
        engine = SteeringEngine()
        candidates = self._make_candidates()

        directive = SteeringDirective(action=SteeringAction.REJECT)
        result = engine.apply_directive(directive, candidates)
        assert len(result) == 0

    def test_apply_directive_modify(self):
        engine = SteeringEngine()
        candidates = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]

        directive = SteeringDirective(
            action=SteeringAction.MODIFY,
            parameters={"x": 99.0},
        )
        result = engine.apply_directive(directive, candidates)
        assert len(result) == 2
        assert all(c["x"] == 99.0 for c in result)
        # y should remain unchanged
        assert result[0]["y"] == 2.0
        assert result[1]["y"] == 4.0

    def test_directive_history_tracking(self):
        engine = SteeringEngine()
        assert len(engine.directive_history) == 0
        assert engine.n_directives == 0

        d1 = SteeringDirective(action=SteeringAction.ACCEPT)
        d2 = SteeringDirective(
            action=SteeringAction.FOCUS_REGION,
            region_bounds={"x": (0, 10)},
        )

        engine.apply_directive(d1, [])
        engine.apply_directive(d2, [{"x": 5.0}])

        history = engine.directive_history
        assert len(history) == 2
        assert engine.n_directives == 2
        assert history[0].action == SteeringAction.ACCEPT
        assert history[1].action == SteeringAction.FOCUS_REGION
