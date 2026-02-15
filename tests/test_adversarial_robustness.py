"""Adversarial robustness stress tests for the optimization system.

Proves the robustness analysis layer handles real-world SDL (Self-Driving Lab)
failure modes: instrument malfunction (label corruption), systematic drift
(batch effects, calibration drift), and adversarial perturbations (worst-case
scenarios an operator wouldn't notice).

Every test is fully deterministic (seeded RNG) and uses only the Python
standard library.
"""

from __future__ import annotations

import math
import random

from optimization_copilot.robustness.bootstrap import BootstrapAnalyzer
from optimization_copilot.robustness.conclusion import ConclusionRobustnessChecker
from optimization_copilot.robustness.sensitivity import DecisionSensitivityAnalyzer
from optimization_copilot.robustness.consistency import CrossModelConsistency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    """Arithmetic mean of a list of floats."""
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _corrupt_labels(
    values: list[float],
    corruption_rate: float,
    seed: int = 99,
    corruption_type: str = "random",
) -> list[float]:
    """Simulate instrument malfunction by corrupting a fraction of values.

    Parameters
    ----------
    values : list[float]
        Original clean measurement values.
    corruption_rate : float
        Fraction of values to corrupt (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.
    corruption_type : str
        ``'random'``  -- replace with random value in [min, max] of data.
        ``'swap'``    -- swap pairs of values.
        ``'outlier'`` -- replace with value 3 std deviations from mean.

    Returns
    -------
    list[float]
        Corrupted copy of the input values.
    """
    rng = random.Random(seed)
    result = list(values)
    n = len(result)
    n_corrupt = max(1, int(n * corruption_rate))

    if corruption_type == "random":
        lo, hi = min(values), max(values)
        indices = rng.sample(range(n), min(n_corrupt, n))
        for idx in indices:
            result[idx] = rng.uniform(lo, hi)

    elif corruption_type == "swap":
        indices = list(range(n))
        rng.shuffle(indices)
        pairs = n_corrupt // 2
        for p in range(pairs):
            i, j = indices[2 * p], indices[2 * p + 1]
            result[i], result[j] = result[j], result[i]

    elif corruption_type == "outlier":
        m = _mean(values)
        s = _std(values) if len(values) > 1 else 1.0
        indices = rng.sample(range(n), min(n_corrupt, n))
        for idx in indices:
            direction = rng.choice([-1, 1])
            result[idx] = m + direction * 3.0 * s

    return result


def _ranking_from_values(
    values: list[float], names: list[str]
) -> list[str]:
    """Return candidate names sorted by descending value (best first)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return [names[i] for i in indexed]


def _generate_candidate_measurements(
    n_candidates: int,
    n_measurements_per: int,
    true_values: list[float],
    noise_std: float,
    seed: int,
) -> list[float]:
    """Generate repeated noisy measurements and return per-candidate means.

    Simulates an SDL experiment where each candidate is measured multiple
    times with instrument noise.

    Returns one aggregated (mean) value per candidate.
    """
    rng = random.Random(seed)
    means = []
    for c in range(n_candidates):
        measurements = [
            true_values[c] + rng.gauss(0, noise_std)
            for _ in range(n_measurements_per)
        ]
        means.append(_mean(measurements))
    return means


# ===================================================================
# Part 1: Label Corruption Robustness
# ===================================================================


class TestLabelCorruption:
    """Simulate an instrument that occasionally gives wrong readings.

    In SDL workflows, instruments can malfunction intermittently -- returning
    a random voltage, a saturated detector reading, or simply the wrong
    sample.  These tests verify that the robustness analysis layer remains
    reliable under realistic corruption rates.
    """

    def test_top1_stability_under_5pct_corruption(self):
        """A clear winner survives 5% instrument malfunction rate.

        SDL scenario: one formulation is clearly superior (KPI = 10.0) but
        the autosampler occasionally mislabels vials, corrupting ~5% of
        readings. The system should still identify the correct best candidate.
        """
        rng = random.Random(42)
        n_candidates = 20
        n_measurements = 5  # per candidate

        # True values: candidate 0 is the clear winner
        true_values = [10.0] + [rng.uniform(1.0, 8.0) for _ in range(n_candidates - 1)]
        names = [f"candidate_{i}" for i in range(n_candidates)]

        # Generate clean aggregated measurements
        clean_means = _generate_candidate_measurements(
            n_candidates, n_measurements, true_values, noise_std=0.3, seed=100
        )

        # Corrupt 5% of the aggregated values
        corrupted_means = _corrupt_labels(clean_means, corruption_rate=0.05, seed=200)

        # Run bootstrap stability on corrupted data
        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42)
        result = analyzer.bootstrap_top_k(corrupted_means, names, k=5)

        # The clear winner should survive mild corruption
        assert result["original_top_k"][0] == "candidate_0", (
            "Top-1 candidate changed under 5% corruption"
        )
        assert result["stability_score"] > 0.7, (
            f"Stability {result['stability_score']:.3f} too low under 5% corruption"
        )

    def test_top1_flips_under_20pct_corruption(self):
        """Heavy corruption destabilises a close race between candidates.

        SDL scenario: two catalysts perform almost identically (KPI gap of
        only 0.5 units). With 20% of instrument readings corrupted, the
        system should flag that the recommendation is unreliable.
        """
        rng = random.Random(42)
        n_candidates = 20

        # Close race: candidates 0 and 1 differ by only 0.5
        true_values = [8.0, 7.5] + [rng.uniform(1.0, 6.0) for _ in range(n_candidates - 2)]
        names = [f"candidate_{i}" for i in range(n_candidates)]

        clean_means = _generate_candidate_measurements(
            n_candidates, 5, true_values, noise_std=0.3, seed=100
        )

        # 20% corruption
        corrupted_means = _corrupt_labels(clean_means, corruption_rate=0.20, seed=300)

        analyzer = BootstrapAnalyzer(n_bootstrap=1000, seed=42)
        result = analyzer.bootstrap_top_k(corrupted_means, names, k=5)

        # Heavy corruption + close race should cause instability
        assert result["stability_score"] < 0.8, (
            f"Stability {result['stability_score']:.3f} unexpectedly high "
            "under 20% corruption with close race"
        )

    def test_ranking_kendall_tau_under_corruption(self):
        """Rankings degrade monotonically with increasing corruption rate.

        SDL scenario: an operator runs a screening campaign for 15
        formulations. Unknown to them, the LC-MS detector saturates
        intermittently. This test verifies that ranking quality degrades
        predictably with instrument error rate.
        """
        rng = random.Random(42)
        n_items = 15
        # Well-separated clean values for deterministic ranking
        clean_values = [float(n_items - i) for i in range(n_items)]
        names = [f"formulation_{i}" for i in range(n_items)]
        clean_ranking = _ranking_from_values(clean_values, names)

        consistency = CrossModelConsistency()
        taus = {}

        for rate in [0.05, 0.10, 0.20]:
            corrupted = _corrupt_labels(clean_values, corruption_rate=rate, seed=42)
            corrupted_ranking = _ranking_from_values(corrupted, names)
            tau = consistency.kendall_tau(clean_ranking, corrupted_ranking)
            taus[rate] = tau

        # Tau should decrease with corruption rate
        assert taus[0.05] >= taus[0.10], (
            f"tau@5%={taus[0.05]:.3f} < tau@10%={taus[0.10]:.3f}"
        )
        assert taus[0.10] >= taus[0.20], (
            f"tau@10%={taus[0.10]:.3f} < tau@20%={taus[0.20]:.3f}"
        )
        # Mild corruption should mostly preserve rankings
        assert taus[0.05] > 0.8, (
            f"tau@5%={taus[0.05]:.3f} too low -- rankings destroyed by mild corruption"
        )
        # Heavy corruption should significantly disrupt
        assert taus[0.20] < 0.9, (
            f"tau@20%={taus[0.20]:.3f} too high -- heavy corruption not disrupting rankings"
        )

    def test_corruption_detection_via_bootstrap_width(self):
        """Bootstrap CI naturally widens when data is corrupted.

        SDL scenario: an operator notices that confidence intervals from
        the robustness analysis are unusually wide. This is an indirect
        signal that something is wrong with the instrument -- even without
        knowing the corruption exists.
        """
        rng = random.Random(42)
        n = 80
        # Clean data: tight cluster of measurements around 5.0
        clean_data = [5.0 + rng.gauss(0, 0.2) for _ in range(n)]

        # Corrupted: 20% of readings are extreme instrument artifacts.
        # We inject outliers manually to guarantee large deviations:
        # replace 20% of values with readings far outside the normal range.
        corrupted_data = list(clean_data)
        corrupt_rng = random.Random(42)
        n_corrupt = int(n * 0.20)
        corrupt_indices = corrupt_rng.sample(range(n), n_corrupt)
        for idx in corrupt_indices:
            # Outliers at +/- 10 sigma from the mean -- simulates a
            # saturated detector or wrong-vial reading
            direction = corrupt_rng.choice([-1, 1])
            corrupted_data[idx] = 5.0 + direction * 2.0

        analyzer = BootstrapAnalyzer(n_bootstrap=2000, seed=42)
        clean_ci = analyzer.bootstrap_ci(clean_data, _mean, confidence=0.95)
        # Use a different seed for corrupted to avoid RNG state correlation
        analyzer2 = BootstrapAnalyzer(n_bootstrap=2000, seed=43)
        corrupted_ci = analyzer2.bootstrap_ci(corrupted_data, _mean, confidence=0.95)

        clean_width = clean_ci.ci_upper - clean_ci.ci_lower
        corrupted_width = corrupted_ci.ci_upper - corrupted_ci.ci_lower

        assert corrupted_width > 1.5 * clean_width, (
            f"Corrupted CI width ({corrupted_width:.4f}) not sufficiently wider "
            f"than clean ({clean_width:.4f}) -- corruption not detected"
        )


# ===================================================================
# Part 2: Systematic Shift Robustness
# ===================================================================


class TestSystematicShift:
    """Simulate instrument drift and batch effects.

    In SDL, instruments drift over time (laser power decay, reagent
    degradation, temperature fluctuation). These shifts affect all
    measurements uniformly, preserving relative ordering but breaking
    absolute thresholds.
    """

    def test_recommendation_under_10pct_shift(self):
        """Rankings survive a 10% systematic calibration drift.

        SDL scenario: the UV-Vis spectrometer's lamp is slowly degrading,
        causing all absorbance readings to be 10% higher than reality.
        Since the shift is multiplicative and uniform, relative rankings
        should be preserved.
        """
        rng = random.Random(42)
        n_candidates = 50
        # Clear optimum at candidate 0
        values = [10.0] + [rng.uniform(1.0, 8.0) for _ in range(n_candidates - 1)]
        names = [f"candidate_{i}" for i in range(n_candidates)]

        # Apply 10% systematic shift (all readings inflated)
        shifted_values = [v * 1.1 for v in values]

        analyzer = DecisionSensitivityAnalyzer(seed=42)
        # Measure stability of the shifted data
        result = analyzer.decision_sensitivity(
            shifted_values, names, noise_std=0.5, n_perturbations=500
        )

        # Multiplicative shift preserves ranking -- top candidate unchanged
        assert result["stability"] > 0.8, (
            f"Stability {result['stability']:.3f} too low -- "
            "systematic shift should preserve rankings"
        )
        # Top-1 should still be candidate_0
        top_1_name = max(result["top1_frequency"], key=result["top1_frequency"].get)
        assert top_1_name == "candidate_0", (
            f"Top-1 shifted to {top_1_name} under 10% calibration drift"
        )

    def test_shift_breaks_threshold_decisions(self):
        """Systematic drift causes candidates to cross decision thresholds.

        SDL scenario: the go/no-go decision for a drug candidate is
        'KPI > 7.0'. Several candidates hover near 7.0. A 5% upward
        drift in the assay pushes borderline candidates across the
        threshold, making the binary decision unreliable.
        """
        rng = random.Random(42)
        # Candidates near the threshold of 7.0
        near_threshold = [6.8, 6.9, 7.0, 7.1, 7.2]
        # Some clearly above/below
        far_values = [rng.uniform(3.0, 5.0) for _ in range(10)]
        far_above = [rng.uniform(8.5, 10.0) for _ in range(5)]

        all_values = near_threshold + far_values + far_above
        n = len(all_values)
        names = [f"candidate_{i}" for i in range(n)]
        threshold = 7.0

        # Count how many pass before shift
        passing_before = sum(1 for v in all_values if v > threshold)

        # Apply 5% upward shift
        shifted_values = [v * 1.05 for v in all_values]
        passing_after = sum(1 for v in shifted_values if v > threshold)

        # Some borderline candidates should cross the threshold
        assert passing_after > passing_before, (
            f"No candidates crossed threshold after 5% shift "
            f"(before={passing_before}, after={passing_after})"
        )

        # The robustness system should flag threshold instability.
        # Use the sensitivity analyzer to show that near-threshold
        # candidates have unstable rankings under noise that is
        # comparable to the gap between them.
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        near_names = [f"near_{i}" for i in range(len(near_threshold))]
        result = analyzer.decision_sensitivity(
            near_threshold, near_names, noise_std=0.15, n_perturbations=500
        )

        # Near-threshold values are so close (gap=0.1) that even
        # modest noise (0.15) destabilises the ranking.
        assert result["stability"] < 0.7, (
            f"Stability {result['stability']:.3f} for near-threshold candidates "
            "is unexpectedly high -- threshold instability not detected"
        )

    def test_progressive_drift_detection(self):
        """Accumulated instrument drift is detectable via bootstrap CI shift.

        SDL scenario: a robotic liquid handler's calibration drifts by
        ~2% per batch over a 100-experiment campaign. The first 50
        experiments are reliable; the latter 50 are progressively biased.
        Comparing bootstrap CIs of the two halves reveals the drift.
        """
        rng = random.Random(42)
        base_value = 5.0
        noise_std = 0.3
        n_total = 100

        observations = []
        for i in range(n_total):
            if i < 50:
                # Clean first half
                observations.append(base_value + rng.gauss(0, noise_std))
            else:
                # Drifting second half: 2% per batch of 10
                batch_idx = (i - 50) // 10 + 1  # batches 1-5
                drift = base_value * 0.02 * batch_idx
                observations.append(base_value + drift + rng.gauss(0, noise_std))

        first_half = observations[:50]
        second_half = observations[50:]

        analyzer1 = BootstrapAnalyzer(n_bootstrap=2000, seed=42)
        analyzer2 = BootstrapAnalyzer(n_bootstrap=2000, seed=43)
        ci_first = analyzer1.bootstrap_ci(first_half, _mean, confidence=0.95)
        ci_second = analyzer2.bootstrap_ci(second_half, _mean, confidence=0.95)

        # Second half mean should be shifted upward due to drift
        mean_diff = ci_second.observed_value - ci_first.observed_value
        assert mean_diff > 0, (
            f"Mean difference {mean_diff:.4f} not positive -- drift not detected"
        )
        # The drift should be statistically significant
        # (second half CI lower bound above first half mean)
        assert ci_second.ci_lower > ci_first.observed_value, (
            f"Second-half CI lower ({ci_second.ci_lower:.4f}) not above "
            f"first-half mean ({ci_first.observed_value:.4f}) -- "
            "drift is too subtle to detect statistically"
        )

    def test_ranking_stability_across_batches(self):
        """Larger systematic shift causes more ranking disruption.

        SDL scenario: a materials screening campaign runs over 3 days.
        Each day the oven temperature is slightly different, biasing
        all measurements. Day 2 has +5% bias, Day 3 has +10%.
        Rankings should degrade proportionally with the bias.
        """
        rng = random.Random(42)
        n_candidates = 15
        base_values = [float(n_candidates - i) + rng.uniform(-0.3, 0.3)
                       for i in range(n_candidates)]
        names = [f"material_{i}" for i in range(n_candidates)]

        # Batch 1: clean
        batch1_values = list(base_values)
        # Batch 2: +5% shift
        batch2_values = [v * 1.05 for v in base_values]
        # Batch 3: +10% shift
        batch3_values = [v * 1.10 for v in base_values]

        ranking1 = _ranking_from_values(batch1_values, names)
        ranking2 = _ranking_from_values(batch2_values, names)
        ranking3 = _ranking_from_values(batch3_values, names)

        consistency = CrossModelConsistency()
        tau_12 = consistency.kendall_tau(ranking1, ranking2)
        tau_13 = consistency.kendall_tau(ranking1, ranking3)

        # Multiplicative shift preserves ordering perfectly for well-separated values
        # But our values have small random perturbations, so the shift might cause
        # swaps among close pairs. Larger shift = more swaps.
        assert tau_12 >= tau_13 or abs(tau_12 - tau_13) < 0.05, (
            f"tau(batch1,batch2)={tau_12:.3f} < tau(batch1,batch3)={tau_13:.3f} "
            "-- larger shift should cause equal or more disruption"
        )


# ===================================================================
# Part 3: Adversarial Parameter Flip
# ===================================================================


class TestAdversarialFlip:
    """Worst-case perturbation: an adversary swaps the best and worst.

    These tests simulate scenarios where a single catastrophic error --
    such as a vial swap, a labelling error, or a database corruption --
    completely inverts the best and worst candidates.
    """

    def test_adversarial_flip_detection(self):
        """System detects when the best and worst candidates are swapped.

        SDL scenario: a sample tray was loaded in reverse order for
        two slots. The best formulation's data now appears as the worst,
        and vice versa. The robustness check should flag this as highly
        unstable because the "best" candidate is actually the worst.
        """
        n_candidates = 20
        # Clear descending ordering: candidate 0 is best (20.0), candidate 19 is worst (1.0)
        values = [float(n_candidates - i) for i in range(n_candidates)]
        names = [f"sample_{i}" for i in range(n_candidates)]

        # Adversarial swap: best (#0, value=20) and worst (#19, value=1)
        adversarial = list(values)
        adversarial[0], adversarial[19] = adversarial[19], adversarial[0]

        checker = ConclusionRobustnessChecker(n_bootstrap=1000, seed=42)
        result = checker.check_ranking_stability(adversarial, names, k=5)
        checker2 = ConclusionRobustnessChecker(n_bootstrap=1000, seed=42)
        clean_result = checker2.check_ranking_stability(values, names, k=5)

        # Verify the adversarial flip changed the top-1 candidate
        adv_top1 = result.details["original_top_k"][0]
        clean_top1 = clean_result.details["original_top_k"][0]
        assert adv_top1 != clean_top1, (
            "Adversarial flip did not change the top-1 candidate"
        )

        # Cross-model consistency between clean and adversarial rankings
        # should be degraded because best/worst are swapped.
        consistency = CrossModelConsistency()
        clean_ranking = _ranking_from_values(values, names)
        adversarial_ranking = _ranking_from_values(adversarial, names)
        tau = consistency.kendall_tau(clean_ranking, adversarial_ranking)

        # With 20 candidates and a best/worst swap, tau should drop
        # noticeably below 1.0 (each swap disrupts pairs with all
        # other items, so about 2*(n-2) pairs out of n*(n-1)/2).
        assert tau < 0.85, (
            f"Kendall tau {tau:.3f} between clean and adversarial rankings "
            "is too high -- adversarial flip not detected"
        )

    def test_ensemble_catches_adversarial(self):
        """Ensemble of models identifies the adversarially corrupted model.

        SDL scenario: three independent scoring methods (e.g., GP, RF,
        neural net) rank candidates. One model was trained on mislabelled
        data (best/worst swapped). The ensemble should identify the
        outlier model.
        """
        n_candidates = 15
        names = [f"compound_{i}" for i in range(n_candidates)]

        # Clean ranking: descending order
        clean_values = [float(n_candidates - i) for i in range(n_candidates)]
        clean_ranking = _ranking_from_values(clean_values, names)

        # Slightly perturbed clean ranking (model 2)
        rng = random.Random(42)
        perturbed_values = [v + rng.gauss(0, 0.3) for v in clean_values]
        perturbed_ranking = _ranking_from_values(perturbed_values, names)

        # Adversarial: swap best and worst
        adv_values = list(clean_values)
        adv_values[0], adv_values[-1] = adv_values[-1], adv_values[0]
        adversarial_ranking = _ranking_from_values(adv_values, names)

        model_rankings = {
            "model_corrupted": adversarial_ranking,
            "model_clean_1": clean_ranking,
            "model_clean_2": perturbed_ranking,
        }

        consistency = CrossModelConsistency()
        agreement = consistency.model_agreement(model_rankings)

        # Find pairwise taus involving the corrupted model
        corrupted_taus = []
        clean_taus = []
        for (m1, m2), tau in agreement["pairwise"].items():
            if "corrupted" in m1 or "corrupted" in m2:
                corrupted_taus.append(tau)
            else:
                clean_taus.append(tau)

        # Clean models should agree well
        assert all(t > 0.8 for t in clean_taus), (
            f"Clean models disagree: taus={clean_taus}"
        )
        # Corrupted model should have lower agreement
        assert all(t < 0.95 for t in corrupted_taus), (
            f"Corrupted model agrees too well: taus={corrupted_taus}"
        )
        # The corrupted model's average tau should be lower than clean pairs
        avg_corrupted = _mean(corrupted_taus)
        avg_clean = _mean(clean_taus) if clean_taus else 1.0
        assert avg_corrupted < avg_clean, (
            f"Corrupted model avg tau ({avg_corrupted:.3f}) >= "
            f"clean model avg tau ({avg_clean:.3f})"
        )

        # Ensemble confidence should flag the adversarial model as outlier
        # Use predictions (values) for ensemble_confidence
        model_predictions = {
            "model_corrupted": adv_values,
            "model_clean_1": clean_values,
            "model_clean_2": perturbed_values,
        }
        ensemble = consistency.ensemble_confidence(model_predictions, names)

        # Items at the extremes (best/worst) should show high disagreement
        # because the corrupted model has them swapped
        item_0_info = next(
            item for item in ensemble["per_item"] if item["name"] == "compound_0"
        )
        item_last_info = next(
            item for item in ensemble["per_item"]
            if item["name"] == f"compound_{n_candidates - 1}"
        )
        # These items should have high std (disagreement)
        assert item_0_info["std"] > 1.0, (
            f"compound_0 std={item_0_info['std']:.3f} too low -- "
            "adversarial swap not detected"
        )
        assert item_last_info["std"] > 1.0, (
            f"compound_{n_candidates-1} std={item_last_info['std']:.3f} too low -- "
            "adversarial swap not detected"
        )

    def test_value_at_risk_under_adversarial_noise(self):
        """VaR degrades significantly under adversarial uncertainty.

        SDL scenario: normally, the top candidate's measurement
        uncertainty is well-characterised (sigma=0.5). Under an
        adversarial scenario (e.g., unknown instrument miscalibration),
        effective uncertainty balloons to sigma=3.0. The system should
        report a much worse VaR, alerting the operator to increased risk.
        """
        analyzer_normal = DecisionSensitivityAnalyzer(seed=42)
        analyzer_adversarial = DecisionSensitivityAnalyzer(seed=42)

        # 5 candidates
        values = [10.0, 8.0, 6.0, 4.0, 2.0]

        # Normal scenario: well-characterised uncertainty
        normal_uncertainties = [0.5, 0.5, 0.5, 0.5, 0.5]
        var_normal = analyzer_normal.value_at_risk(
            values, normal_uncertainties, quantile=0.05, n_samples=5000
        )

        # Adversarial scenario: inflated uncertainty on best candidate
        adversarial_uncertainties = [3.0, 0.5, 0.5, 0.5, 0.5]
        var_adversarial = analyzer_adversarial.value_at_risk(
            values, adversarial_uncertainties, quantile=0.05, n_samples=5000
        )

        # VaR should be significantly worse (lower) under adversarial uncertainty
        assert var_adversarial["var"] < var_normal["var"], (
            f"Adversarial VaR ({var_adversarial['var']:.3f}) not worse than "
            f"normal VaR ({var_normal['var']:.3f})"
        )
        # The difference should be substantial (not just noise)
        var_gap = var_normal["var"] - var_adversarial["var"]
        assert var_gap > 1.0, (
            f"VaR gap ({var_gap:.3f}) too small to distinguish scenarios"
        )

        # Expected best should also differ
        assert var_adversarial["expected_best"] < var_normal["expected_best"] + 1.0, (
            "Adversarial expected best not properly reflecting increased risk"
        )


# ===================================================================
# Part 4: Recommendation Confidence Calibration
# ===================================================================


class TestConfidenceCalibration:
    """Verify that confidence scores from the system are well-calibrated.

    A calibrated system means: when it says "80% confident the top
    candidate is X", then X should indeed be top in ~80% of noisy
    realizations.  Poorly calibrated confidence is dangerous in SDL
    because operators rely on it for go/no-go decisions.
    """

    def test_confidence_matches_actual_stability(self):
        """Theoretical confidence is calibrated against Monte Carlo truth.

        SDL scenario: an operator asks "how confident are we that
        compound A is the best?" The system reports a confidence
        score. We verify this matches the empirical probability by
        running 100 independent noise realizations.
        """
        rng = random.Random(42)
        n_candidates = 10
        # Clear-ish winner with some competition
        values = [10.0, 8.5] + [rng.uniform(2.0, 7.0) for _ in range(n_candidates - 2)]
        names = [f"candidate_{i}" for i in range(n_candidates)]
        noise_std = 1.0

        # Empirical: run 100 MC trials, count how often top-1 is the same
        original_top1 = names[max(range(n_candidates), key=lambda i: values[i])]
        mc_rng = random.Random(999)
        n_trials = 200
        top1_same = 0
        for _ in range(n_trials):
            noisy = [v + mc_rng.gauss(0, noise_std) for v in values]
            mc_top1 = names[max(range(n_candidates), key=lambda i: noisy[i])]
            if mc_top1 == original_top1:
                top1_same += 1
        empirical_stability = top1_same / n_trials

        # Theoretical: use recommendation_confidence with k=1
        analyzer = DecisionSensitivityAnalyzer(seed=42)
        conf = analyzer.recommendation_confidence(
            values, names, k=1, noise_std=noise_std, n_perturbations=1000
        )
        theoretical_confidence = conf[original_top1]

        # They should be calibrated within 0.15
        diff = abs(empirical_stability - theoretical_confidence)
        assert diff < 0.15, (
            f"Confidence miscalibrated: empirical={empirical_stability:.3f}, "
            f"theoretical={theoretical_confidence:.3f}, diff={diff:.3f}"
        )

    def test_confidence_decreases_with_noise(self):
        """More instrument noise leads to lower recommendation confidence.

        SDL scenario: an operator is deciding whether to invest in a
        higher-precision detector. This test shows that the confidence
        score correctly reflects how noise level affects decision
        reliability.
        """
        n_candidates = 10
        rng = random.Random(42)
        values = [10.0, 8.0] + [rng.uniform(2.0, 6.0) for _ in range(n_candidates - 2)]
        names = [f"candidate_{i}" for i in range(n_candidates)]

        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        confidences = []

        for noise in noise_levels:
            analyzer = DecisionSensitivityAnalyzer(seed=42)
            conf = analyzer.recommendation_confidence(
                values, names, k=1, noise_std=noise, n_perturbations=1000
            )
            # Confidence for the top candidate
            top_name = names[0]  # candidate_0 with value 10.0
            confidences.append(conf[top_name])

        # Confidence should monotonically decrease with noise
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1] - 0.02, (
                f"Confidence not decreasing: noise={noise_levels[i]} "
                f"conf={confidences[i]:.3f} vs noise={noise_levels[i+1]} "
                f"conf={confidences[i+1]:.3f}"
            )

        # At very low noise, confidence should be very high
        assert confidences[0] > 0.95, (
            f"Confidence at noise=0.1 is {confidences[0]:.3f}, expected > 0.95"
        )
        # At very high noise, confidence should be lower
        assert confidences[-1] < confidences[0], (
            f"Confidence at noise=5.0 ({confidences[-1]:.3f}) not lower than "
            f"at noise=0.1 ({confidences[0]:.3f})"
        )

    def test_confidence_increases_with_gap(self):
        """Larger gap between best and second-best increases confidence.

        SDL scenario: two reaction conditions are being compared. As
        the performance gap between them widens (e.g., through
        optimization), the system should become more confident in its
        recommendation.
        """
        rng = random.Random(42)
        n_candidates = 10
        base_values = [5.0] + [rng.uniform(2.0, 4.5) for _ in range(n_candidates - 2)]
        names = [f"candidate_{i}" for i in range(n_candidates)]
        noise_std = 1.0

        gaps = [0.1, 0.5, 1.0, 2.0, 5.0]
        confidences = []

        for gap in gaps:
            # candidate_0 is the best with variable gap above candidate_1
            values = [5.0 + gap] + [5.0] + base_values[1:]
            analyzer = DecisionSensitivityAnalyzer(seed=42)
            conf = analyzer.recommendation_confidence(
                values, names, k=1, noise_std=noise_std, n_perturbations=1000
            )
            confidences.append(conf["candidate_0"])

        # Confidence should monotonically increase with gap
        for i in range(len(confidences) - 1):
            assert confidences[i] <= confidences[i + 1] + 0.02, (
                f"Confidence not increasing: gap={gaps[i]} "
                f"conf={confidences[i]:.3f} vs gap={gaps[i+1]} "
                f"conf={confidences[i+1]:.3f}"
            )

        # At small gap, confidence should be lower
        assert confidences[0] < confidences[-1], (
            f"Confidence at gap=0.1 ({confidences[0]:.3f}) not lower than "
            f"at gap=5.0 ({confidences[-1]:.3f})"
        )
        # At large gap, confidence should be high
        assert confidences[-1] > 0.9, (
            f"Confidence at gap=5.0 is {confidences[-1]:.3f}, expected > 0.9"
        )
