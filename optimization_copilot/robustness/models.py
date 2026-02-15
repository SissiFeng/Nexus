"""Data models for robustness analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BootstrapResult:
    """Result of a bootstrap confidence interval computation.

    Attributes
    ----------
    statistic_name : str
        Human-readable name of the statistic being bootstrapped.
    observed_value : float
        Value of the statistic computed on the original data.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    confidence_level : float
        Confidence level, e.g. 0.95 for a 95% CI.
    n_bootstrap : int
        Number of bootstrap resamples used.
    bootstrap_distribution : list[float]
        The full distribution of bootstrapped statistic values.
    std_error : float
        Standard error of the bootstrap distribution.
    """

    statistic_name: str
    observed_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    bootstrap_distribution: list[float] = field(default_factory=list)
    std_error: float = 0.0


@dataclass
class ConclusionRobustness:
    """Assessment of how robust a scientific conclusion is under resampling.

    Attributes
    ----------
    conclusion_type : str
        Type of conclusion: ``"ranking"``, ``"importance"``, or ``"pareto"``.
    stability_score : float
        Score from 0.0 (completely unstable) to 1.0 (perfectly stable).
    n_bootstrap : int
        Number of bootstrap resamples used for the assessment.
    details : dict
        Additional information specific to the conclusion type.
    """

    conclusion_type: str
    stability_score: float
    n_bootstrap: int
    details: dict = field(default_factory=dict)


@dataclass
class RobustnessReport:
    """Aggregated report across multiple robustness analyses.

    Attributes
    ----------
    analyses : list[ConclusionRobustness]
        Individual robustness analysis results.
    overall_robustness : float
        Average stability score across all analyses.
    warnings : list[str]
        Any warnings generated during the analysis.
    """

    analyses: list[ConclusionRobustness] = field(default_factory=list)
    overall_robustness: float = 0.0
    warnings: list[str] = field(default_factory=list)
