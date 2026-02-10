# CLAUDE.md - Optimization Copilot Development Guide

## Project Overview
OptimizationAgent: An intelligent optimization decision layer that automatically selects,
adapts, and justifies optimization strategies based on experimental history.

## Architecture
See `plan.md` for the full specification. Key modules to implement:
- `optimization_copilot/core/` — CampaignSnapshot, StrategyDecision, ProblemFingerprint
- `optimization_copilot/diagnostics/` — 14-signal diagnostic engine
- `optimization_copilot/profiler/` — Problem profiler and fingerprinting
- `optimization_copilot/backends/` — Plugin-based optimization backend pool
- `optimization_copilot/meta_controller/` — Phase orchestration and switching logic
- `optimization_copilot/stabilization/` — Data conditioning and noise handling
- `optimization_copilot/screening/` — High-dimensional variable screening
- `optimization_copilot/feasibility/` — Failure learning and safe-region mapping
- `optimization_copilot/multi_objective/` — Pareto front tracking and dominance
- `optimization_copilot/explainability/` — Decision explanation layer
- `optimization_copilot/plugins/` — Plugin architecture and contracts
- `optimization_copilot/validation/` — Golden scenarios and regression tests

## Tech Stack
- Python 3.10+
- Claude Opus 4.6 via Anthropic API for agent intelligence
- No heavy ML frameworks unless strictly needed — prefer lightweight implementations
- Plugin architecture for extensibility

## Development Rules
- All code must be deterministic and reproducible (given same seed)
- Every decision must produce full audit trail
- Use type hints throughout
- Write tests alongside implementation
- Follow existing project structure and patterns
- Keep implementations minimal — YAGNI principle
- Each module should be independently testable

## Model
- Default: claude-opus-4-6
- API key in environment variable ANTHROPIC_API_KEY

## Auto-Development Mode
- All PRs from Claude are auto-approved for merge
- Claude has full write access to implement features
- Follow plan.md section ordering for implementation priority
