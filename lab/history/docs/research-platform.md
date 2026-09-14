# Simulation research platform direction

This document owns the long-range design for the post-foundation simulation platform. It does not select current work. `TODO.md` remains the only current-work selector.

## Activation gate

Do not build this platform before the foundation produces a concrete need.

The minimum entry state is:

1. C-1N has earned `STAND` with reproducible support-aware evidence.
2. A learned locomotion loop can produce rollout populations.
3. A real failure requires more scale, reproducibility, validation, or structured analysis.

Until then, use this document only as a design constraint.

## System boundary

The target flow is:

```text
declared world distribution or grammar
-> typed world specification
-> deterministic validators
-> versioned world manifest
-> reproducible rollout population
-> retained metrics, failures, and artifacts
-> comparison and counterexample analysis
```

Deterministic simulator contracts remain authoritative.

## Required capabilities

### World generation

- Generate structured pseudorandom physical worlds from declared distributions or grammars.
- Keep each generated world inspectable as a typed specification.
- Record the seed, generator version, simulator version, and parameter distribution.

### Validation

- Reject worlds that violate MuJoCo, physics, semantic, or experiment-specific constraints.
- Retain invalid worlds and validation failures.
- Do not silently repair a world after validation fails.

### Execution

- Run reproducible rollout populations.
- Keep treatment conditions, policy versions, simulator versions, and seeds explicit.
- Preserve failed rollouts as evidence.

### Analysis

- Compare compatible populations under declared conditions.
- Support treatment comparison, failure clustering, and counterexample reduction.
- Keep observations, inferences, and speculation separate.
- Record simulator limits and uncertainty with each conclusion.

## Agent control plane

A natural-language agent is a first-class control plane. It is not an unrestricted chatbot wrapper.

The agent must compile a goal into:

- an explicit hypothesis;
- typed interventions and controls;
- declared measurements and evaluation criteria;
- typed generation, validation, execution, and analysis operations.

The agent must not override deterministic validators or simulator contracts. It must not change hidden parameters to rescue a failed result.

## Scientific constraints

- Declare hypotheses, interventions, controls, measured variables, and evaluation criteria before execution when the question permits it.
- Keep training objectives separate from evaluation metrics.
- Create a new analysis record when a metric or protocol changes after results are visible.
- Record code, simulator, seed, distribution, policy, and metric provenance.
- Require compatible populations and declared conditions for comparisons.
- Retain invalid worlds, negative results, and failures.

## Routing rule

Let platform failures select the next engineering or learning block.

- World-validity failures can pull in mechanics, numerical methods, or simulator contracts.
- Weak population evidence can pull in statistics or experiment design.
- Calibration or transfer gaps can pull in system identification and uncertainty.
- Throughput or reproducibility limits can pull in simulation or distributed-systems architecture.

Do not build infrastructure or study a topic only because it appears in this design.

## Non-goals

- This platform is not a prerequisite for `STAND` or the first learned gait.
- It is not a reason to create generic frameworks before a concrete experiment needs them.
- It does not make hardware a graduation requirement.
- It does not permit an agent to replace scientific judgment with unsupported conclusions.
