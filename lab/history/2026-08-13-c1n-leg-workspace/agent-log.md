# Agent Interaction: C-1N one-leg workspace prediction

```yaml
id: AI-20260813-002
date: 2026-08-13
sources:
  - kind: human-observation
    system: Codex
    reference: issue #31 / current session
  - kind: coding-agent
    system: Codex
    reference: current session
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch:
  commit:
  changed_files:
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.ipynb
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.py
    - experiments/2026-08-13-c1n-leg-workspace/README.md
    - experiments/2026-08-13-c1n-leg-workspace/agent-log.md
related:
  experiment: experiments/2026-08-13-c1n-leg-workspace
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [31]
  files:
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.ipynb
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.py
objects:
  question: Q-20260813-002
  response: R-20260813-002
  evaluation: E-20260813-002
  action: A-20260813-002
  outcome: O-20260813-002
librarian:
  status: pending
  record_ids: []
```

## Q — Question

Can one current C-1N front-left leg reach the outward-and-down body-frame
target `[+0.30, +0.32, -0.30] m` with its existing joint axes and limits?

### Human prediction

The target should be reachable. The joint axes and orientations have not yet
been inspected closely.

### Purpose

Separate a kinematic morphology limit from gravity compensation or control.

## R — Response summary

Fix the torso, disable gravity, and sample only the two existing front-left
joints. Compare sampled body-frame foot positions with one support target.

## E — Human evaluation

### Verification required

Execute the notebook and inspect top and side workspace projections.

## A — Action

Created a Jupyter workspace sweep against C-1N's current XML model.

## O — Outcome

The current two-DOF leg missed the target by `0.067 m`, chiefly `0.053 m` in
the outward body-Y direction. With the experimental proximal axis, a positive
5-degree probe changed the foot by `[-0.0217, +0.0264, +0.0012] m` and reduced
target distance from `0.0670 m` to `0.0328 m`.

The constrained inverse-kinematics section is scaffolded but intentionally
unrun. The user will supply predictions, execute it, and interpret its result
before it becomes experiment evidence.

### Effect on current belief

- Before: The support target was assumed plausible from leg length.
- After: The workspace sweep motivates a constrained inverse-kinematics check;
  its result must be read with the user before it changes the model.
- Evidence status: The grid experiment is executed. The constrained optimizer
  is an unrun scaffold. The canonical C-1N model remains unchanged.

### Follow-up decision

The evidence supports a minimal morphology change, but promotion to C-1N is a
user decision. Do not change the canonical model until the user accepts the
specific design direction and scope.

## AI-20260813-003 — Situated apprenticeship routing

```yaml
id: AI-20260813-003
date: 2026-08-13
sources:
  - kind: human-observation
    system: Codex
    reference: current session
status: acted
evaluation: accepted
repo_state:
  repository: robotics-test-bench
  branch:
  commit:
  changed_files:
    - integrations/c1n.md
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.ipynb
    - experiments/2026-08-13-c1n-leg-workspace/agent-log.md
related:
  experiment: experiments/2026-08-13-c1n-leg-workspace
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [12, 24, 31]
  files:
    - integrations/c1n.md
    - experiments/2026-08-13-c1n-leg-workspace/c1n_leg_workspace.ipynb
objects:
  question: Q-20260813-003
  response: R-20260813-003
  evaluation: E-20260813-003
  action: A-20260813-003
  outcome: O-20260813-003
librarian:
  status: not-needed
  record_ids: []
```

## Q — Question

How should the standing objective guide the interaction without replacing the
user's own physical reasoning or forcing an artificial pace?

## R — Response summary

Treat the goal as a routing and quality guardrail. Use Jupyter as a shared
reasoning surface. Explain observables before asking the user to make their own
prediction. Do not supply a predicted outcome for them to accept or falsify.

## E — Human evaluation

### Accepted

The user defined the work as a situated engineering apprenticeship. A robot
capability without end-to-end understanding would fail the objective.

## A — Action

Added the working agreement to the C-1N integration contract. Added a neutral
hip-geometry preview and user-owned prediction prompt to the #31 notebook.

## O — Outcome

The next experimental-DOF prediction remains intentionally open. The notebook
now exposes the relevant geometry without stating an expected result.

### Effect on current belief

- Before: The active goal was treated too aggressively as a sequence of gates.
- After: It is a guardrail for evidence, learning quality, and scope. Progress
  includes the user's ability to explain the causal chain.
- Evidence status: Human-directed working agreement and notebook structure;
  no new kinematic result.
