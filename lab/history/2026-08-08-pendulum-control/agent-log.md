# Agent log

## AI-20260808-001 — P vs. PD prediction

```yaml
id: AI-20260808-001
date: 2026-08-08
agent: Codex
status: resolved
evaluation: confirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-08-pendulum-control/pendulum.py
related:
  experiment: experiments/2026-08-08-pendulum-control
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [1]
  files:
    - experiments/2026-08-08-pendulum-control/README.md
    - experiments/2026-08-08-pendulum-control/pendulum.py
objects:
  question: Q-20260808-001
  response: R-20260808-001
  evaluation: E-20260808-001
  action: A-20260808-001
  outcome: O-20260808-001
librarian:
  status: pending
  record_ids: []
```

## Q — Question

What should look different when the pendulum uses position-only feedback versus position + velocity feedback?

### Human prediction

P-only should overshoot or oscillate more. PD should damp motion and settle more cleanly.

### Purpose

Build intuition for feedback control before moving to a more complex mechanism.

### Context supplied

- One-link MuJoCo pendulum.
- One hinge motor.
- Direct `qpos`, `qvel`, and `ctrl` access.
- P and PD modes in the same script.

## R — Response summary

The implemented experiment exposes constant, P, and PD control directly. P uses position error. PD adds velocity feedback so damping can be observed by changing `Kp` and `Kd`.

## E — Human evaluation

### Accepted

- P-only overshot or oscillated more.
- PD damped the motion and settled more cleanly.

### Rejected

- None.

### Unresolved

- What obviously too-high `Kp` and too-high `Kd` look like.

### Verification required

- Verified by running the MuJoCo viewer and comparing P and PD behavior.

## A — Action

Ran the pendulum control experiment and compared P and PD behavior.

## O — Outcome

The Iteration 0 prediction matched the observed behavior.

### Effect on current belief

Position error creates corrective drive; velocity feedback adds damping. This intuition is strong enough to move on rather than over-study the MuJoCo API.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-08-pendulum-control/agent-log.md
  commit:
objects:
  - id: Q-20260808-001
    type: Question
    summary: How should P-only and PD feedback differ on a one-link pendulum?
    status: resolved
  - id: R-20260808-001
    type: Response
    summary: P uses position error; PD adds velocity feedback so damping can be observed directly.
    status: confirmed
  - id: E-20260808-001
    type: Evaluation
    summary: Human prediction matched observed P and PD behavior.
    status: confirmed
  - id: A-20260808-001
    type: Action
    summary: Ran P and PD control modes in the MuJoCo pendulum experiment.
    status: completed
  - id: O-20260808-001
    type: Outcome
    summary: P oscillated more and PD damped motion, matching the prediction.
    status: observed
relations:
  - subject: Q-20260808-001
    predicate: receives
    object: R-20260808-001
  - subject: R-20260808-001
    predicate: receives
    object: E-20260808-001
  - subject: E-20260808-001
    predicate: causes
    object: A-20260808-001
  - subject: A-20260808-001
    predicate: produces
    object: O-20260808-001
unresolved_questions:
  - What do obviously too-high Kp and too-high Kd look like?
superseded_claims: []
```

Related: #1
