# Agent Interaction: Ant policy-learning scaffold prediction

```yaml
id: AI-20260827-001
date: 2026-08-27
sources:
  - kind: chat
    system: Codex
    reference: current session
  - kind: coding-agent
    system: Codex
    reference: current session
status: open
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: codex/inherit-learning-contract
  commit:
  changed_files:
    - experiments/2026-08-27-ant-policy-learning/README.md
    - experiments/2026-08-27-ant-policy-learning/ant_policy_learning.ipynb
    - experiments/2026-08-27-ant-policy-learning/agent-log.md
related:
  experiment: experiments/2026-08-27-ant-policy-learning
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [25]
  files:
    - experiments/2026-08-27-ant-policy-learning/ant_policy_learning.ipynb
objects:
  question: Q-20260827-001
  response: R-20260827-001
  evaluation: E-20260827-001
  action: A-20260827-001
  outcome: O-20260827-001
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How could a velocity-focused objective be exploited without producing stable
forward walking?

### Human prediction

speed up to a high peak velocity without moving much

### Purpose

Record an Iteration 0 failure mode before choosing or running a policy
objective for the Ant learning fixture.

## R — Response summary

The notebook scaffold separates control, misspecified, and corrected treatment
slots. The misspecified reward form remains a user-authored TODO so its
behavior can be tested rather than assumed.

## E — Human evaluation

### Unresolved

The prediction needs a defined treatment and an executed evaluation rollout.

### Verification required

Record fixed-seed measurements of displacement, peak velocity, survival,
control/contact costs, and termination cause after the user implements the
objective and evaluation cells.

## A — Action

Created an incomplete Ant-v5 policy-learning notebook scaffold with explicit
TODO boundaries for the rollout, objective, PPO, training, and evaluation work.

## O — Outcome

Pending

### Effect on current belief

- Before: policy behavior could satisfy a poorly chosen objective without intended walking.
- After: the specific peak-velocity-with-low-displacement prediction is recorded for testing.
- Evidence status: Human prediction, not experimental evidence.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-27-ant-policy-learning/agent-log.md
  commit:
provenance:
  - kind: chat
    system: Codex
    reference: current session
  - kind: coding-agent
    system: Codex
    reference: current session
objects:
  - id: Q-20260827-001
    type: Question
    summary: Identify a reward-exploitation failure mode before policy training.
    status: open
  - id: R-20260827-001
    type: Response
    summary: Preserve separate control, misspecified, and corrected treatment slots.
    status: open
  - id: E-20260827-001
    type: Evaluation
    summary: Fixed-seed evaluation remains required.
    status: unconfirmed
  - id: A-20260827-001
    type: Action
    summary: Created the incomplete Ant-v5 learning scaffold.
    status: acted
  - id: O-20260827-001
    type: Outcome
    summary: Pending experimental outcome.
    status: pending
relations:
  - subject: Q-20260827-001
    predicate: receives
    object: R-20260827-001
  - subject: R-20260827-001
    predicate: receives
    object: E-20260827-001
  - subject: E-20260827-001
    predicate: causes
    object: A-20260827-001
  - subject: A-20260827-001
    predicate: produces
    object: O-20260827-001
unresolved_questions:
  - Which misspecified velocity objective should be implemented and tested?
superseded_claims: []
```
