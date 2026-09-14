# Agent Interaction: static-support boundary prediction

```yaml
id: AI-20260813-001
date: 2026-08-13
sources:
  - kind: human-observation
    system: Codex
    reference: issue #24 / current session
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
    - experiments/2026-08-13-static-support-boundary/static_support_boundary.py
    - experiments/2026-08-13-static-support-boundary/static_support_boundary.ipynb
    - experiments/2026-08-13-static-support-boundary/README.md
    - experiments/2026-08-13-static-support-boundary/agent-log.md
related:
  experiment: experiments/2026-08-13-static-support-boundary
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [24]
  files:
    - experiments/2026-08-13-static-support-boundary/static_support_boundary.py
objects:
  question: Q-20260813-001
  response: R-20260813-001
  evaluation: E-20260813-001
  action: A-20260813-001
  outcome: O-20260813-001
librarian:
  status: pending
  record_ids: []
```

## Q — Question

Does a fixed three-foot support margin become zero before the rear contact
unloads as the body mass moves forward?

### Human prediction

The support margin reaches zero earlier. The current morphology may be less
stable during forward motion because a gait reduces the available contacts.

### Clarification after the rollout

The human clarified that the typed claim that rear load would increase was a
misread of the prompt. Their intended model was that forward ballast increases
front load and decreases rear load.

### Purpose

Separate the static support relation from the later moving-gait failure.

## R — Response summary

Use a fixed-foot MuJoCo model. Sweep only payload position. Measure support
margin, normal loads, and torso attitude.

## E — Human evaluation

### Unresolved

The ordering of zero support margin and rear-contact unloading needs a run.

### Verification required

Repeat the model in a C-1N stance-only pose after the leg-workspace gate.

## A — Action

Created and ran a minimal fixed-foot MuJoCo support probe. Added an executed
Jupyter sweep that records support margin and observed normal contact loads.

## O — Outcome

At a `+0.96 m` payload shift, the margin was `+0.0010 m` and each rear foot
carried `0.046 N`. At `+0.98 m`, both rear contacts were unloaded and the
final margin was `-0.0965 m` after the body tipped. The transition is bracketed
between those two samples.

### Effect on current belief

- Before: COM projection was the broad suspected standing failure.
- After: The static support boundary is consistent with limiting rear contact
  unloading. Dynamic instability from a reduced gait contact set remains a
  separate C-1N question.
- Evidence status: Executed headless fixture and notebook results. This is not
  C-1N morphology or gait evidence.
