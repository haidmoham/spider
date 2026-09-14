# Agent log

## AI-20260809-001 — Multi-DOF coupling

```yaml
id: AI-20260809-001
date: 2026-08-09
agent: Codex
status: resolved
evaluation: confirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-two-link-coupling/README.md
    - experiments/2026-08-09-two-link-coupling/two_link_coupling.py
    - experiments/2026-08-09-two-link-coupling/agent-log.md
related:
  experiment: experiments/2026-08-09-two-link-coupling
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [2]
  files:
    - experiments/2026-08-09-two-link-coupling/README.md
    - experiments/2026-08-09-two-link-coupling/two_link_coupling.py
objects:
  question: Q-20260809-001
  response: R-20260809-001
  evaluation: E-20260809-001
  action: A-20260809-001
  outcome: O-20260809-001
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How does commanding joint 1 change joint 2's behavior in a planar two-link arm, and what does that reveal about multi-joint dynamics?

### Human prediction

For the passive-elbow run, expect a janky, loosely waving arm: shoulder-driven motion with the elbow trailing or moving in a way that was not explicitly commanded.

### Refined prediction

The elbow motor never drives in passive mode, so the observed flopping should be understood as the behavior of an unpowered joint rather than a motor that is too weak to overcome gravity.

### Next prediction

With joint 2 switched to `hold` while keeping elbow-down configuration and normal gravity, expect a more distinctive, intentional-looking motion than the passive flopping.

### Gravity-off prediction

With gravity disabled, expect the staged motion to look cleaner because the motors do not need to oppose the gravitational load.

### Purpose

Move beyond one-joint control intuition with the smallest experiment that can expose state-dependent multi-joint dynamics.

### Context supplied

- GitHub issue #2.
- The prior one-link pendulum experiment.
- Direct MuJoCo access to `qpos`, `qvel`, `ctrl`, and `qacc`.

## R — Response summary

A two-link MuJoCo arm was used to compare a passive elbow, a held elbow, and gravity-on/off behavior while keeping the controller deliberately simple.

## E — Human evaluation

### Accepted

- In the passive, elbow-down run with normal gravity, joint 2 visibly moved while only joint 1 was commanded.
- The motion was qualitatively close to the predicted janky, loosely waving arm.
- With the same configuration and gravity but `joint2_mode=hold`, the motion became more organized and wave-like.
- With gravity disabled, the staged motion looked cleaner and more recognizable, matching the gravity-off prediction.
- The experiment made the important distinction between physical coupling and controller-imposed coupling visible. Explicit target cross-feed can create a coordinated gesture, but it is not evidence for the passive physical coupling that motivated issue #2.
- Further parameter search for a more exact wave was judged to have low learning value once the coupling behavior was understood.

### Rejected

- “The elbow motor is too weak to overcome gravity” was rejected. In passive mode, the elbow motor receives `ctrl[1] = 0.0` and never drives.

### Deferred

- A precise decomposition of which coupling effects dominate across pose, gravity, and velocity can be revisited when a later experiment needs it.
- The elbow-up comparison remains available as a follow-up, but it is not required for the current learning target.

### Verification required

None for the current closure. Deferred comparisons should make a fresh prediction if they are revisited.

## A — Action

Built the minimal two-link experiment, compared passive and held joint-2 behavior, and tested a gravity-off case. Added a coordinated wave mode for exploration, then stopped waveform tuning when it no longer tested the multi-DOF mental model.

## O — Outcome

Issue #2 is resolved for now. The experiment produced direct evidence that one joint's behavior cannot be reasoned about as an isolated pendulum once it is part of a moving two-link system. The passive elbow moved with zero elbow command, changing joint-2 control changed the motion, and removing gravity changed the resulting behavior.

The recognizable wave was useful as an engaging visualization, but reproducing a particular waveform became a controller-tuning search rather than the learning target.

### Effect on current belief

The independent-joint mental model is insufficient for this system. Multi-joint behavior depends on the state and motion of the full mechanism, not only the local command at each joint. The next useful work does not need more tuning of this experiment.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-two-link-coupling/agent-log.md
  commit:
objects:
  - id: Q-20260809-001
    type: Question
    summary: How does driving one joint affect the other joint in a two-link arm?
    status: resolved
  - id: R-20260809-001
    type: Response
    summary: Used a minimal two-link comparison with passive/held elbow behavior and gravity as discriminating conditions.
    status: confirmed
  - id: E-20260809-001
    type: Evaluation
    summary: Passive joint 2 moved with zero elbow command; holding it changed the motion; gravity-off behavior changed as predicted; waveform tuning stopped when it ceased to test the coupling model.
    status: confirmed
  - id: A-20260809-001
    type: Action
    summary: Built and ran the two-link coupling experiment, compared meaningful control conditions, and stopped presentation-oriented tuning after the learning target was met.
    status: completed
  - id: O-20260809-001
    type: Outcome
    summary: Independent-joint intuition was replaced by a full-state multi-joint mental model; issue #2 is resolved for now.
    status: resolved
relations:
  - subject: Q-20260809-001
    predicate: receives
    object: R-20260809-001
  - subject: R-20260809-001
    predicate: receives
    object: E-20260809-001
  - subject: E-20260809-001
    predicate: causes
    object: A-20260809-001
  - subject: A-20260809-001
    predicate: produces
    object: O-20260809-001
unresolved_questions:
  - Deferred: which coupling terms dominate across pose, gravity, and velocity when a later task requires that decomposition?
superseded_claims:
  - The initial motor-strength explanation was superseded by the code-level observation that passive joint 2 is unpowered.
```

Related: #2

## AI-20260809-003 — Wave coupling audit

```yaml
id: AI-20260809-003
date: 2026-08-09
agent: Codex
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-two-link-coupling/two_link_coupling.py
    - experiments/2026-08-09-two-link-coupling/agent-log.md
related:
  experiment: experiments/2026-08-09-two-link-coupling
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [2]
  files:
    - experiments/2026-08-09-two-link-coupling/two_link_coupling.py
objects:
  question: Q-20260809-003
  response: R-20260809-003
  evaluation: E-20260809-003
  action: A-20260809-003
  outcome: O-20260809-003
librarian:
  status: pending
  record_ids: []
```

## Q — Question

Does `--control-coupling` change the joint-2 wave target at the configured phase?

### Human prediction

Not recorded. This was a code-level audit, not a physics interpretation.

### Purpose

Ensure the documented coupling comparison changes its stated input.

### Context supplied

- `experiments/2026-08-09-two-link-coupling/two_link_coupling.py`
- The README claim that `--control-coupling 0` removes cross-feed.

## R — Response summary

At `WAVE_PHASE = pi`, the former base-wave amplitude term cancelled the cross-feed. The target was therefore independent of `control_coupling`.

## E — Human evaluation

### Accepted

- Pending human review.

### Rejected

- None.

### Unresolved

- None for the resolved Issue #2 learning target.

### Verification required

- Direct target calculation after the correction.

## A — Action

Removed the cancelling amplitude term. Joint 2 now uses a fixed phase-shifted wave plus the documented cross-feed term.

## O — Outcome

The direct target calculation changes with `control_coupling`. This corrects the optional wave mode without reopening the resolved physical-coupling question.

### Effect on current belief

No change to Issue #2's recorded learning outcome.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-two-link-coupling/agent-log.md
  commit:
objects:
  - id: Q-20260809-003
    type: Question
    summary: Does the documented control-coupling flag change the joint-2 target?
    status: resolved
  - id: R-20260809-003
    type: Response
    summary: The prior phase and amplitude terms cancelled the coupling effect.
    status: verified-by-inspection
  - id: E-20260809-003
    type: Evaluation
    summary: Awaiting human review; the resolved Issue #2 conclusion is unchanged.
    status: unconfirmed
  - id: A-20260809-003
    type: Action
    summary: Removed the cancelling base-wave amplitude term.
    status: completed
  - id: O-20260809-003
    type: Outcome
    summary: The target now varies with the coupling value.
    status: observed
relations:
  - subject: Q-20260809-003
    predicate: receives
    object: R-20260809-003
  - subject: R-20260809-003
    predicate: receives
    object: E-20260809-003
  - subject: E-20260809-003
    predicate: causes
    object: A-20260809-003
  - subject: A-20260809-003
    predicate: produces
    object: O-20260809-003
unresolved_questions: []
superseded_claims:
  - The prior implementation exposed tunable cross-feed, but its wave-target algebra cancelled it at the configured phase.
```

## AI-20260809-011 — Task-space Jacobian placement in standing control

```yaml
id: AI-20260809-011
date: 2026-08-09
sources:
  - kind: chat
    system: ChatGPT
    reference: conversation-local: task-space-jacobians-standing
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-two-link-coupling/agent-log.md
related:
  experiment: experiments/2026-08-09-two-link-coupling
  hypotheses: []
  experiments: []
  claims: []
  decisions:
    - Preserve the Jacobian as a local task-space motion map, not as the whole feedback controller.
  issues: [2, 6]
  files:
    - TODO.md
objects:
  question: Q-20260809-011
  response: R-20260809-011
  evaluation: E-20260809-011
  action: A-20260809-011
  outcome: O-20260809-011
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How do task space and Jacobians relate to stable standing in C-1N, and where does the Jacobian sit in the control stack?

### Human prediction

The Jacobian may provide the information needed for the robot to continuously re-adjust into stable standing by translating between the current joint configuration and the corrective motion required by the task.

### Purpose

Connect the multi-joint coupling result to the next task-space experiment without collapsing the Jacobian into a complete standing controller.

### Context supplied

- Issue #2 established that synchronized or isolated joint reasoning is insufficient for a coupled mechanism.
- `TODO.md` selects issue #6, Jacobians and task space, as the next experiment.
- C-1N is the motivating integration target after the isolated bench experiment closes.

## R — Response summary

Task space is an output representation selected for the behavior of interest, such as foot position or body height. A task map `x = f(q)` maps configuration space into that representation. The Jacobian `J(q) = df/dq` is the pose-dependent local linear map that relates joint-space motion to task-space motion through `xdot = J(q) qdot`.

For standing, the full loop is broader: estimate state, measure task error, choose a correction, map that correction through the current mechanism, command joints, then observe the new state. The Jacobian occupies the local motion-mapping part of that loop. It can support conversion between desired body or foot corrections and joint corrections, but it does not define stability, estimate state, model contact, or close the controller by itself.

The door-hinge example repaired the local-map intuition: as the hinge angle changes, the same angular motion maps to a task-space velocity whose direction changes because `J(q)` changes with configuration.

## E — Human evaluation

### Accepted

- Task space is not a subspace or normalization of joint space. It is a task-selected output representation of robot configuration.
- The Jacobian is well understood as a localized, pose-dependent generalization of a gradient for a vector-valued task map.
- `J(q)` changes as the robot traverses configuration space.
- In `xdot = J(q) qdot`, a fixed joint velocity can produce changing task-space velocity because the local map changes with pose.
- The Jacobian belongs in the translation layer of a feedback stack; it is one tool in that layer rather than the full loop.

### Rejected

- Treating the Jacobian itself as the complete feedback loop.
- Treating task space as a collection, normalization, or linear-algebra subspace of joint spaces.

### Unresolved

- Which task variables are sufficient for C-1N standing.
- Which additional models are required once contact, center of mass, forces, and stability margins matter.

### Verification required

- Issue #6 must verify the local joint-to-task mapping experimentally before this conversation-derived model is treated as bench evidence.

## A — Action

Preserve this belief update as the conceptual bridge from issue #2 to issue #6. Do not change the issue #6 experiment design or skip its Iteration 0 prediction. Use the bench experiment to test the pose-dependent mapping before applying task-space telemetry to C-1N.

## O — Outcome

The conceptual stack is now separated cleanly: feedback supplies the repeated correction loop, while the Jacobian supplies a local geometry-aware map inside that loop.

### Effect on current belief

- Before: The relationship among joint space, task space, Jacobians, and standing was unclear; the Jacobian was tentatively treated as the information feedback loop required for continuous stabilization.
- After: Task space is a task-specific output representation `x = f(q)`. The Jacobian is the pose-dependent local derivative of that map and sits inside a larger feedback architecture as a joint-motion-to-task-motion translation layer.
- Evidence status: Conversation-derived human understanding supported by mathematical explanation. Not yet experimentally verified in issue #6.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-two-link-coupling/agent-log.md
  commit:
provenance:
  - kind: chat
    system: ChatGPT
    reference: conversation-local: task-space-jacobians-standing
objects:
  - id: Q-20260809-011
    type: Question
    summary: Where does the Jacobian sit between joint-space coupling and stable task-space standing?
    status: resolved-conceptually
  - id: R-20260809-011
    type: Response
    summary: The Jacobian is the pose-dependent local map from joint-space motion to task-space motion inside a broader feedback loop.
    status: explanation
  - id: E-20260809-011
    type: Evaluation
    summary: The human accepted the local-map model and rejected both the full-feedback-loop and task-space-as-subspace models.
    status: accepted-unverified
  - id: A-20260809-011
    type: Action
    summary: Preserve the belief update as the bridge into issue #6 without replacing its required prediction or experiment.
    status: completed
  - id: O-20260809-011
    type: Outcome
    summary: Joint space, task space, Jacobian, and feedback now occupy distinct logical layers; issue #6 remains the required verification step.
    status: conceptually-resolved
relations:
  - subject: Q-20260809-011
    predicate: receives
    object: R-20260809-011
  - subject: R-20260809-011
    predicate: receives
    object: E-20260809-011
  - subject: E-20260809-011
    predicate: causes
    object: A-20260809-011
  - subject: A-20260809-011
    predicate: produces
    object: O-20260809-011
unresolved_questions:
  - Which task variables are sufficient for stable C-1N standing?
  - Which contact, force, center-of-mass, or stability models become necessary after the kinematic mapping is verified?
superseded_claims:
  - The Jacobian itself is the feedback loop required for stable standing.
  - Task space is a normalized or linear-algebra subspace of joint space.
```
