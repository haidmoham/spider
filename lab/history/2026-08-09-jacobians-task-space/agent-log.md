# Agent log

## AI-20260809-011 — Shared-frame Jacobian prediction

```yaml
id: AI-20260809-011
date: 2026-08-09
sources:
  - kind: chat
    system: Codex
    reference: current task
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/jacobians_task_space.py
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
    - experiments/telemetry.py
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6]
  files:
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/jacobians_task_space.py
    - experiments/telemetry.py
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

How does the same small positive first-joint perturbation map into shared-frame foot X motion as the base orientation changes?

### Human prediction

Holding geometry, joint pose, perturbation size, controller, and timestep fixed, the same small positive first-joint perturbation should produce a large shared-X foot displacement at 0 degrees, near-zero X displacement at 90 degrees, and an equally large displacement with the opposite X sign at 180 degrees. The foot should keep moving at 90 degrees, but along shared Z. At 45 and 135 degrees, the shared-X magnitude should be about 71% of the 0-degree magnitude, with opposite signs.

### Purpose

Make the joint-space-to-task-space mapping observable before adding foot telemetry to C-1N.

### Context supplied

- `TODO.md` issue #6 selector.
- C-1N's shared-frame joint-space versus foot-motion mismatch.
- A one-link planar geometry chosen to isolate frame orientation.

## R — Response summary

Use a fixed shared frame and rotate only the base. Compare joint telemetry with foot X telemetry, and validate the X Jacobian entry with a centered finite difference.

## E — Human evaluation

### Accepted

- A local foot-motion vector can stay nonzero while its shared-X component vanishes.
- At the initial straight-down pose, base orientation changes the X projection of the same local joint perturbation.

### Verification required

- Run 0, 90, and 180 degree cases and compare the Jacobian with the finite-difference estimate.
- Check whether observed initial X motion supports the predicted sign reversal and near-zero 90-degree projection.

## A — Action

Scaffolded a one-link MuJoCo experiment that varies only base pitch, reports joint and foot X position/velocity/acceleration, prints MuJoCo-versus-finite-difference Jacobian telemetry, and adds a three-orientation viewer overlay with C-1N's rolling graph-stack pattern for direct visual comparison.

## O — Outcome

Pending experiment run.

### Effect on current belief

- Before: synchronized joint commands were not yet connected to shared-frame foot motion.
- After: the working prediction is that base orientation rotates the local foot-motion vector and changes only its shared-frame projection.
- Evidence status: Human-accepted conceptual prediction; simulation observation is pending.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-jacobians-task-space/agent-log.md
  commit:
provenance:
  - kind: chat
    system: Codex
    reference: current task
objects:
  - id: Q-20260809-011
    type: Question
    summary: How does one joint perturbation map into shared-frame foot X motion across base orientations?
    status: open
  - id: R-20260809-011
    type: Response
    summary: Use a one-link planar model, fixed shared frame, and finite-difference Jacobian check.
    status: acted
  - id: E-20260809-011
    type: Evaluation
    summary: The human accepted the projection-based prediction; simulation verification remains pending.
    status: unconfirmed
  - id: A-20260809-011
    type: Action
    summary: Scaffolded base-orientation comparisons with joint and task-space telemetry.
    status: completed
  - id: O-20260809-011
    type: Outcome
    summary: Pending runs at 0, 90, and 180 degrees.
    status: pending
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
  - Do the observed initial X Jacobian signs and magnitudes match the prediction across orientations?
superseded_claims: []
```

Related: #6

## AI-20260809-013 — Static tripod support prediction

```yaml
id: AI-20260809-013
date: 2026-08-09
sources:
  - kind: chat
    system: Codex
    reference: current task
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: codex/responsive-jacobian-overlay
  commit:
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/tripod_static_support.py
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6]
  files:
    - experiments/2026-08-09-jacobians-task-space/tripod_static_support.py
objects:
  question: Q-20260809-013
  response: R-20260809-013
  evaluation: E-20260809-013
  action: A-20260809-013
  outcome: O-20260809-013
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How can three planted two-joint legs hold a free body under gravity while the body load shifts forward?

### Human prediction

Each leg needs a joint-space posture controller to maintain a usable bent configuration. Gravity and the body/COM position determine three ground-support forces; shifting forward increases the front force and decreases the rear forces. Each leg maps its assigned foot force into joint torques with its Jacobian transpose. If the support point reaches the triangle edge, lowering the body alone does not restore static balance: fixed feet must move the body projection back, while a walking robot can step.

### Purpose

Apply #6's pose-dependent foot Jacobian to the smallest physically grounded standing problem without adding gait logic.

## R — Response summary

Separate posture regulation from support-force allocation. Use the whole-robot free-body balance to choose foot forces, then use each leg's `J(q).T @ f` to map its assigned force into motor torques.

## E — Human evaluation

### Accepted

- Posture torque is derived from joint configuration error, not from the foot-force allocation.
- Ground reactions are external whole-robot forces; actuator torques appear only when isolating a leg.

### Verification required

- Centered run holds body height with all three positive support forces.
- A forward load-shift run increases the front vertical allocation and decreases both rear allocations.

## A — Action

Added a minimal MuJoCo tripod with a free body, three planted two-joint legs, a posture PD term, and Jacobian-transpose support torques. Added the local free-body diagram asset as the visual reference for the experiment and future portfolio post.

## O — Outcome

Pending the centered hold and one forward-load perturbation.

### Effect on current belief

- Before: the Jacobian was understood as a local motion map but not yet connected to static foot support.
- After: standing is modeled as posture regulation plus a force allocation whose per-leg forces become motor torques through `J(q).T @ f`.
- Evidence status: Human prediction accepted; physical simulation observation is pending.

Related: #6

## AI-20260809-012 — Two-joint velocity-map continuation

```yaml
id: AI-20260809-012
date: 2026-08-09
sources:
  - kind: chat
    system: Codex
    reference: current task
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: codex/responsive-jacobian-overlay
  commit:
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/two_joint_planar_leg.py
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6]
  files:
    - experiments/2026-08-09-jacobians-task-space/two_joint_planar_leg.py
objects:
  question: Q-20260809-012
  response: R-20260809-012
  evaluation: E-20260809-012
  action: A-20260809-012
  outcome: O-20260809-012
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How does the same small two-joint velocity map into foot velocity when only the planar leg pose changes?

### Human prediction

Not yet recorded. Predict the relative world-X and world-Z foot motion before running the two poses.

### Purpose

Continue #6 from one Jacobian column and a rotated frame to the complete 2-by-2 planar velocity map, without introducing locomotion.

## R — Response summary

Set the same `qdot` at two fixed poses, compare `J(q) @ qdot` to MuJoCo's foot velocity, then attribute any velocity change to the changed local map rather than to a changed command.

## E — Human evaluation

### Verification required

- Confirm that predicted and MuJoCo-reported X-Z foot velocities agree at both poses.
- Record the learner's prediction and observed pose-dependent change before explaining it further.

## A — Action

Added the two-joint planar-leg velocity comparison as a continuation inside the active #6 directory, with an optional viewer that alternates the two evaluated poses.

## O — Outcome

Pending the learner's prediction and observation.

### Effect on current belief

- Before: the one-link comparison connected a single joint column to task-space motion across frames.
- After: the next test isolates the full local two-joint velocity mapping while holding `qdot` fixed.
- Evidence status: Experiment scaffold only; no learner interpretation is recorded.

Related: #6

## AI-20260809-014 — Equal-foot-force failure is evidence, not a #6 fix target

```yaml
id: AI-20260809-014
date: 2026-08-09
sources:
  - kind: human-observation
    system: Codex
    reference: current task
  - kind: coding-agent
    system: Codex
    reference: headless tripod runs
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: codex/responsive-jacobian-overlay
  commit: 60af8d5
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/tripod_static_support.py
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6, 20]
  files:
    - experiments/2026-08-09-jacobians-task-space/tripod_static_support.py
objects:
  question: Q-20260809-014
  response: R-20260809-014
  evaluation: E-20260809-014
  action: A-20260809-014
  outcome: O-20260809-014
librarian:
  status: not-needed
  record_ids: []
```

## Q — Question

Should the equal-foot-force tripod experiment become a standing controller, or should an impossible negative rear normal-force request go to #20 if observed?

## R — Response summary

Keep #6 scoped to the joint-motion → foot-motion → Jacobian → torque chain. Defer feasible allocation, contact feedback, friction, and planted-foot control to #20.

## E — Human evaluation

### Accepted

- The equal-force failure is useful evidence, not a defect to conceal.
- #6 must not solve full standing control.

## A — Action

Marked the code and README with the #6 scope boundary. Left the equal-foot-force behavior unchanged.

## O — Outcome

The checked 2 s and 8 s forward headless runs kept rear vertical allocations positive. A negative normal-force request is an unverified #20 diagnostic, not evidence from the current run.

### Effect on current belief

- Before: the tripod continuation risked expanding into a standing controller.
- After: it is a bounded inspection of Jacobian-transpose torque mapping. Contact-feasibility control is deferred to #20.
- Evidence status: The force-mapping behavior was run locally. The negative-normal condition is not reproduced. The scope decision is human-directed.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-jacobians-task-space/agent-log.md
  commit: 60af8d5
provenance:
  - kind: human-observation
    system: Codex
    reference: current task
  - kind: coding-agent
    system: Codex
    reference: headless tripod runs
objects:
  - id: Q-20260809-014
    type: Question
    summary: Keep #6 bounded when force feasibility becomes relevant.
    status: resolved
  - id: R-20260809-014
    type: Response
    summary: Defer standing-control mechanisms to #20.
    status: acted
  - id: E-20260809-014
    type: Evaluation
    summary: The human accepted preserving the equal-force failure as evidence.
    status: confirmed
  - id: A-20260809-014
    type: Action
    summary: Recorded the scope boundary without adding feasibility control.
    status: completed
  - id: O-20260809-014
    type: Outcome
    summary: Negative rear normal force is not reproduced in the checked runs.
    status: unconfirmed
relations:
  - subject: Q-20260809-014
    predicate: receives
    object: R-20260809-014
  - subject: R-20260809-014
    predicate: receives
    object: E-20260809-014
  - subject: E-20260809-014
    predicate: causes
    object: A-20260809-014
  - subject: A-20260809-014
    predicate: produces
    object: O-20260809-014
unresolved_questions:
  - Under what body pose does the unconstrained allocation request a negative normal force?
superseded_claims:
  - The prior claim that the checked forward run had negative rear normal force.
```

Related: #6, #20

## AI-20260810-016 — Intentional stop before the combined orientation probe

```yaml
id: AI-20260810-016
date: 2026-08-10
sources:
  - kind: human-observation
    system: Codex
    reference: current task
status: decided
evaluation: accepted
repo_state:
  repository: robotics-test-bench
  branch: codex/responsive-jacobian-overlay
  commit:
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6]
  files:
    - experiments/2026-08-09-jacobians-task-space/README.md
objects:
  question: Q-20260810-016
  response: R-20260810-016
  evaluation: E-20260810-016
  action: A-20260810-016
  outcome: O-20260810-016
librarian:
  status: not-needed
  record_ids: []
```

## Q — Question

Must the remaining combined two-DOF, three-orientation static probe run before the existing #6 evidence can be projected publicly?

## R — Response summary

The remaining probe is a useful coverage check, but it does not add a new learning target after the learner has independently verified the orientation finite difference, the full two-joint velocity map, and the pose-dependent force-to-torque bridge.

## E — Human evaluation

The human intentionally skipped the remaining combined probe after demonstrating the required physical model in a cold explanation. This is a stop decision, not a claim that the issue's implementation checklist is complete.

## A — Action

Preserve the evidence gap in the experiment README. Project only the verified observations and explicit stop boundary into the portfolio Working Notes page.

## O — Outcome

The experiment has a public-facing stop boundary while GitHub issue #6 remains open. The skipped probe remains available if a later task needs the combined local-versus-shared-frame static check.

### Effect on current belief

- Before: completion was tied to one unrun combined probe.
- After: the learner has enough verified evidence to state the local-map model; the unrun probe is documented as an intentional scope stop rather than erased.
- Evidence status: Human decision accepted. The skipped probe is unverified by design.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-jacobians-task-space/agent-log.md
  commit:
provenance:
  - kind: human-observation
    system: Codex
    reference: current task
objects:
  - id: Q-20260810-016
    type: Question
    summary: Decide whether to run the remaining combined probe before public projection.
    status: resolved
  - id: R-20260810-016
    type: Response
    summary: Treat the remaining probe as an explicit stop boundary, not hidden missing evidence.
    status: acted
  - id: E-20260810-016
    type: Evaluation
    summary: The human accepted the intentional skip after a successful cold explanation.
    status: confirmed
  - id: A-20260810-016
    type: Action
    summary: Keep the gap visible and project only verified evidence.
    status: completed
  - id: O-20260810-016
    type: Outcome
    summary: #6 remains open; its documented evidence is eligible for a bounded public note.
    status: confirmed
relations:
  - subject: Q-20260810-016
    predicate: receives
    object: R-20260810-016
  - subject: R-20260810-016
    predicate: receives
    object: E-20260810-016
  - subject: E-20260810-016
    predicate: causes
    object: A-20260810-016
  - subject: A-20260810-016
    predicate: produces
    object: O-20260810-016
unresolved_questions:
  - When needed, what does the combined two-DOF three-orientation static probe add beyond the recorded separate checks?
superseded_claims: []
```

Related: #6

## AI-20260810-015 — #6 closure and portfolio visual requirements

```yaml
id: AI-20260810-015
date: 2026-08-10
sources:
  - kind: human-observation
    system: Codex
    reference: current task
  - kind: coding-agent
    system: Codex
    reference: headless Jacobian verification
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: codex/responsive-jacobian-overlay
  commit:
  changed_files:
    - experiments/2026-08-09-jacobians-task-space/README.md
    - experiments/2026-08-09-jacobians-task-space/agent-log.md
related:
  experiment: experiments/2026-08-09-jacobians-task-space
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [6, 20]
  files:
    - experiments/2026-08-09-jacobians-task-space/jacobians_task_space.py
    - experiments/2026-08-09-jacobians-task-space/two_joint_planar_leg.py
    - experiments/2026-08-09-jacobians-task-space/assets/tripod-support-fbd.png
objects:
  question: Q-20260810-015
  response: R-20260810-015
  evaluation: E-20260810-015
  action: A-20260810-015
  outcome: O-20260810-015
librarian:
  status: not-needed
  record_ids: []
```

## Q — Question

Has #6 established the pose-dependent local motion map and its limited force-to-torque use well enough to close without solving standing control?

## R — Response summary

Verify the one-link finite-difference comparison at 0, 90, and 180 degrees. Verify `J(q) @ qdot` against MuJoCo at two two-joint poses. Keep static support as an inspectable bridge only.

## E — Human evaluation

The learner distinguished a coordinate conversion from a physical force-to-torque mapping. The learner stated that a fixed desired foot force needs new joint torques after a pose change because the leg's force leverage changes.

## A — Action

Ran the focused headless checks. Recorded the portfolio visual requirements: pose-dependent velocity arrows, shared-frame orientation overlay, full FBD, and an interactive fixed-force / changing-torque view.

## O — Outcome

At the initial one-link pose, MuJoCo and centered finite differences agreed on `Jx`: -1.000000 at 0 degrees, approximately 0 at 90 degrees, and +1.000000 at 180 degrees. For the two-joint bent and open poses, `J(q) @ qdot` matched MuJoCo foot X-Z velocity with zero reported norm difference. The same `qdot` produced different task-space velocity because `q` changed the local map.

This is not final #6 closure. The issue still requires one two-DOF planar mechanism across 0, 90, and 180 degrees, including local/base and shared-frame position comparison and static `Δx`/`Δy` probes.

### Effect on current belief

- Before: the Jacobian risked being treated as a coordinate conversion or as a force/stability solver.
- After: `J(q)` is the current-pose local map from joint velocity to foot velocity; `J(q).T` maps a chosen foot force to joint torque through current force leverage. Force allocation and standing stability remain separate #20 work.
- Evidence status: Headless numerical checks and human cold-rep explanation confirm the current subclaims. The issue-level combined two-DOF orientation check remains unverified.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-jacobians-task-space/agent-log.md
  commit:
provenance:
  - kind: human-observation
    system: Codex
    reference: current task
  - kind: coding-agent
    system: Codex
    reference: headless Jacobian verification
objects:
  - id: Q-20260810-015
    type: Question
    summary: Determine whether #6 is complete without expanding into standing control.
    status: open
  - id: R-20260810-015
    type: Response
    summary: Verify velocity and finite-difference mappings and preserve visual teaching requirements.
    status: acted
  - id: E-20260810-015
    type: Evaluation
    summary: The learner correctly separated coordinate descriptions from physical force leverage.
    status: confirmed
  - id: A-20260810-015
    type: Action
    summary: Ran focused checks and documented visual requirements for portfolio projection.
    status: completed
  - id: O-20260810-015
    type: Outcome
    summary: Numerical and conceptual subchecks passed; issue-level combined two-DOF orientation evidence remains missing.
    status: unconfirmed
relations:
  - subject: Q-20260810-015
    predicate: receives
    object: R-20260810-015
  - subject: R-20260810-015
    predicate: receives
    object: E-20260810-015
  - subject: E-20260810-015
    predicate: causes
    object: A-20260810-015
  - subject: A-20260810-015
    predicate: produces
    object: O-20260810-015
unresolved_questions:
  - Can one two-DOF planar mechanism show the required local-versus-shared-frame static probes across 0, 90, and 180 degrees?
  - How should #20 allocate feasible contact forces and use feedback to stand?
superseded_claims:
  - The Jacobian is only a coordinate conversion.
  - The Jacobian itself allocates foot forces or solves standing stability.
```

Related: #6, #20
