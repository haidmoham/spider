# Agent log

## AI-20260809-009 — Computed-torque mismatch prediction

id: AI-20260809-009
date: 2026-08-09
agent: Codex
status: resolved
evaluation: confirmed
sources:
  - kind: chat
    system: Codex
    reference: 2026-08-09 model-based-control learning session
  - kind: coding-agent
    system: Codex
    reference: experiments/2026-08-09-model-based-control/model_based_control.py headless 16-second runs
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-model-based-control/model_based_control.py
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-009
  response: R-20260809-009
  evaluation: E-20260809-009
  action: A-20260809-009
  outcome: O-20260809-009
librarian:
  status: pending
  record_ids: []

## Q — Question

With the plant, trajectory, feedback gains, and timestep fixed, what changes when inverse dynamics supplies the desired motion torque? What changes if the controller alone assumes link 2 has half its real mass?

## R — Response summary

The human prediction is two-stage. Accurate feedforward should be most useful near target reversals, where desired acceleration magnitude is largest. With an underestimated controller mass, the initial feedforward torque should be too weak. Error should then grow until feedback applies a larger correction, which can create a larger acceleration magnitude or overshoot.

## E — Human evaluation

### Accepted

- The physical plant remains unchanged at a 0.7 kg link-2 mass.
- The mismatch is only in the controller's 0.35 kg inverse-dynamics model.
- The test must distinguish initial under-acceleration from any later, larger corrective acceleration.

## A — Action

Add accurate computed torque and controller-only wrong-mass conditions. Record tracking error, acceleration error, feedback torque, feedforward torque, and turnaround-window RMS values over the final complete trajectory cycle.

## O — Outcome

The final complete cycle, 7.853982 to 15.707963 seconds, was simulated headlessly for all four conditions.

- Accurate computed torque reduced position-error RMS to [0.000414, 0.000519] rad and feedback-torque RMS to [0.000536, 0.000149] N-m. Its turnaround position-error RMS was [0.000154, 0.000178] rad.
- The controller-only wrong-mass condition increased position-error RMS to [0.268017, 0.053296] rad, acceleration-error RMS to [0.035255, 0.018318] rad/s², and feedback-torque RMS to [4.826850, 0.962259] N-m.
- At the target reversal with desired acceleration [-0.224, -0.288] rad/s², the wrong-mass feedforward torque was [7.318840, 1.166463] N-m instead of [10.963537, 2.332925] N-m. Its shortfall was [3.644697, 1.166463] N-m.

The run confirms that the inaccurate model leaves a large residual that feedback must correct. It does not isolate a time-resolved overshoot event, so the predicted later higher acceleration magnitude remains unconfirmed and is outside this experiment's stop boundary.

### Effect on current belief

- Before: Model-based feedforward was expected to reduce error, but the distinct effect of a controller-only mass error was untested.
- After: Accurate desired-state inverse dynamics greatly reduces the residual in this model. A controller-only mass error restores a measurable residual and feedback burden.
- Evidence status: Experimentally observed for the reported RMS values and reversal torque shortfall. The separate claim about later corrective acceleration remains unconfirmed and is not required for issue #4 closure.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
provenance:
  - kind: chat
    system: Codex
    reference: 2026-08-09 model-based-control learning session
  - kind: coding-agent
    system: Codex
    reference: experiments/2026-08-09-model-based-control/model_based_control.py headless 16-second runs
objects:
  - id: Q-20260809-009
    type: Question
    summary: What does desired-state inverse dynamics change, and what does a controller-only mass error reveal?
    status: resolved
  - id: R-20260809-009
    type: Response
    summary: Accurate feedforward should help near acceleration extrema; an underestimated mass should first under-accelerate, then require stronger feedback correction.
    status: confirmed-with-boundary
  - id: E-20260809-009
    type: Evaluation
    summary: The test design holds plant, target, gains, and timestep fixed while changing only the controller model.
    status: confirmed
  - id: A-20260809-009
    type: Action
    summary: Run accurate and wrong-mass computed-torque conditions with position, acceleration, and torque telemetry.
    status: completed
  - id: O-20260809-009
    type: Outcome
    summary: Accurate feedforward nearly eliminated the residual; a controller-only wrong mass restored position, acceleration, and feedback residuals.
    status: confirmed-with-boundary
relations:
  - subject: Q-20260809-009
    predicate: receives
    object: R-20260809-009
  - subject: R-20260809-009
    predicate: receives
    object: E-20260809-009
  - subject: E-20260809-009
    predicate: causes
    object: A-20260809-009
  - subject: A-20260809-009
    predicate: produces
    object: O-20260809-009
unresolved_questions:
  - Deferred outside issue #4: does the wrong-mass trace show initial under-acceleration followed by a larger correction at a resolved time point?
superseded_claims: []
```

## AI-20260809-004 — Iteration 0 prediction

id: AI-20260809-004
date: 2026-08-09
agent: Codex
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit: 57b626d
  changed_files:
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-004
  response: R-20260809-004
  evaluation: E-20260809-004
  action: A-20260809-004
  outcome: O-20260809-004
librarian:
  status: pending
  record_ids: []

## Q — Question

With the arm, target, and PD gains unchanged, what effect should adding gravity compensation have?

## R — Response summary

The human prediction was that gravity compensation should make the controller's job easier.

## E — Human evaluation

### Accepted

- Gravity compensation should reduce the burden on the PD feedback.
- Refined prediction: the visual motion may look faster or less delayed, the tracking error may differ with the arm's configuration, and the two controllers will differ in whether their command includes a gravity-related term.

### Clarification

- The experiment changes neither the PD gains nor the plant gravity. It only adds a model-predicted joint-torque vector to the unchanged PD command.
- Tracking error is measured relative to the moving target trajectory, not a fixed reference point.
- Both plants experience gravity; only one controller explicitly estimates and adds a compensating torque.
- “Faster” is not yet a precise prediction because the feedback gains and plant dynamics are unchanged.

### Unresolved

- Whether the visual motion, periodic steady-state tracking error, and feedback control effort show that prediction.

## A — Action

Recorded the Iteration 0 prediction before launching the two visual comparison conditions.

## O — Outcome

Pending visual comparison.

### Effect on current belief

The belief that compensation makes the control job easier is retained, while “faster” has been marked as an ambiguous visual proxy rather than a direct speed prediction. No experiment observation has changed the belief yet.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
objects:
  - id: Q-20260809-004
    type: Question
    summary: What should gravity compensation change when the plant, target, and PD gains are fixed?
    status: answered-as-prediction
  - id: R-20260809-004
    type: Response
    summary: Gravity compensation should reduce the burden on PD feedback.
    status: recorded
  - id: E-20260809-004
    type: Evaluation
    summary: The prediction is plausible but still needs observation and telemetry.
    status: unconfirmed
  - id: A-20260809-004
    type: Action
    summary: Recorded Iteration 0 before running either controller condition.
    status: completed
  - id: O-20260809-004
    type: Outcome
    summary: Visual and telemetry comparison pending.
    status: pending
relations:
  - subject: Q-20260809-004
    predicate: receives
    object: R-20260809-004
  - subject: R-20260809-004
    predicate: receives
    object: E-20260809-004
  - subject: E-20260809-004
    predicate: causes
    object: A-20260809-004
  - subject: A-20260809-004
    predicate: produces
    object: O-20260809-004
unresolved_questions:
  - Do the visual motion and telemetry support the prediction?
superseded_claims: []
```

Related: #4

## AI-20260809-010 — Overlay comparison aid

id: AI-20260809-010
date: 2026-08-09
agent: Codex
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - experiments/2026-08-09-model-based-control/model_based_control.py
    - experiments/2026-08-09-model-based-control/README.md
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-010
  response: R-20260809-010
  evaluation: E-20260809-010
  action: A-20260809-010
  outcome: O-20260809-010
librarian:
  status: pending
  record_ids: []

## Q — Question

Can the two controller states be compared visually without placing them in one physical simulation?

## R — Response summary

Run the PD and gravity-compensation controllers in separate MuJoCo data objects. Render a target-pose ghost plus the two controller states, and show live rolling plots at the viewer edge as visual aids. Add phase-lag and turnaround-window metrics to the headless summary.

## E — Human evaluation

### Accepted

- The overlay is for visual comparison only.

### Verification required

- Confirm that the two arms render together and that existing headless telemetry is unchanged.

## A — Action

Added `--controller overlay`. It advances both controllers independently, adds a gray target ghost and the orange arm as user-scene geoms over the translucent blue PD arm, and shows right-side live plots for the latest six seconds of applied torque plus its first and second numerical derivatives. The headless comparison now reports phase lag and turnaround-window error.

## O — Outcome

Pending visual inspection. The overlay is not experiment evidence and does not change the controller comparison.

### Effect on current belief

None. This is instrumentation for inspection, not a result.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
objects:
  - id: Q-20260809-010
    type: Question
    summary: Can two controller states be compared in one viewer without sharing physics?
    status: answered-by-design
  - id: R-20260809-010
    type: Response
    summary: Use independent data objects and a visualization-only overlay with compact rolling plots.
    status: implemented
  - id: E-20260809-010
    type: Evaluation
    summary: Visual inspection and telemetry-preservation checks remain required.
    status: unconfirmed
  - id: A-20260809-010
    type: Action
    summary: Added a viewer-only overlay mode with compact derivative plots for the PD and gravity-compensation states.
    status: completed
  - id: O-20260809-010
    type: Outcome
    summary: Pending visual inspection; not experiment evidence.
    status: pending
relations:
  - subject: Q-20260809-010
    predicate: receives
    object: R-20260809-010
  - subject: R-20260809-010
    predicate: receives
    object: E-20260809-010
  - subject: E-20260809-010
    predicate: causes
    object: A-20260809-010
  - subject: A-20260809-010
    predicate: produces
    object: O-20260809-010
unresolved_questions:
  - Does the overlay make the controller-state difference easier to inspect?
superseded_claims: []
```

## AI-20260809-007 — Final-cycle telemetry comparison

id: AI-20260809-007
date: 2026-08-09
agent: Codex
status: acted
evaluation: confirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit: 869c764
  changed_files:
    - experiments/2026-08-09-model-based-control/model_based_control.py
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-007
  response: R-20260809-007
  evaluation: E-20260809-007
  action: A-20260809-007
  outcome: O-20260809-007
librarian:
  status: pending
  record_ids: []

## Q — Question

Does the final full target cycle show the gravity-shaped burden moving from baseline feedback torque into an explicit gravity term under compensation?

## R — Response summary

Both modes were run headlessly for 16 simulated seconds with identical model, target, initial pose, timestep, and gains. RMS values were computed over the final complete target cycle, 7.853982 to 15.707963 seconds. The projection coefficient tests whether baseline feedback is aligned with the model gravity term.

## E — Human evaluation

### Accepted

- Baseline PD RMS tracking error was [0.679127, 0.082077] radians; gravity compensation was [0.019344, 0.004964].
- Baseline PD feedback-torque RMS was [12.226788, 1.485622] N-m; gravity compensation feedback-torque RMS was [0.356078, 0.093290].
- Model gravity-torque RMS was [12.255165, 1.534452] N-m in the baseline comparison and [12.846722, 2.229886] N-m under compensation.
- Baseline feedback projected onto the model gravity term with coefficients [0.997479, 0.967702] per joint.
- Gravity-compensation total-torque RMS was [12.869362, 2.194443] N-m, so total effort did not fall in the same way as the feedback residual.

## A — Action

Added deterministic 16-second headless execution and telemetry decomposition for target error, feedback torque, model gravity torque, and applied total torque. Summarized the final complete target cycle and recorded the baseline feedback-to-gravity projection.

## O — Outcome

The telemetry supports the existing error-dynamics explanation. Baseline feedback torque was almost one-for-one aligned with the model gravity term, while gravity compensation made the feedback residual and tracking error much smaller. The explicit gravity term carried the burden in the compensated run; total applied torque remained similar rather than disappearing.

### Effect on current belief

The visual “easier” interpretation is quantitatively supported. The result does not support a claim that total torque is lower; it supports a claim that gravity-shaped load is moved out of error-driven feedback.

Related: #4

## AI-20260809-005 — Visual comparison observation

id: AI-20260809-005
date: 2026-08-09
agent: Codex
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit: 519eef4
  changed_files:
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-005
  response: R-20260809-005
  evaluation: E-20260809-005
  action: A-20260809-005
  outcome: O-20260809-005
librarian:
  status: pending
  record_ids: []

## Q — Question

What visible difference does the gravity-compensation condition produce relative to baseline PD?

## R — Response summary

The human observed that the gravity-compensation run looked more fluid, as if the whole arm moved in greater unison.

## E — Human evaluation

### Accepted

- The visual difference was noticeable in the orange gravity-compensation condition.

### Unresolved

- Whether the appearance corresponds to smaller target-tracking error, lower feedback correction, or a timing/viewpoint difference.

## A — Action

Recorded the visual observation without treating its cause as verified.

## O — Outcome

Pending telemetry comparison.

### Effect on current belief

The prediction that gravity compensation may make the control job easier is qualitatively supported by the visual impression, but the mechanism remains unconfirmed.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
objects:
  - id: Q-20260809-005
    type: Question
    summary: What visible difference does gravity compensation produce relative to baseline PD?
    status: answered-by-observation
  - id: R-20260809-005
    type: Response
    summary: The compensated arm looked more fluid and coordinated.
    status: observed
  - id: E-20260809-005
    type: Evaluation
    summary: The visual difference is real, but its mechanism is not established.
    status: unconfirmed
  - id: A-20260809-005
    type: Action
    summary: Recorded the visual observation without causal overclaiming.
    status: completed
  - id: O-20260809-005
    type: Outcome
    summary: Telemetry comparison remains pending.
    status: pending
relations:
  - subject: Q-20260809-005
    predicate: receives
    object: R-20260809-005
  - subject: R-20260809-005
    predicate: receives
    object: E-20260809-005
  - subject: E-20260809-005
    predicate: causes
    object: A-20260809-005
  - subject: A-20260809-005
    predicate: produces
    object: O-20260809-005
unresolved_questions:
  - Does the visual difference correspond to lower feedback effort or tracking error?
superseded_claims: []
```

Related: #4

## AI-20260809-006 — Mathematical interpretation

id: AI-20260809-006
date: 2026-08-09
agent: Codex
status: acted
evaluation: unconfirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit: eff6885
  changed_files:
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  issues: [4]
objects:
  question: Q-20260809-006
  response: R-20260809-006
  evaluation: E-20260809-006
  action: A-20260809-006
  outcome: O-20260809-006
librarian:
  status: pending
  record_ids: []

## Q — Question

Why might the gravity-compensation arm look more coordinated or fluid?

## R — Response summary

For manipulator dynamics M(q)qdd + C(q,qdot)qdot + g(q) = tau, baseline PD leaves g(q) as a disturbance. PD plus gravity compensation adds an estimate of g(q), so the feedback loop handles mainly tracking error and residual dynamic terms.

## E — Human evaluation

### Accepted

- The hypothesis is mathematically plausible: removing the configuration-dependent gravity forcing can reduce uneven corrective errors between joints.

### Unresolved

- The total torque command need not be smaller because it includes the explicit gravity torque.
- The current visual impression still needs a split between feedback torque and gravity torque to test “easier” directly.

## A — Action

Derived the error dynamics and evaluated the initial gravity-torque vector numerically at the experiment's starting pose.

## O — Outcome

At q=(0.35,-0.70), the model's gravity torque is approximately (13.64, 2.58) N·m. A static PD-only balance estimate with Kp=18 is approximately (0.758, 0.143) radians of position error; this is a nonlinear, moving-target comparison aid, not an observed trajectory result.

### Effect on current belief

The fluid-motion observation is consistent with gravity compensation removing a large configuration-dependent forcing term, but visual smoothness alone does not establish lower feedback effort or lower total control effort.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
objects:
  - id: Q-20260809-006
    type: Question
    summary: Why might gravity compensation make the arm look more coordinated or fluid?
    status: answered-as-hypothesis
  - id: R-20260809-006
    type: Response
    summary: Gravity compensation removes g(q) from the feedback loop's apparent disturbance when the model is accurate.
    status: mathematically-supported
  - id: E-20260809-006
    type: Evaluation
    summary: The explanation is plausible, but telemetry must separate feedback, gravity, and total torque.
    status: unconfirmed
  - id: A-20260809-006
    type: Action
    summary: Derived the error dynamics and evaluated gravity torque at the starting pose.
    status: completed
  - id: O-20260809-006
    type: Outcome
    summary: The explanation predicts lower feedback burden, not necessarily lower total torque.
    status: hypothesis-refined
relations:
  - subject: Q-20260809-006
    predicate: receives
    object: R-20260809-006
  - subject: R-20260809-006
    predicate: receives
    object: E-20260809-006
  - subject: E-20260809-006
    predicate: causes
    object: A-20260809-006
  - subject: A-20260809-006
    predicate: produces
    object: O-20260809-006
unresolved_questions:
  - Does measured feedback torque decrease under gravity compensation while tracking error improves or stays comparable?
superseded_claims:
  - Treating lower total actuator torque as the direct meaning of “easier” for the feedback controller.
```

Related: #4

## AI-20260809-008 — Prescriptive vs descriptive dynamics repair

```yaml
id: AI-20260809-008
date: 2026-08-09
sources:
  - kind: chat
    system: ChatGPT
    reference: conversation-local: sinusoid-gravity-compensation
status: resolved
evaluation: confirmed
repo_state:
  repository: robotics-test-bench
  branch: main
  commit:
  changed_files:
    - AGENTS.md
    - templates/agent-interaction.md
    - experiments/2026-08-09-model-based-control/agent-log.md
related:
  experiment: experiments/2026-08-09-model-based-control
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: [4]
  files:
    - experiments/2026-08-09-model-based-control/model_based_control.py
objects:
  question: Q-20260809-008
  response: R-20260809-008
  evaluation: E-20260809-008
  action: A-20260809-008
  outcome: O-20260809-008
librarian:
  status: pending
  record_ids: []
```

## Q — Question

How should the sinusoidal target, gravity term, and gravity compensation be distinguished conceptually, and what knowledge gap was blocking the current interpretation?

### Human prediction

A useful distinction might be that the sinusoidal trajectory is prescriptive while the gravity function is descriptive. Better physics intuition might make the relationship easier to understand.

### Purpose

Repair the smallest conceptual gap blocking interpretation of the current model-based-control experiment without leaving the experiment for a broad prerequisite study.

### Context supplied

- The current sinusoidal joint target.
- The existing gravity-compensation comparison and telemetry decomposition.
- The observed result that baseline feedback torque is strongly aligned with the model gravity term.

## R — Response summary

The useful abstraction is the role of each function. The target trajectory specifies desired motion. The gravity term describes configuration-dependent torque. Evaluating the gravity model along a moving trajectory can make that descriptive term vary sinusoidally in time. Gravity compensation then uses the descriptive model to prescribe an explicit compensating torque.

## E — Human evaluation

### Accepted

- Prescriptive versus descriptive is a useful distinction for reading the dynamics.
- The missing knowledge was narrower than “learn more physics”: the blocker was knowing which abstraction mattered in the current equation.
- The repair made the surrounding mechanics easier to interpret in context.
- This interaction is evidence that local knowledge gaps can be identified and repaired while continuing robotics work.

### Rejected

- A broad physics course is required before the current MuJoCo work can be understood.

### Unresolved

- Future mechanics gaps still need their own local repair or experiment; this interaction does not establish that every conceptual explanation is correct without verification.

### Verification required

- No new experiment is required to record the epistemic change itself.
- Physical claims should continue to be checked against model inspection or telemetry when they become decision-relevant.

## A — Action

Extended the repository interaction convention to require explicit source provenance and explicit before -> after belief updates. Recorded this conversation as the first `chat`-sourced interaction under the existing Q/R/E/A/O ontology. No controller code changed.

## O — Outcome

The current gravity-compensation telemetry can now be read through a cleaner mental model: desired trajectory is an instruction, gravity is a modeled physical effect, and compensation turns that model into a control action. The conversation produced a durable epistemic repair without becoming a transcript or being mislabeled as experimental evidence.

### Effect on current belief

- Before: The relationship among sinusoidal motion, gravity, and compensation felt like evidence of a broad physics-intuition gap.
- After: The blocking gap is better described as an abstraction-role gap: distinguish desired motion from modeled dynamics, then use the model to reason about control.
- Evidence status: This is a human-accepted conceptual repair from a ChatGPT conversation, consistent with the already recorded MuJoCo telemetry. It is not new experimental evidence.

## Librarian update

```yaml
source:
  repository: robotics-test-bench
  path: experiments/2026-08-09-model-based-control/agent-log.md
  commit:
provenance:
  - kind: chat
    system: ChatGPT
    reference: conversation-local: sinusoid-gravity-compensation
objects:
  - id: Q-20260809-008
    type: Question
    summary: Which abstraction distinguishes sinusoidal target motion, gravity, and gravity compensation?
    status: resolved
  - id: R-20260809-008
    type: Response
    summary: Desired trajectory is prescriptive, gravity is descriptive, and compensation uses the descriptive model to prescribe control torque.
    status: accepted
  - id: E-20260809-008
    type: Evaluation
    summary: The blocker was narrowed from broad physics knowledge to the role of each function in the dynamics.
    status: confirmed
  - id: A-20260809-008
    type: Action
    summary: Extended source-provenance conventions and recorded the conversation as a structured epistemic repair.
    status: completed
  - id: O-20260809-008
    type: Outcome
    summary: The model-based-control experiment is easier to interpret without pausing for broad prerequisite study.
    status: resolved
relations:
  - subject: Q-20260809-008
    predicate: receives
    object: R-20260809-008
  - subject: R-20260809-008
    predicate: receives
    object: E-20260809-008
  - subject: E-20260809-008
    predicate: causes
    object: A-20260809-008
  - subject: A-20260809-008
    predicate: produces
    object: O-20260809-008
unresolved_questions: []
superseded_claims:
  - Broad physics study is required before the current model-based-control experiment can be understood.
```

Related: #4
