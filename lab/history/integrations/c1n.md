# C-1N integration target

Target repository: `haidmoham/spider` (legacy repository slug; public robot identity: **C-1N**)

C-1N is the longitudinal simulation robot. It integrates bench concepts and preserves visible checkpoints as the simulation becomes more physically grounded, statistically evaluated, and increasingly learned.

The current cross-repository standing requirements and gates live in
[`requirements.md`](requirements.md).

## Identity and checkpoint grammar

Public checkpoints use:

```text
C-1N // NN · CODENAME
```

The number preserves chronology. The codename records a capability or understood failure worth comparing with prior behavior.

Current lineage:

```text
C-1N // 00 · SPAWN    historical deterministic six-foot spawn baseline
C-1N // 01 · SHUFFLE  current coordinated gait failure
C-1N // 02 · STAND    recorded six-contact baseline; see canonical C-1N README
C-1N // 03 · STRIDE   reserved for first materially better sustained walk
```

`SPAWN` does not demonstrate standing. C-1N records STAND as a six-contact
baseline, with disturbance recovery excluded. STRIDE remains reserved.
This records the canonical repository's claim. It does not certify the human's
present understanding. `TODO.md` selects the user-written RL/PPO route.

## Contract

- The test bench isolates questions and repairs mental models.
- C-1N integrates learned mechanisms into one evolving robot.
- Treat this work as a situated engineering apprenticeship. The objective is
  both a working robot and the user's end-to-end physical understanding of why
  it works or fails.
- Use Jupyter as the shared reasoning surface. Prepare geometry, force, or
  telemetry fixtures and explain the observables. Follow the shared practice
  rule in `AGENTS.md`: get a real user attempt in conversation before advancing
  the learning step. Do not require a numbered prediction form or code gate.
- Do not supply the user's attempt for them. Give requested explanations and
  offer hints. Do not mistake response time for a blocker.
- Use the active goal as a routing and quality guardrail. It must not pressure
  the user, replace explanation with hill-climbing, or redefine understanding
  as a secondary deliverable.
- Preserve useful failures. Do not rewrite the project history into a clean final demo.
- C-1N is simulation-first. Hardware integration is not a required checkpoint or graduation step.
- Physics and controls remain necessary because learned and simulated behavior must be physically interpretable.
- After the standing foundation, prefer integrations that deepen simulation, statistics, optimization, evaluation, model inference, uncertainty, or learned behavior.
- A bench issue never requires C-1N work for closure unless its own contract explicitly defines a post-close integration checkpoint.
- Instrumentation alone does not require a public checkpoint.
- Legacy issue #6 torso-frame foot task-space telemetry remains an optional non-checkpoint integration.
- Keep browser and WASM work downstream of useful simulation behavior. The web surface exposes evidence; it does not create the learning target.

## Standing bridge and evidence boundary

The following bridge records the standing integration requirements. The bench
now contains #24 and #31 evidence. C-1N records its STAND baseline. Preserve
these sources; use `TODO.md` and the [notebook guide](../../spider/LEARNING.md)
to select current work. The requirements ledger preserves the historical rows
and their reconciled status. It does not select an additional standing prerequisite.

First isolate static support in the bench. Define standing through contact geometry, center-of-mass projection, support load, body moment, and reproducible rollout behavior.

Then transfer that measurement model to C-1N with the gait clock disabled.

The `C-1N // 02 · STAND` checkpoint requires:

- support-aware contact evidence;
- center-of-mass/support geometry in a shared frame;
- torso attitude over a fixed rollout;
- a stated success tolerance or failure condition;
- repeatable evidence that the robot maintains stable support rather than only initializing into a plausible pose.

Standing is required because later locomotion objectives and evaluation need a physically meaningful baseline. It is not a commitment to a controls or hardware career path.

## Simulation-first hook targets

| Bench lane | C-1N integration |
| --- | --- |
| #24 Foundation — support state | Establish the first support-aware stable stance and preserve it as `C-1N // 02 · STAND`. |
| #25 Learn — learned locomotion | Train a locomotion policy. Preserve objective exploits and the first understandable learned failure. `STRIDE` requires materially better sustained locomotion under fixed evaluation. |
| #26 Evaluate — behavior as a distribution | Evaluate C-1N across fixed seeds, initial conditions, and scenarios. Compare distributions, not cherry-picked rollouts. |
| #27 Model — simulator calibration | Hide one interpretable C-1N model parameter, estimate it from one rollout set, and validate it on another. |
| #28 Uncertainty — distributions and shift | Randomize a small set of understood physical or sensing parameters and measure in-distribution and held-out degradation. |
| #29 Differentiate — differentiable dynamics | Propagate rollout loss to one interpretable simulated quantity and verify the gradient numerically. |
| #30 Scale — simulation systems | Make C-1N rollouts reproducible, batchable, observable, and fast enough for population-level experiments. |

These are lanes, not a fixed order after `STAND`. Learned locomotion is the first forcing function. Its failures choose which lane becomes useful next.

## Post-foundation simulation frontier

After locomotion and population evaluation are credible, extend the scale lane into procedurally generated, deterministically validated world populations. C-1N is one useful policy-under-test, not the only generator of future requirements.

The future control plane should let an agent compile a goal such as “generate uneven-terrain worlds within declared slope and friction bounds; reject invalid cases; evaluate the policy; cluster failures; return minimal reproducible cases” into typed generation, validation, rollout, and analysis operations.

This remains a deep-toy architectural objective. It does not create a new checkpoint or bypass the standing and locomotion evidence gates. Validators, declared treatments, retained failures, provenance, and simulator contracts remain authoritative.

## Supporting mechanisms

Legacy topics such as trajectory tracking, contact mechanics, actuator limits, state estimation, or numerical sensitivity are not deleted knowledge. They are no longer permanent open routes.

Bring one back when a current C-1N or bench failure makes it causally necessary. Create a focused experiment for the actual failure instead of restoring the old concept graph.

## Version rule

Create a new checkpoint only when a capability or understood failure is worth preserving and comparing with prior behavior. Do not increment for instrumentation, cleanup, presentation polish, a new training run, or elapsed time.

The intended evidence loop is:

`bench question -> prediction -> simulation evidence -> model update -> C-1N integration -> population evaluation -> new failure -> next bench question`
