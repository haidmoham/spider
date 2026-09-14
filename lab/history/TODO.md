# Experiment queue

This file is the authoritative selector for the next robotics test-bench experiment.

- Exactly one item is `NEXT`.
- GitHub issues are active experiment lanes, not a fixed syllabus.
- Closed legacy issues remain historical provenance only.
- Do not route new work through a closed issue unless a current failure revives its mechanism.
- Re-evaluate the queue after each resolved experiment or integrated C-1N failure.
- Preparing an experiment directory does not authorize solving or running the user's learning work. Follow the shared practice rule before advancing that work.
- `docs/research-platform.md` records long-range design. It does not select current work.

## Current C-1N state

`C-1N // 02 · STAND` is earned. Treat the standing controller, support telemetry,
proximal hinge, deterministic baseline, and known disturbance failures as frozen
evidence. Do not make ROBUST_STAND, contact geometry, or additional standing work
a planned gate before locomotion.

Revisit standing, contact, actuator, estimation, or morphology questions only
when learned locomotion exposes a concrete failure that requires them.

## NEXT

### #25 Learn — user-written RL, PPO, and control treatments

**Status:** NEXT

The user confirmed on 2026-09-10 that STAND is earned as recorded and requested
a clean base before implementing a rudimentary walking policy. Use practice
for the policy and its design. Use production for peripheral cleanup and
verification. Do not implement the policy or infer a completed exercise during
preparation. This confirms the existing route; it does not certify support
understanding or add a standing-robustness gate.

The same day's issue cleanup archived #24, #26-#30, and #35-#37 as not planned.
No remaining implementation or learning outcome was completed by closing them.
Only #25 is active here, paired with `haidmoham/spider#17`. Historical issue
bodies and experiment records remain available. Reopen a lane or create a
bounded experiment only when current work supplies a concrete reason.

Start with the [C-1N notebook learning route](../spider/LEARNING.md).
Trace one control step before defining the learning problem. Then write a basic
RL loop and PPO with PyTorch. Keep observation selection, reward terms, episode
boundaries, action timing, and interpretation with the human.

Compare fixed neutral targets, the existing stance controller, and PPO under
shared evaluation conditions. Begin with joint-target offsets through the
existing actuators. Compare other action interfaces after the first treatment
comparison is understood.

Move into the smallest useful policy-learning loop from issue #25.
The first learned gait can be ugly. Its job is to make

`objective -> policy -> physical behavior -> failure`

inspectable.

Before training, discuss the user's first attempt at the objective, including
how it could reward behavior that misses the intended motion. Keep that
iteration conversational; no prediction form or code gate is required.

The first C-1N locomotion policy should preserve rollout state, actions, objective
terms, seeds, policy checkpoints, and fixed evaluation scenarios. Do not promote
`C-1N // 03 · STRIDE` from one attractive rollout. STRIDE requires materially
better sustained locomotion under fixed evaluation.

## Route reconciliation — 2026-09-04

Source: the user's explicit refactor and algorithm-ownership instructions, plus
remote state refreshed during landing. The earlier local checkout lagged the
remote support evidence and Ant scaffold. Preserve the merged #24 and #31
experiments and all stable records. Do not recreate them as prerequisites.

The older #25 instruction and Ant-v5 scaffold use an existing PPO trainer.
The current user instruction supersedes that algorithm-ownership rule: the
human writes RL and PPO. Preserve the Ant scaffold as prior work. It does not
select a framework or supply a new prediction for the current route.

This is a route decision, not experiment closure. Existing implementation and
bench evidence do not establish the human's present independent understanding.
Keep #25 as the single next lane. Follow the initial control-step notebook with
the human's learning-problem design and prediction before training.

## After first learned locomotion

Let the first understandable learned failure select the next lane.

Current lane and archived references (not queued prerequisites):

- #25 Learn — active: objective -> policy -> behavior.
- #26 Evaluate — archived: treat behavior as a distribution.
- #27 Model — archived: identify and calibrate simulator parameters from rollouts.
- #28 Uncertainty — archived: train and test across distributions and shift.
- #29 Differentiate — archived: backpropagate through simulated dynamics.
- #30 Scale — archived: make simulation experiments reproducible, observable, and fast.

Controls, contact mechanics, actuator limits, state estimation, numerical methods,
and other robotics concepts are supporting mechanisms. Pull one back in only
when a concrete simulation failure makes it necessary.

The intended direction is:

`physical intuition -> support mechanics -> leg reachability -> STAND -> learned locomotion -> distributional evaluation -> system identification and calibration -> uncertainty and randomization -> scalable simulation`

## Long-range design

`docs/research-platform.md` owns the design for procedural validated worlds,
reproducible rollout populations, and scientifically constrained agentic
experimentation.

Keep that design inactive until both conditions exist:

1. A learned locomotion loop can produce rollout populations.
2. A concrete failure requires more scale, reproducibility, validation, or structured analysis.

Use current experiment failures to select the next learning or engineering block.
Do not build platform infrastructure only because it appears in the long-range design.
