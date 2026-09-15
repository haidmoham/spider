# C-1N gait and chassis work

## User decisions, 2026-09-15

- Preserve Notebook 4 and the accepted PPO-100 as frozen controls.
- Chassis geometry, joint axes/ranges and physics may now be reconsidered in
  isolated candidate models. This supersedes the earlier no-physics-change
  constraint for candidate work; it does not authorize replacing the controls.
- Exactly **six legs**, three per side, is invariant.
- The original-chassis untrained tripod is the [approved gait baseline](../artifacts/tripod-openloop-baseline-20260915/README.md).
- The [candidate silhouette](../artifacts/chassis-design-20260915/README.md) is the
  approved design direction: metallic black, red lights/accents, white googly
  eyes. Pupils respect gravity. Red lights breathe and pulse.
- Prepare policy training after visual review. Each substantial training/tuning
  round still requires a concrete, bounded budget. None was started in this pass.

## Evidence and limits

All speeds below are actual displacement divided by five seconds. The open-loop
controllers used one canonical-reset rollout each, not twelve independent tests.

| Chassis | Controller | Speed (m/s) | Maximum tilt | Fall |
|---|---|---:|---:|---|
| Original | Accepted PPO-100 mean | 0.37883 | see accepted archive | no |
| Original | Wave reference | 0.10886 | 4.54 degrees | no |
| Original | Ripple reference | 0.21715 | 6.94 degrees | no |
| Original | Approved tripod reference | 0.30616 | 7.71 degrees | no |
| Candidate | Wave reference | 0.11906 | 4.93 degrees | no |
| Candidate | Ripple reference | 0.20753 | 5.12 degrees | no |
| Candidate | Tripod reference | 0.28052 | 6.34 degrees | no |

The candidate used the same foot anchors, 18 cm stroke, 4 cm lift, 37 cm torso
height and gait timing. The new joints reduced peak tripod tilt but also speed.
These observations do not establish a cause or superiority across scenarios.

The original 22 cm stroke / 5.5 cm lift sketch exceeded reachable distance or
joint limits. Exact inverse kinematics checks selected a reachable envelope
before dynamics were run. Foot locations match MuJoCo forward kinematics; stance
feet remain fixed under prescribed translation; lift/plant velocity is continuous.

## Reproduce the previews

These commands are explicit bounded evaluations or rendering, not training.
Each output directory must be new. The first command needs the accepted mean
recording in the existing local forward/exploration round.

```powershell
.venv/Scripts/python -m spider.gait_preview --output telemetry/tuning/gait-preview-new
.venv/Scripts/python -m spider.gait_feasibility --output telemetry/tuning/gait-physics-new
.venv/Scripts/python -m spider.gait_feasibility --chassis candidate --output telemetry/tuning/candidate-physics-new
.venv/Scripts/python -m spider.chassis_preview --measured telemetry/tuning/candidate-physics-new/tripod --output telemetry/tuning/chassis-review-new
```

`chassis_preview --refresh-appearance` can render existing dynamics with current
styling only if physical-model equality passes. It does not rerun or alter states.
The standard movie is 5 seconds, 1280 x 720, 25 fps, 125 frames, H.264, at 1x.
Kinematic target poses are always labelled separately from measured physics.

## Training direction

PPO will learn bounded corrections around the approved tripod structure.
Zero deterministic correction must reproduce the underlying candidate reference
controller. Normalizing observations must use the candidate's limits. Reward
must credit signed forward travel and retain fall/stability constraints.
Actor KL stopping must not cancel critic fitting.

`spider/reference_training.py` now implements this opt-in preparation without
changing the old trainers. It has a zero-output actor, local sampled-action RNG,
candidate-specific joint/actuator bounds, filtered bounded corrections, signed
forward reward, and independent actor/critic update loops. Checkpoints include
the model XML, reference anchors/config, controller contract and optimizer state.
The runtime rejects an incompatible physical model.

Preparation is explicit and does not collect trajectories or optimize a policy:

```powershell
.venv/Scripts/python -m spider.reference_training --prepare-only --seed 11 --output telemetry/tuning/reference-prepared-new
```

The local reviewed preparation is `telemetry/tuning/20260915-reference-ppo-prepared-01`.
It contains an explicitly untrained `checkpoint-00000.pt`. There are no training
step or update logs because no training was run.

After an agreed budget, `--updates N` selects training instead of preparation.
Each update uses eight episodes of at most five seconds. An explicit follow-up
evaluation command is available; it never promotes a policy automatically:

```powershell
.venv/Scripts/python -m spider.reference_evaluation --checkpoint PATH_TO_CHECKPOINT --output telemetry/tuning/reference-evaluation-new
```

That command runs mean seed 201 plus sampled seeds 201 through 212. It records
model/state/trace evidence and tests full duration, falls, finite states, joint
limits (1 milliradian numeric tolerance), and both speed thresholds. Foot-slip
diagnostics use contact endpoints at the control cadence and can miss within-step
sliding. The numeric gate still requires visual acceptance. The new evaluation
command was prepared and its rejection logic tested; no 13-episode learned-policy
evaluation has been run in this pass.

Acceptance remains a user-approved normal-speed learned stride, a full five-second
mean-action run, and twelve fixed five-second sampled runs with zero falls and
average speed at least **0.37882745 m/s**. A design animation, an untrained controller,
or improved reward alone does not satisfy this requirement. Any model change
must be reported separately from policy improvement.

## Research sources

- [Policies Modulating Trajectory Generators](https://research.google/pubs/policies-modulating-trajectory-generators/): periodic references with learned modulation/corrections.
- [Walk These Ways](https://gmargo11.github.io/walk-these-ways/): posture, foot swing and gait commands for a learned controller.
- [Hexapod motion priors](https://arxiv.org/html/2511.03167v1): motion-style learning and comparisons against task rewards alone; its compute budget is not a local runtime promise.

These motivate experiments; none is evidence that this C-1N candidate beats its control.
