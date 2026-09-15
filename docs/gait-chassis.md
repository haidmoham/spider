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
command also supports `--notebook-seeds`: all twelve seeds in both sampled and
mean modes, as in Notebook 4. Mean-action seeds use the same initial condition;
they check consistency rather than distinct robustness scenarios.

### Checkpoint comparison

The user authorized one continuous run to Notebook 4's **50 and 100 updates** on
2026-09-15. The run uses seed 11, eight episodes per update, and a maximum of five
seconds per episode. It retains the approved candidate model and prepared reward.
Notebook 4 supplies the evaluation schedule and seeds; its pedagogical action
mapping and chassis are not substituted into the candidate controller.

The local run directory is `telemetry/tuning/20260915-reference-ppo-round-01`.
It retains authorization, frozen sources, model, optimizer states, and every
checkpoint. Use this command on saved checkpoints to reproduce the comparison:

```powershell
.venv/Scripts/python -m spider.reference_comparison --training telemetry/tuning/20260915-reference-ppo-round-01 --output telemetry/tuning/reference-comparison-new
```

This evaluates 24 episodes per checkpoint and opens one window per checkpoint.
Top left is frozen PPO-100. Top right is the approved original untrained tripod.
Bottom left is the candidate sampled policy. Bottom right is its mean policy.
Both candidate panes replay measured seed 201 at 1x. The window title identifies
the checkpoint. Results include per-seed CSV, acceptance JSON, model/state/trace
recordings, viewer logs, and a first-frame screenshot. Failed numerical gates
do not suppress the comparison or promote the candidate.

The completed run produced 800 full training episodes and 200,000 control steps.
At 50 updates, mean speed was **0.299001 m/s** and sampled average was
**0.288070 m/s**. At 100 updates, those speeds were **0.317506 m/s** and
**0.321624 m/s**. Both checkpoints completed all 24 evaluation runs without falls
or joint-limit violations. Both failed the accepted baseline speed gate.
Neither is promoted. Both labelled windows opened at 1x without viewer errors.
[Weights, all 48 recordings, metrics, and source evidence](../artifacts/reference-ppo-evaluation-20260915/README.md)
are preserved independently of the frozen baselines.

### Continue from saved weights

`reference_training --resume CHECKPOINT --updates N` restores actor, critic,
both Adam optimizers, optimizer RNG, seed, saved model and controller settings.
`N` means **additional** updates: resuming n=100 with `--updates 20` ends at n=120.
The output directory must be new. Parent checkpoint path/hash and starting update
are recorded; local step/update logs contain only the continuation segment.

`--frequency-hz` changes the reference cadence and its speed observation while
preserving stroke length, stance fraction, chassis, actuator parameters and weights.
The change is saved in checkpoint provenance. `--prepare-only` saves restored
weights and selected cadence without rollouts or PPO updates. For example:

```powershell
.venv/Scripts/python -m spider.reference_training --resume artifacts/reference-ppo-evaluation-20260915/checkpoint-00100.pt --frequency-hz 1.5 --prepare-only --output telemetry/tuning/cadence-prepared-new
```

Three variants are prepared locally in `telemetry/tuning/20260915-cadence-prepared-01`
at 1.1, 1.35 and 1.5 Hz. Preparation ran no rollouts or continuation updates.
The user then approved three five-second deterministic evaluations and a four-pane
replay. The completed probes used identical actor weights and physical models:

| Cadence | Mean speed | Falls | Stance foot-speed RMS |
| --- | ---: | ---: | ---: |
| 1.1 Hz | 0.317506 m/s | 0/1 | 0.069600 m/s |
| 1.35 Hz | 0.396974 m/s | 0/1 | 0.084540 m/s |
| 1.5 Hz | 0.470187 m/s | 0/1 | 0.125228 m/s |

All three completed five seconds with finite states and no joint-limit violations.
The faster cadences exceed baseline speed on this one mean-action run, without
retraining. Increased stance foot speed and raw contact transitions warrant review
for slip and contact chatter; raw liftoff counts do not identify clean strides.
Neither faster cadence has passed the twelve-seed population gate. No promotion
or new PPO updates occurred. [Saved probe evidence](../artifacts/cadence-evaluation-20260915/README.md).

The user subsequently authorized **100 additional updates at 1.5 Hz**, restoring
the saved n=100 weights and both optimizer states. The continuation directory is
`telemetry/tuning/20260915-reference-ppo-cadence-15-round-01`; its budget ends at
n=200. Evaluations at n=150 and n=200 use both modes and seeds 201–212.
`reference_comparison --control RECORDING` can put the pre-continuation 1.5 Hz
mean recording at top right, so the view separates further training gains from
the cadence change. The original accepted PPO-100 remains top left.

The user stopped that fixed-cadence continuation at saved **n=185**. The last
evaluated checkpoint, n=150, reached 0.518150 m/s mean and 0.516098 m/s sampled
average with zero falls across 24 evaluations and passed the numerical gate.
It remains unapproved. [Stopped experiment evidence](../artifacts/fixed-cadence-continuation-20260915/README.md)
preserves n=150 and n=185; no n=200 completion is claimed.

## Cadence versus more PPO: literature check, 2026-09-15

Our fixed-frequency residual policy learns 18 joint corrections. It cannot choose
cadence. The 1.1→1.5 Hz probe improved mean speed by about 48%, while the next
50 PPO updates at fixed 1.5 Hz improved it by about 10%. These are different
interventions, not evidence that PPO adds no speed.

- [Iscen et al., Policies Modulating Trajectory Generators, CoRL 2018](https://proceedings.mlr.press/v87/iscen18a/iscen18a.pdf)
  lets learned control modify generator parameters, including frequency, as well
  as correct its outputs. This is the missing action in our current implementation.
- [Margolis and Agrawal, Walk These Ways, CoRL 2022](https://gmargo11.github.io/walk-these-ways/)
  demonstrates one trained policy whose gait parameters can be tuned at runtime.
  Its demonstrated flexibility does not establish that our fixed-cadence policy
  will generalize to arbitrary frequencies without testing.
- [Yang et al., Terrain-adaptive CPGs with RL for Hexapod Locomotion, 2023](https://arxiv.org/abs/2310.07744)
  combines synchronized gait generation with learned adjustments for terrain
  adaptation in simulation. This supports studying a generator-plus-feedback
  structure on a six-leg robot, not a transfer guarantee for C-1N.

The user's selected next experiment returns to locked `walk_stable_100` and gives
PPO a cadence action with continuous integrated phase. The original actor outputs
and critic should transfer exactly; new action parameters need fresh optimizer
state. Cadence/phase traces, zero-adjustment parity, and shared training/evaluation
timing are required before running it. Each ablation gets a separate experiment ID.

### Policy-controlled cadence experiment

`spider/cadence_action_training.py` adds a separate cadence output to the existing
18 joint-residual actions. It retains the original actor tensors and their Adam
history, the critic and its Adam history, and the optimizer RNG. New cadence
parameters have fresh Adam state. The locked parent hash is checked on transfer.

The cadence action maps through tanh to 0.8–1.8 Hz around 1.1 Hz, then passes a
0.15-second smoothing filter. The controller integrates the held cadence offset:
`phase = elapsed_seconds * 1.1 + integrated_frequency_offset`. The startup ramp
uses real elapsed time. Changing cadence cannot instantaneously jump phase.
Observation phase and speed follow current cadence; the reference geometry stays
fixed. The reward and PPO settings are unchanged for this ablation.

Training and evaluation use the same controller. A separate action-noise generator
for cadence preserves the legacy 18-action sampling stream. Snapshot schema
`cadence_action_v1` records parent, cadence bounds/filter, networks, both optimizers,
RNG, model XML and sources. Evaluation records command cadence/phase/latent along
with posture, contact, slip and joint motion. Command diagnostics refer to the
held command for that interval; training phase diagnostics are recorded after it.

Preparation verification reproduced every state and timestamp of the locked
five-second mean replay exactly (maximum state error zero). The complete local
suite passed 102 tests. The authorized experiment is
`telemetry/tuning/20260915-policy-cadence-action-round-01`, with 100 additional
updates and evaluations at n=150 and n=200. No additional rounds are implicit.

```powershell
.venv/Scripts/python -m spider.cadence_action_training --from-stable artifacts/walk_stable_100/walk_stable_100.pt --updates 100 --output telemetry/tuning/policy-cadence-action-new
```

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
