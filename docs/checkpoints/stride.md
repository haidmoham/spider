# C-1N v0.3 - STRIDE

**Status: complete. Checkpoint: `walk_fast_500`. Accepted: 2026-09-15.**

The user explicitly requested: "mark STRIDE as completed by walk_fast_500".
This accepts learned forward locomotion on the six-leg candidate chassis
within the recorded flat-ground evaluation scope.

## Evidence

- [Checkpoint manifest](../../artifacts/walk_fast_500/manifest.json): exact
  weights, parent checkpoint, device information, and SHA-256 identity.
- [Experiment archive](../../artifacts/walk_fast_500/experiments/cuda-continuation-20260915/README.md):
  model, objective/configuration, frozen source, training log, checkpoints,
  full evaluation records, and recorded simulation states.
- Final protocol: seeds 201 through 212, five seconds each in sampled and mean
  modes. All 24 runs completed with finite states, valid actuator targets,
  zero falls, and zero measured joint-limit violations. Deterministic mean
  repetitions share an initial state and are not independent robustness trials.
- Mean-policy speed: **0.887987 m/s**. Sampled average: **0.821139 m/s**.
  Both exceed the accepted original PPO-100 mean threshold, **0.37882745 m/s**.
  The locked candidate-chassis `walk_stable_100` mean is **0.317506 m/s**.
- Posture, clearance, contact timing, joint motion, and stance-foot speed are
  preserved in the traces and acceptance records. The n=500 four-pane viewer
  replays the measured control, stable walk, sampled policy, and mean policy at 1x.

The original model, STAND evidence, crude PPO-100, and Notebook 04 remain controls.
The stable-to-fast comparison shares the candidate chassis. The comparison
with original PPO-100 includes both chassis and policy changes; it does not
isolate their causal contributions. Earlier chassis feasibility evidence is
preserved in [gait-chassis.md](../gait-chassis.md).

## Limits retained with the capability

Contact fragmentation and slip remain. The n=500 stance-foot speed RMS proxy
is 0.204586 m/s; the locked stable walk's proxy is 0.069600 m/s. This endpoint
estimate is not a full within-step slip measurement or an energy metric.
STRIDE completion does not claim a perfected gait, terrain robustness, push
recovery, hardware transfer, or human mastery. It does not authorize new training.

[Spider issue 17](https://github.com/haidmoham/spider/issues/17) owns the capability.
[Historical test-bench issue 25](https://github.com/haidmoham/robotics-test-bench/issues/25)
remains source provenance, not a new capability or learning claim.
