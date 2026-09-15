# Approved tripod gait baseline — 2026-09-15

`TRIPOD-OPENLOOP-20260915` preserves the user-approved **untrained trajectory
controller** on the original chassis. The user reviewed the four-pane physics
replay and said: “this tripod walk is a great baseline btw, consider it approved”.

One canonical-reset five-second rollout travelled **1.53081 m** at **0.30616 m/s**
without reaching the height or tilt termination limits. This is a gait-design
baseline, not population validation, a trained PPO policy, or an earned STRIDE
checkpoint. The accepted PPO-100 remains the learned speed comparator.

Reference settings: 37 cm torso height, 18 cm foot stroke, 4 cm lift, 1.1 Hz
cycle, 65% stance, alternating tripods, and 0.8 s initial command ramp. The
actual torso height settled near 36.7 cm. Masses, contacts, joints, actuators,
and canonical model parameters were unchanged for this recording.

- `replay.zip`: exact model, states, metadata, trace, and source copies.
- `sources/`: controller, evaluator, simulation core and model used for the run.
- `metadata.json` and `trace.csv`: measured evidence and configuration.
- `manifest.json`: approval, evidence limits, and SHA-256 hashes.

The separate redesigned chassis is experimental and is not covered by this approval.
