# Pendulum Control

Question: How does feedback change the behavior of a one-joint pendulum?

Prediction:
- P control should push based on position error and tend to overshoot or oscillate.
- PD control should use velocity feedback to damp motion and settle more cleanly.

Smallest test:
- One rigid link.
- One hinge joint.
- Gravity.
- One motor.
- Direct access to `qpos`, `qvel`, and `ctrl`.

Result:
- Initial MuJoCo perturbations behaved as expected.
- The P vs. PD predictions matched the observed experiment.

Interpretation:
- P answers "how far from target?"
- D adds "how fast am I moving?"
- Simulation failures can come from the model, actuator, controller, contact, or integrator.

Next question:
- What do obviously too-high `Kp` and too-high `Kd` look like?
- After that, move to a two-link planar arm instead of polishing this controller.
