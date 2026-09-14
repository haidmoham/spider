# C-1N front-left leg workspace

## Question

Can the current two-joint front-left C-1N leg reach the body-frame support
target `[+0.30, +0.32, -0.30] m` without adding morphology?

## Initial prediction

The human predicts the target is reachable. The uncertain part is the current
joint axes and their orientation, rather than leg length alone.

## Method

`c1n_leg_workspace.py` is the executable fixture. It owns C-1N model loading,
the target, and torso-frame foot measurement. The notebook fixes the torso,
disables gravity, and samples only the front-left hip and knee through their
declared joint limits. It records each sampled foot position in the torso frame
and compares the nearest sample with the target. No contact, torque, or
controller is involved.

Use [`c1n_leg_workspace.ipynb`](c1n_leg_workspace.ipynb) as the analysis
companion. It shares the fixture's stem and does not replace it.
