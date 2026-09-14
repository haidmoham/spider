# Two-link coupling

Question: How does moving one joint change the other joint's behavior in a planar 2-DOF arm?

## Initial prediction — make a prediction first

Before running anything, write down:

- If joint 2 is passive and joint 1 tracks a target, which way will joint 2 initially move?
- How will that prediction change between `elbow-down` and `elbow-up` configurations?
- Which difference do you expect when gravity is off?

Discuss your own first attempt with your pair programmer before advancing the experiment. Ask for explanations when you need them.

## Smallest useful experiment

`two_link_coupling.py` creates a two-link arm in the vertical x-z plane. Joint 1 follows a sinusoidal target. Joint 2 can stay passive, hold its initial angle, or follow a phase-shifted sinusoid for continuous waving. The terminal reports a compact state sample every 0.25 seconds:

The shoulder base is at `z=2.5 m`, leaving the elbow-down arm above the floor while you inspect joint coordination. The arm is rotated `90°` within its motion plane (about the hinge axis), and a small sphere marks the link tip as a hand proxy. In `wave` mode, the target is staged as ready → wave → smooth return → hold. The current wave frequency, `-1.2` to `+1.2 rad` joint-2 target range, and tracking gains are exposed as constants near the top of the script for speed tuning.

- `qpos`: arm configuration
- `qvel`: joint velocities
- `ctrl`: motor commands
- `qacc`: resulting joint accelerations

Start with the passive elbow and change only one condition per run.

```powershell
python -m pip install mujoco
python two_link_coupling.py --joint2-mode passive --configuration elbow-down
```

To run the coordinated wave controller:

```powershell
python two_link_coupling.py --joint2-mode wave --configuration elbow-down
```

In `wave` mode, the joint targets are control-coupled: joint 2's target includes `control_coupling * (joint1_target - joint1_start)`. Set `--control-coupling 0` to remove that cross-feed, or increase it to make joint 2 follow more of joint 1's desired motion.

For one-variable comparisons, repeat with one change at a time:

```powershell
python two_link_coupling.py --joint2-mode passive --configuration elbow-up
python two_link_coupling.py --joint2-mode passive --configuration elbow-down --gravity off
python two_link_coupling.py --joint2-mode hold --configuration elbow-down
```

## What to record

Record an observation before explaining it:

- Did joint 2 move or accelerate while only joint 1 was commanded?
- Did the motion depend on the starting pose?
- Did disabling gravity remove all of the effect, or only part of it?
- Was joint 2's behavior predicted by treating the arm as two independent pendulums?

Update `agent-log.md` after each meaningful run with the observation and your interpretation.

## Boundaries

This is deliberately simple, direct joint-space PD control. Do not add computed torque yet. Let the observations decide whether model-based control is the next question.

Related: #2
