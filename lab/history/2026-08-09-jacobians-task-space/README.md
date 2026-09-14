# Jacobians and task space

Question: How does the same first-joint motion map into shared-frame foot X motion when only the base orientation changes?

## Initial prediction

Holding geometry, joint pose, perturbation size, controller, and timestep fixed, the same small positive first-joint perturbation should produce a large shared-X foot displacement at 0 degrees, near-zero X displacement at 90 degrees, and an equally large displacement with the opposite X sign at 180 degrees. The foot should not stop moving at 90 degrees; its motion should be along shared Z instead. At intermediate orientations, the shared-X magnitude should scale with the cosine of the base orientation: 45 and 135 degrees should have about 71% of the 0-degree magnitude, with opposite signs.

## Smallest useful experiment

`jacobians_task_space.py` is a one-link leg in the X-Z plane. It holds the model, initial joint pose, PD controller, and timestep fixed while changing only the base pitch.

Before the motion starts, the script prints two estimates of the first Jacobian column's X entry:

- MuJoCo's position Jacobian, `Jx`;
- a centered finite difference of foot X position with respect to joint 1.

During the run, it reports joint position, velocity, and acceleration beside shared-frame foot X position, X velocity, and X acceleration. X velocity is `Jx * qvel`; X acceleration is the measured change in that velocity, so it includes the changing Jacobian as the leg moves.

Run a single base orientation at a time:

```bash
python jacobians_task_space.py --base-degrees 0
python jacobians_task_space.py --base-degrees 90
python jacobians_task_space.py --base-degrees 180
```

To watch those three physical runs together, use the overlay viewer borrowed from the Experiment 3 comparison pattern:

```bash
mjpython jacobians_task_space.py --overlay
```

Blue is 0 degrees, orange is 90 degrees, and green is 180 degrees. Each color is an independently stepped, zero-gravity simulation, so the controller produces the same realized joint motion in every color and the visual difference isolates the frame transformation. The overlay affects rendering only. The single-orientation runs above retain normal gravity.

The overlay camera faces the X-Z motion plane directly: screen-left/right is shared X and screen-up/down is shared Z. Read the three base angles in that visible plane, not from a perspective camera.

The colored links are not three legs on one robot. They are three independent copies of the same one-link model, drawn around one shared hinge for comparison. Foot labels identify each base orientation; black markers label the shared hinge, +X, and +Z.

The left graph stack compares the shared joint position, velocity, and acceleration against the sinusoidal target. Because the zero-gravity runs have identical joint dynamics, one actual trace represents all three base orientations without hiding it under duplicate lines. The right stack compares foot X displacement from each foot's starting position, X velocity, and X acceleration. Fixed vertical ranges make sign, magnitude, and phase comparisons stable instead of rescaling each frame. Stack width, height, margins, and gaps respond to the current viewer viewport, so resizing preserves plotting area without overlapping the two stacks. Both stacks use the test bench's canonical viewer helper in [`experiments/telemetry.py`](../telemetry.py), adapted from C-1N.

For a quick non-visual comparison:

```bash
python jacobians_task_space.py --base-degrees 45 --headless --duration 2
```

## Continuation: a full two-joint velocity map

`two_joint_planar_leg.py` stays inside this experiment. It does not add a controller, gait, or locomotion objective; it freezes a two-joint planar leg at two poses and gives both poses the identical small joint velocity:

```text
qdot = [ +0.70, -0.35 ] rad/s
```

Before running, predict which pose makes that same `qdot` move the foot more in world X, and whether either world-X or world-Z velocity can reverse sign. Then run:

```bash
python two_joint_planar_leg.py
```

For each pose, the terminal prints:

- `q`, the current joint configuration;
- `x(q)`, the foot position in the world X-Z plane;
- the 2-by-2 translational Jacobian `J(q)` from MuJoCo;
- `J(q) @ qdot`, the predicted foot velocity;
- MuJoCo's reported foot velocity, and their difference.

It then changes only `q`, keeps `qdot` fixed, and repeats the comparison. The displayed difference should be near numerical precision: this is a velocity-level kinematic check, not a locomotion simulation.

To inspect the two poses in the MuJoCo viewer, sweeping smoothly between them:

```bash
mjpython two_joint_planar_leg.py --viewer
```

At the foot, the live arrows make the Jacobian geometric: blue is the first column (a unit hip velocity), orange is the second (a unit knee velocity), and green is their actual weighted sum `J(q) @ qdot`. Their shared scale is fixed, so the change in arrow direction and size comes from the pose-dependent Jacobian rather than from auto-rescaling.

Learning target: **the Jacobian is the pose-dependent local map from joint-space motion to task-space motion.**

## Verification status

The recorded checks establish important parts of the #6 learning target:

- At base angles 0, 90, and 180 degrees, MuJoCo's initial `Jx` matched the centered finite-difference estimate: -1, approximately 0, and +1 respectively.
- For the two-joint leg, `J(q) @ qdot` matched MuJoCo's world X-Z foot velocity at numerical precision in both the bent and open poses.
- The same `qdot` produced different foot velocities because the pose changed `J(q)`, not because the joint command changed.

Issue #6 remains open. Its missing evidence is one planar two-DOF mechanism evaluated at 0, 90, and 180 degrees, with local/base and shared-frame position comparison plus static `Δx` and `Δy` probes. The current one-link orientation probe and two-joint velocity probe do not replace that combined check.

Full standing control remains out of scope. The tripod continuation only makes the force-to-torque bridge inspectable; feasible contact-force allocation belongs to #20.

## Portfolio visual brief

Preserve these visual teaching beats when this experiment is projected into a portfolio post:

1. **Pose-dependent motion.** Show the same joint velocity at two leg poses. Draw the two Jacobian columns as foot-velocity arrows and their weighted sum `J(q) @ qdot`. The viewer already provides this source view in `two_joint_planar_leg.py --viewer`.
2. **Shared-frame orientation comparison.** Overlay 0, 90, and 180-degree one-link copies in the X-Z plane. Make the 90-degree case visibly move in Z while its initial world-X component is near zero. The source view is `jacobians_task_space.py --overlay`.
3. **Full free-body diagram.** Separate the whole robot's external gravity and ground-reaction forces from one isolated leg's actuator torques. Use the checked-in PNG fallback when rich rendering is unavailable: [`assets/tripod-support-fbd.png`](assets/tripod-support-fbd.png).
4. **Force-to-torque interaction.** Let the reader change a two-link pose while holding a desired foot-force vector fixed. The visible result should be changing hip and knee torque demands from `tau = J(q).T @ f`; this is not a coordinate conversion, but a change in physical force leverage.

The visual sequence should establish geometry first, then use `J(q)` for motion and `J(q).T` for force-to-torque mapping. Do not imply that the Jacobian allocates contact forces or solves whole-body stability.

## Tripod-support visual

The reusable free-body diagram for the next #6 continuation lives in [`assets/tripod-support-fbd.svg`](assets/tripod-support-fbd.svg), with [`assets/tripod-support-fbd.png`](assets/tripod-support-fbd.png) as its chat-safe rendered fallback. It separates the whole robot's external forces—gravity and three ground reactions—from an isolated leg's actuator torques. Keep this source asset local to the experiment; it is the source for a later portfolio post. When a rich inline visualization fails to render, show the checked-in PNG rather than emitting a raw visualization directive again.

## Continuation: static tripod support

`tripod_static_support.py` is the first #6 application. A free body rests on three bent, two-joint legs with planted feet. Every control step has two deliberately separate torque terms:

```text
posture torque:  Kp * (q_ref - q) - Kd * qvel
support torque:  J_foot(q).T @ desired_foot_force
total torque:    posture torque + support torque
```

### Scope boundary

This continuation exists only to make the physical chain inspectable: joint posture change → foot task-space motion → pose-dependent foot Jacobian → motor torque. Its equal-foot-force rule is deliberately incomplete. A negative rear normal-force request would be physically impossible because the ground cannot pull on a foot, but the current checked forward run does not reproduce one. Do not add wrench allocation, friction constraints, contact-force feedback, or planted-foot control here; those belong to [#20 static support & equilibrium](../../integrations/c1n.md).

Gravity and the body position determine a three-foot vertical-force allocation: the force sum equals the model weight, while its X and Y moments balance the requested support point. Each leg then maps its own assigned foot force into two motor torques with its current foot Jacobian transpose.

Run the centered hold first:

```bash
python tripod_static_support.py --headless --duration 2
mjpython tripod_static_support.py
```

Then change only the desired center-of-mass X position:

```bash
python tripod_static_support.py --headless --duration 2 --body-shift 0.08
mjpython tripod_static_support.py --body-shift 0.08
```

For a visible combined perturbation, lower the desired center of mass while requesting forward motion:

```bash
mjpython tripod_static_support.py --squat-drop 0.12 --body-shift 0.60
```

The squat changes the shared two-joint posture reference to shorten every planted leg, physically lowering the rigid chassis box and its COM. Center-of-mass feedback then makes the desired COM-X error into a desired net horizontal force and the desired COM-Z error into a requested upward ground force. The allocator distributes those forces across the planted feet while its vertical-force moments follow the current COM projection; each leg maps its full assigned foot force through `-J(q).T @ f`. Prediction recorded before this run: the posture term keeps a useful bent leg shape; with the forward shift, the front foot's vertical support force rises and the rear two decrease. This is not a stepping controller yet. If a requested support point makes a foot's required normal force negative, a fixed-foot controller must move the body projection back inside the support triangle; a walking controller would instead take a step.

## What to record

- Do the Jacobian and finite-difference estimates agree in sign and rough magnitude?
- Does the 90-degree run still show joint and foot motion while shared X motion is near zero at the initial pose?
- Does reversing the base to 180 degrees reverse the initial X Jacobian sign without changing the joint perturbation?

Keep controller, geometry, joint start, and timestep unchanged while comparing orientations. Do not add inverse kinematics or a task-space controller yet.

Related: #6
