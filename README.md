# C-1N

C-1N is a longitudinal six-legged MuJoCo robot project. The name is rooted in *Cien Fleur: Spider Net*: a Straw Hat reference hidden inside a machine-like model designation.

C-1N starts intentionally simple, and it may be bad at first. The goal is to integrate robotics concepts as they become understood in [`robotics-test-bench`](https://github.com/haidmoham/robotics-test-bench), then use the robot as a persistent simulation subject for learned locomotion, statistical evaluation, simulator calibration, uncertainty, and increasingly sophisticated simulation experiments.

Meaningful checkpoints preserve either a new robotics capability or an understood failure.

## Checkpoint grammar

Public checkpoints use:

```text
C-1N // NN · CODENAME
```

Current and planned lineage:

```text
C-1N // 00 · POSE     motor-assisted static-pose baseline
C-1N // 01 · SHUFFLE  current coordinated gait failure
C-1N // 02 · STAND    reserved for the first support-aware stable stance
C-1N // 03 · STRIDE   reserved for the first materially better sustained walk
```

Only `// 01 · SHUFFLE` is current. `POSE` is historical. `STAND` and `STRIDE` are reserved future boundaries.

`POSE` does not claim that C-1N learned to stand. It records only that the simulator can initialize a static pose and that the position actuators can hold that pose under baseline conditions.

`STAND` requires understood support and stable equilibrium over a reproducible rollout. `STRIDE` requires materially better sustained locomotion under a fixed evaluation protocol rather than one attractive run.

## C-1N // 01 · SHUFFLE

The current implementation is a deliberately limited baseline:

- a six-legged MuJoCo body with two hinge joints per leg;
- independent joint-position actuators;
- a motor-assisted static pose baseline;
- a low-clearance, phase-shifted tripod shuffle gait;
- a headless stepping script and interactive viewer with rolling telemetry.

The gait shares one phase across all six legs. It reads torso orientation and foot-ground contacts and uses those signals to bias joint targets. It does not produce sustained walking. That failure is preserved as part of the project.

Run the static pose baseline with:

```bash
pip install mujoco
python simulate.py
```

Run the walking viewer with:

```bash
python view.py
```

For a headless walking check:

```bash
python walk.py --headless --duration 20
```

## Current route

The immediate route is robotics-test-bench issue #24: **support state — make standing measurable**.

The bench first isolates:

`contact geometry + center-of-mass projection -> support load -> net body force/moment -> body attitude`

Then the same measurement model returns to C-1N with gait disabled. If C-1N demonstrates support-aware stable equilibrium under a fixed rollout and explicit success criteria, preserve that boundary as:

```text
C-1N // 02 · STAND
```

Standing is intentionally preserved as a required physical-intuition milestone. It gives later locomotion objectives a meaningful baseline.

After `STAND`, move directly into learned locomotion. The first policy may be ugly or exploit the objective. Preserve those failures and use them to choose the next simulation experiment.

The active simulation lanes in the test bench are:

- learned locomotion and objective design;
- statistical evaluation across seeds and scenarios;
- system identification and simulator calibration;
- uncertainty and distribution shift;
- differentiable dynamics;
- reproducible and scalable simulation infrastructure.

Controls, contact mechanics, actuator limits, estimation, and numerical methods remain available as supporting concepts when a concrete simulation failure demands them. Hardware is not a required destination for this project.

The browser artifact should preserve historical checkpoints so behavior, objective changes, and the reasoning that produced them can be compared over time.
