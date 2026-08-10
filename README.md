# C-1N

C-1N is a six-legged MuJoCo robot simulation for locomotion, evaluation, and simulation experiments.

Checkpoint lineage:

- `C-1N v0.0 - SPAWN`: deterministic six-foot spawn baseline.
- `C-1N v0.1 - SHUFFLE`: current live open-loop gait attempt.

The current implementation includes:

- twelve actuated hinge joints across six legs;
- the `v0.0 - SPAWN` static baseline;
- the `v0.1 - SHUFFLE` phase-shifted tripod gait;
- torso-orientation and foot-contact telemetry;
- interactive and headless simulation modes.

The current gait is intentionally limited and does not yet produce sustained walking.

## Run

Install MuJoCo:

```bash
pip install mujoco
```

Run the static pose baseline:

```bash
python simulate.py
```

Run the interactive viewer:

```bash
python view.py
```

Run a headless locomotion check:

```bash
python walk.py --headless --duration 20
```
