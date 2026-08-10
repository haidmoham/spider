# C-1N

C-1N is a six-legged MuJoCo robot simulation for locomotion, evaluation, and simulation experiments.

The current implementation includes:

- twelve actuated hinge joints across six legs;
- a static pose baseline;
- a phase-shifted tripod shuffle gait;
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
