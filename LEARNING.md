# Learn C-1N through notebooks

## Resume

Open [02_first_policy.ipynb](notebooks/02_first_policy.ipynb).
Run the setup cell, then edit the cell headed **Edit here: `policy(observation)`**.
Your current draft is preserved. Its last fresh-kernel check failed because it
indexes `MeasuredState` as a sequence. No policy rollout or RL update ran.

The next task is to read a named measurement, define its desired value separately,
and return 18 finite offsets. Choose the responding actuator and use it in both
comparison branches. The notebook supplies recording, plots, and exact-state replay.
Review your function before interpreting a rollout.

## Environment

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-learning.txt
.venv/Scripts/python -m ipykernel install --user --name c1n-pairing --display-name "C-1N pairing (.venv Python 3.12)"
.venv/Scripts/python -m jupyter lab notebooks/02_first_policy.ipynb
```

In VS Code, select the **C-1N pairing** kernel. The notebook finds the repository
from its root or `notebooks/`. Setup imports helpers and resets the robot.
Run All also calls your policy and executes the inspection cells.

## Read only the path you need

1. `c1n/learning.py`: `LearningSimulation.reset()` returns `MeasuredState`.
   `step(offsets, physics_steps=...)` applies neutral-plus-offset targets, clips
   them to actuator limits, and returns the post-step measurement.
2. `c1n/simulation.py`: owns the reset, measurements, and physics step.
   Targets are commands in radians. Measured angles describe the resulting state.
3. `c1n/controllers.py`: existing comparison treatments, not your policy implementation.
4. `c1n/recording.py` and `c1n/viewing/`: inspect these only when recording or display matters.

## Ownership and route

You own hypotheses, observation and reward design, policy code, action timing,
episode boundaries, and interpretation. The agent owns setup, repetitive plumbing,
and verification. Setup success does not demonstrate learning.

Write RL and PPO with PyTorch operations, autograd, and optimizers. The older
Ant/SB3 scaffold in the test bench remains prior work; it does not replace this route.
Proceed from a callable policy to recorded rollouts, then define the learning problem,
write a policy-gradient update, and implement PPO. Pull in mechanics and mathematics
when a concrete question needs them.

Keep a neutral-target control and compare fixed scenarios. Preserve reward terms,
seeds, model settings, checkpoints, and evaluation conditions when training begins.
The current short inspection defaults are editable viewing settings, not a reward
or episode specification. Preserve your attempts and measured evidence in the notebook.

STAND is earned as recorded; disturbance recovery is not a prerequisite. The open
support-understanding question remains separate from the robot's checkpoint.
Your earlier observation that absolute velocity can reward shuffling without net
travel remains useful when we define the objective. It does not establish a rollout result.
