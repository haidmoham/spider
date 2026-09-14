# Learn C-1N through notebooks

## Resume

Open [03_coordinated_baseline.ipynb](lab/notebooks/03_coordinated_baseline.ipynb).
Continue at **One rollout, one REINFORCE update**. The visible multi-step loop is
prepared, not executed. Its cells collect aligned transitions, compute returns,
apply one update, and compare three paired evaluation seeds with a zero control.
The saved single-sample gradients establish autograd plumbing; that check used
a reward from an older recording and did not establish action–reward alignment.
Run the new cells in order when ready, then inspect the exact-state recordings
and reward decomposition. PPO remains a separate step after that review.
The first update retains `next_x + 0.1 * next_z` to isolate the training change;
this is a position-and-height reward, not a validated locomotion objective.
The earlier untrained random policy and saved outputs remain in the notebook.
The fixed coordinated baseline remains a non-executing reference in the notebook.
Its earlier recorded comparison is preserved below.
The eight-second coordinated trial repeated exactly, had valid timestamps and
no clipped targets, and reached at most 0.218 degrees of body tilt. Sampled
contact count ranged from three to six. This is a small motion reference, not
a learned gait or disturbance-recovery result. The neutral control remains fixed.
Run artifacts are in `telemetry/fixed-coordinated-baseline/`.

The earlier [02_first_policy.ipynb](lab/notebooks/02_first_policy.ipynb) retains
the following comparison and its failure:
The current comparison applies the front-left commands alone, copies them to the
front-right leg, or reverses the copied offsets. The user authorized implementation
and execution of these three two-second trials. No RL update ran.

The same-sign trial became numerically unstable at 1.982 s and reset its clock.
Do not treat its final displacement as a valid two-second result. The notebook
labels this failure and preserves the plot and exact-state replay. The other
two trials reached two seconds; this does not establish stable locomotion.
The preceding user draft and run artifacts remain in
`telemetry/paired-leg-comparison/`. Review the coordination and target-offset
interface before deciding how to initialize a learned policy.

## Environment

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-learning.txt
.venv/Scripts/python -m ipykernel install --user --name c1n-pairing --display-name "C-1N pairing (.venv Python 3.12)"
.venv/Scripts/python -m jupyter lab lab/notebooks/02_first_policy.ipynb
```

In VS Code, select the **C-1N pairing** kernel. The notebook finds the repository
from its root or `lab/notebooks/`. Setup imports helpers and resets the robot.
Run All also calls your policy and executes the inspection cells.

## Read only the path you need

1. `spider/learning.py`: `LearningSimulation.reset()` returns `MeasuredState`.
   `step(offsets, physics_steps=...)` applies neutral-plus-offset targets, clips
   them to actuator limits, and returns the post-step measurement.
2. `spider/simulation.py`: owns the reset, measurements, and physics step.
   Targets are commands in radians. Measured angles describe the resulting state.
3. `spider/controllers.py`: existing comparison treatments, not your policy implementation.
4. `spider/recording.py` and `spider/viewing/`: inspect these only when recording or display matters.

## Ownership and route

You own hypotheses, observation and reward design, policy code, action timing,
episode boundaries, and interpretation. The agent owns setup, repetitive plumbing,
and verification. Setup success does not demonstrate learning.

Write RL and PPO with PyTorch operations, autograd, and optimizers. The older
Ant/SB3 scaffold in `lab/notebooks/ant_policy_learning.ipynb` remains prior work;
it does not replace this route. Use `requirements-ant.txt` in a separate
environment if revisiting it. Do not install that trainer in the C-1N environment.
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
