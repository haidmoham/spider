# Learn C-1N through notebooks

## Resume

2026-09-15: the user ran the 50/100-update comparison and accepted it as a crude
learned baseline. [PPO-100-CRUDE-20260915](artifacts/ppo-crude-baseline-20260915/README.md)
owns the preserved weights, settings, measurements, and viewed replays. Notebook 04
is its executed working surface. STRIDE remains unearned.
Notebook 04 is now frozen as the baseline experiment record. The user selected
platform integration next: [PPO runtime and deliberate-stalk comparison](docs/ppo-platform.md)
in Python under `spider/`, with no notebook/lab runtime dependency and no physics
parameter changes. The prepared control treatments are unvalidated. Execution
requires the user's explicit request; preparation did not run new rollouts.

### Prior 30-update work (historical)

Production training and tuning is active at the user's request (2026-09-15).
The user implemented GAE, PPO losses and minibatch updates. Their analytic and
update checks pass. If a PPO implementation defect becomes the next task, switch
back to practice with the user. Do not infer mastery from agent-run experiments.

Notebook 04 is the visible working surface. Put each proposed training/tuning
change there before execution and return outputs to the same notebook. Preserve
its open editor buffer. External outputs do not restore IDE kernel variables.

Completed: fresh PPO through 10 updates, then an unchanged continuation through
30 with actor, critic and optimizer states restored. Section 9 contains that
continuation and its evaluation. The 10-update outputs remain in sections 7–8.
See [the run receipt](lab/notes/2026-09-15-ppo-30.md) for artifacts and measurements.

Sampled forward travel improved from 60.27 to 162.88 mm over 5 seconds on the
same 12 evaluation seeds. Mean-action travel is only 5.65 mm at update 30.
STRIDE remains unearned; target 48/50. No reward, observation, action-interface
or PPO hyperparameter tuning has run yet. Next: inspect the 30-update replay,
then declare the next bounded training or tuning comparison in notebook 04.

The preserved REINFORCE archive is
[here](lab/history/notebooks/04_reinforce_before_ppo_20260914.ipynb), SHA256
`850be75c6642fc9ab89a5550294e425cfa6efe12113d6977fc8da95fcaa1ac12`.
The reference appendix is historical: do not execute it during the PPO path.

### Earlier route and evidence (preserved context)

Open [04_mdp_contract.ipynb](lab/notebooks/04_mdp_contract.ipynb) for
[issue #33](https://github.com/haidmoham/spider/issues/33). It is a prepared scaffold
that continues notebook 03's final fixed-shuffle versus feedback-policy comparison.
The active section is **REINFORCE robot debugging lab**. It carries the existing gait
reference and provides deliberately flawed learning code at the user's request.
The user fixes observation/sampling/reward/credit assignment and training behavior;
the agent owns collection, logs, artifacts, and replay plumbing. No bug count or
answer key is supplied. No robot training ran during preparation.
Initialize a fresh run, inspect a bounded training block, then evaluate and replay
paired policies and controls. Use the later MDP sections to document design choices.
Evaluation uses one synchronized four-pane replay: before, after, neutral, and
fixed shuffle. Space pauses; R restarts; arrows and +/- change the shared camera.
Keep [03_coordinated_baseline.ipynb](lab/notebooks/03_coordinated_baseline.ipynb)
as the prior REINFORCE and gait reference.
Next: experiment 1 in [TODO.md](TODO.md), **Establish the task and a predictable
action interface**. This is a proposed five-experiment route toward visible stride
and explicit PPO. The user owns implementation and interpretation; preparing the
route does not authorize running its experiments. Do not restart the completed
single-sample/REINFORCE work as a prerequisite.

The notebook now preserves one executed REINFORCE update, a corrected joint-to-foot
finite-difference probe, a geometry-informed tripod shuffle, and a corrected
gait-parameter feedback policy. The latest five-second comparison moves the fixed
shuffle forward about 0.1406 m; feedback moves backward about 0.0380 m and sideways
about 0.1507 m. Preserve that failure. The logged sweep reaches both bounds;
the cause and a useful next change remain questions for the user to investigate.
The REINFORCE cells remain the direct-joint reference: aligned transitions,
explicit returns, one update, and three paired evaluation seeds with a zero control.
The saved single-sample gradients establish autograd plumbing; that check used
a reward from an older recording and did not establish action–reward alignment.
Saved outputs report finite nonzero gradients and a weight change. These establish
an optimizer update, not a learned locomotion capability. PPO follows the explicit
advantage and probability-ratio steps in TODO.md; a hand-tuned walking solution is
not a prerequisite for implementing those steps.
The first update retains `next_x + 0.1 * next_z` to isolate the training change;
this is a position-and-height reward, not a validated locomotion objective.
The earlier untrained random policy and saved outputs remain in the notebook.
See the [dated run and demo receipt](lab/notes/2026-09-14-policy-and-shuffle.md)
for local artifacts, exact export settings, and the current evidence limits.
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
.venv/Scripts/python -m pip install --upgrade -r requirements-bootstrap.txt
.venv/Scripts/python -m pip install -r requirements-learning.txt
.venv/Scripts/python -m ipykernel install --user --name c1n-pairing --display-name "C-1N pairing (.venv Python 3.12)"
.venv/Scripts/python -m jupyter lab lab/notebooks/04_mdp_contract.ipynb
```

In VS Code, select the **C-1N pairing** kernel. The notebook finds the repository
from its root or `lab/notebooks/`. Setup imports helpers and resets the robot.
Run All also calls your policy and executes the inspection cells.

The legacy Ant notebook has its own environment:

```powershell
py -3.12 -m venv .venv-ant
.venv-ant/Scripts/python -m pip install --upgrade -r requirements-bootstrap.txt
.venv-ant/Scripts/python -m pip install -r requirements-ant.txt
.venv-ant/Scripts/python -m ipykernel install --user --name c1n-ant --display-name "C-1N Ant (.venv-ant Python 3.12)"
```

Use **C-1N Ant** only for `ant_policy_learning.ipynb`. The shared parent `../.venv`
remains a legacy environment for other notebooks; neither active manifest installs there.
Run each environment's `python -m pip check` after installation. Direct dependencies
are pinned; these files are not a complete transitive lock. Saved run receipts retain
the model and training settings used by an experiment. Restart a kernel to load upgraded packages.

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
