# Experiments and their robot connections

Start with the question you need. The current learning task remains the first
policy notebook; the mechanics experiments are references, not prerequisites.

| Notebook | Bounded question | Robot connection |
| --- | --- | --- |
| [First policy](02_first_policy.ipynb) | Can the current policy produce valid target offsets and an inspectable rollout? | `spider/learning.py`: action interface, recording and measurements |
| [Control step](01_control_step.ipynb) | What changes between a command and the measured next state? | `spider/simulation.py`: targets, stepping and measurement |
| [Feedback and dynamics](feedback_and_dynamics.ipynb) | What do damping, coupling and a dynamics model change? | `model/spider.xml`: position actuators; `spider/controllers.py`: existing stance treatment. The notebook torque controllers are not integrated. |
| [Jacobians](jacobians.ipynb) | How do frame and pose change the joint-to-foot velocity map? | `spider/simulation.py`: foot Jacobians and joint-space update direction |
| [Support boundary](static_support_boundary.ipynb) | When does a shifted payload unload a support contact? | `spider/simulation.py`: `measure_support` and `_support_margin` |
| [Leg workspace](c1n_leg_workspace.ipynb) | What target positions become reachable with another joint axis? | `model/spider.xml`: proximal hinges; `spider/simulation.py`: `foot_placement_diagnostics` |
| [STAND diagnostics](stand_rollout_diagnostics.ipynb) | What do saved standing traces show? | `spider/recording.py` and the recorded STAND evidence |
| [Legacy Ant scaffold](ant_policy_learning.ipynb) | How can an objective miss intended behavior? | Concept reference only; Ant/SB3 is not integrated. Use its separate environment. |

Reusable isolated fixtures live in [lab](../). Integrated
robot behavior lives in [spider](../../spider/). Original dated findings and stable
interaction records live in [test-bench history](../history/).
The [migration map](../history/migration.json) identifies every
source file and its original commit and hash. Historical queue documents do not
select current work; [LEARNING.md](../../LEARNING.md) does.

- [02_first_policy.ipynb](02_first_policy.ipynb): current policy session. Run setup, then edit the marked policy cell.
- [01_control_step.ipynb](01_control_step.ipynb): earlier control-step experiment. Preserve the user's treatment and outputs.
- [stand_rollout_diagnostics.ipynb](stand_rollout_diagnostics.ipynb): reads saved STAND traces.

See [LEARNING.md](../../LEARNING.md) for environment setup and the learning route.
Reusable helpers live in `spider/`; experiment choices and interpretation live here.
Run All can execute physics and open viewers. Preparation checks must not run
learning experiments without the user's instruction.

Executed notebooks and traces are evidence. The 2026-09-10 diagnostics archive is
`telemetry/local-archive/2026-09-10-stand/`; its manifest covers the saved notebook
and 34 traces. The pre-trim 2026-09-14 notebooks are in `telemetry/architecture-before/`.
Both archives are local and ignored by Git. Existing trace paths are preserved.
