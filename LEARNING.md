# Learn C-1N through notebooks

Start with [01_control_step.ipynb](notebooks/01_control_step.ipynb). The notebook is the main learning entry. Python modules hold reusable simulation code. CLI commands remain available for repeatable recordings and viewer work.

## Resume here

The refactor and notebook setup are verified. The first prediction exercise is
still unanswered. No user-written RL or PPO algorithm has been implemented.

From this checkout, resume with the prepared environment:

```powershell
.venv/Scripts/python -m jupyter lab notebooks/01_control_step.ipynb
```

Write which quantities change when one target is written, and which require a
physics step. Leave `RUN_EXPERIMENT = False` until your prediction is recorded.
Then inspect the result with your pair programmer. The next design discussion
is observations, actions, reward terms, episode boundaries, and action timing.

The test bench also contains an older Ant-v5 scaffold using Stable-Baselines3.
Preserve it as prior work. It does not replace your own RL/PPO implementation.

## Start the notebook

From the `spider` root, use Python 3.12 and the same environment for the kernel and CLI:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-learning.txt
.venv/Scripts/python -m jupyter lab
```

Select that environment's Python kernel. The first notebook works from the repo root or `notebooks/`. Its default Run All only prepares the exercise. Write `PREDICTION`, then set `RUN_EXPERIMENT = True` to inspect the result. Restart the kernel when you want a clean replay. Clear outputs and restore the gate before sharing an unanswered exercise.

## Learning standard and ownership

Success means you can trace a control step from the physical picture to math and code. You can predict a change, implement it, and diagnose the result with evidence. A passing test does not verify your understanding. A notebook execution does not close an experiment.

You own predictions, hypotheses, objectives, algorithm code, tradeoffs, and interpretation. The agent prepares imports, environment setup, plotting, and repetitive plumbing. Before conceptual help, the agent asks for your prediction or diagnosis. Review starts from your attempt. Use counterexamples and small independent implementation exercises to check transfer.

Start with gravity, ground contact forces, joint torques, and the actuator response. Introduce the required mathematics when the physical question needs it. Do not require a separate mechanics curriculum before RL.

## Code-to-math reading route

| Question | Source to inspect | Your evidence |
| --- | --- | --- |
| Which joint does one command address? | `model/spider.xml`: hinge joints and position actuators | Trace one named actuator to its joint, limits, and units. |
| What does reset establish? | `simulation.py`: `reset`, `neutral_targets` | Identify the initial state and commanded targets separately. |
| What happens before and after one step? | `simulation.py`: `set_targets`, `step`, `measured_state` | Predict, then compare raw time, target, position, velocity, and actuator force. Check sampling order. |
| How are support and foot errors measured? | `simulation.py` and its measurement helpers | Identify inputs, coordinate frames, units, and what each diagnostic cannot establish. |
| How does a controller choose targets? | `standing.py` and `runtime.py`: target selection and advance | Trace measurement to target without treating a diagnostic as a stability proof. |
| How is evidence recorded? | `telemetry.py` and the existing STAND diagnostics notebook | Relate each recorded sample to stepping and applied forces. |

The first notebook keeps both measurements and plots behind the prediction gate. The existing [STAND diagnostics notebook](notebooks/stand_rollout_diagnostics.ipynb) remains a trace reader. It is a later inspection tool, not the first lesson.

## Route to your PPO implementation

1. **Trace a simulation step.** Complete the first prediction exercise. Try a second joint or offset independently. Explain the mismatch between your prediction and the evidence.
2. **Define the learning problem together.** Choose observations, named reward components, failure conditions, and action timing. Begin with joint-target offsets around the neutral stance through existing actuators. Record your reasons before running training.
3. **Write RL.** You implement policy sampling, rollout storage, episode boundaries, returns, and a basic policy-gradient update. Use small numerical examples to check behavior. The agent must not supply completed algorithm scaffolding.
4. **Write PPO.** You implement the actor and critic, stored action log probabilities, advantages, probability ratios, clipped objective, value loss, entropy term, and minibatch updates. Use PyTorch operations, autograd, and optimizers. Do not replace your implementation with a ready-made trainer.
5. **Compare treatments.** Compare fixed neutral targets, the existing stance controller, and PPO under shared conditions. Preserve reward components, configurations, seeds, checkpoints, model parameters, and evaluation scenarios. Evaluate multiple runs. Report evaluation metrics separately from training reward. Change the action or control treatment after you understand the first comparison.

`learning_env.LearningSimulation()` is a separate physics adapter. `reset()` returns `MeasuredState`. `step(target_offsets_rad, *, physics_steps)` clips neutral-plus-offset targets to actuator limits and returns `MeasuredState`. The caller must supply the number of physics steps. Reward, termination, and policy timing remain outside the physics core. This is not a Gym interface or a training-ready environment. Finalize those choices during the ramp-up. Install PyTorch in that phase and record its version and device with the run.

## Evidence and open understanding

The README records the existing STAND implementation checkpoint and its limits. Those records do not establish your present understanding. The [robotics-test-bench learning route](https://github.com/haidmoham/robotics-test-bench) must keep implementation evidence separate from open learning work. Preserve historical experiments and interaction records. Do not infer closure from this cleanup.

GitHub issue #25 previously routed toward established RL tooling and away from a scratch RL implementation. The explicit user instruction on 2026-09-04 supersedes that rule for this learning work: write RL and PPO with PyTorch. The bench queue now selects that route. This does not close historical issues or certify new robot capability.

The initial refactor is complete when behavior is verified, this reading route is usable, and the first prediction exercise is ready. Learning and PPO completion remain collaborative work. Record a suspected behavior bug separately with a reproduction before changing physics or controller behavior.
