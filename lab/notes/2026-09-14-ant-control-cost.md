# Ant prior and C-1N control-cost diagnostic

## Question and boundary

User hypothesis: does the learned "little box step" exploit unpenalized actuator
movement? Measure first. No coefficient, reward change, observation change, new
training, gait prior, or checkpoint advance is part of this experiment.

Reference: **Gymnasium v1.3.0, Ant-v5**, not an unspecified Ant version.
[Reward implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/ant_v5.py#L291-L355),
[motor XML](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/assets/ant.xml#L70-L79),
[MuJoCo actuator model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).
Reward is a numerical utility, not a conserved physical quantity. Coefficients
convert differently scaled measurements into score units; do not copy their values.

## Reconciliation

Ant defaults below are source facts. Intended behavior is the term's design intent,
not a guarantee. C-1N transfer judgments are hypotheses supported by its interface.

| Term | Exact Ant definition; default | Input units / scale | Intended behavior | Closest C-1N quantity | Existing failure evidence | Semantic transfer |
| --- | --- | --- | --- | --- | --- | --- |
| Forward | `w_f * (x_after-x_before)/dt`; `w_f=1`, `dt=0.05 s` | m/s; weight has score/(m/s) | Forward translation | `dx_m / 0.02 s` | Small, variable progress; fixed shuffle travels farther | Same concept. Current reward uses endpoint `vx_after`, not interval displacement/dt. Preserve it during this diagnostic. |
| Healthy | `1 * I(finite state and 0.2 <= z <= 1.0)`; terminate when unhealthy by default | Boolean; 1 score/action | Remain in admissible state | Finite-state check and torso-height boundary | No falls in this suite; neutral already stands | Health criteria must fit C-1N. Current `0.1*z` is graded height reward, not Ant's health indicator. No evidence here requiring a new survival term. |
| Control | `0.5 * sum(a_i**2)` subtracted | Eight bounded motor commands, each in [-1,1]; cost 0..4 | Discourage large commands | Squared offset command for an interface-level analogy; measured torque squared for an effort analogy | Box-step observation; quantities below, no established cause | Neither target offsets nor absolute targets preserve Ant's torque relationship. See distinctions below. |
| Contact | `5e-4 * sum(clip(cfrc_ext,-1,1)**2)` subtracted | Body spatial wrenches: force N and moment Nm, numerically clipped and mixed | Discourage large external wrenches | MuJoCo `cfrc_ext` / individual contact forces | Contact counts vary; no verified excessive-impact/slip diagnosis | Count is not force. This is not a slip cost or pure force norm. Body count, units and support loads differ; do not copy scale. |

Ant-v5 includes contact cost independently of whether contact forces enter the
observation. Its health reward is conditional on health. Earlier versions differ.
These formulas are per action, not work integrals. Changing action duration changes
episode score scale unless the objective accounts for it.

## What does "control" mean here?

Gravity and ground contact load C-1N's joints even while it stands. A position servo
can exert torque while its target does not change. In this frozen model each actuator
is a direct unit-gear hinge position servo with `kp=300`: torque follows target-minus-
measured-angle error. Ant instead has direct motors with gear 150, so within command
limits its joint torque is `150*a`. Thus its command-square cost is proportional to
torque squared for that fixed model. It is not a measurement of energy consumption.

| Candidate | Equation | Units | Meaning / transfer limit |
| --- | --- | --- | --- |
| Neutral-relative target size | `sum(offset_i**2)` | rad² | Penalizes distance of the command from neutral. Closest analogy to penalizing the policy's bounded output. Not movement, tracking error, or torque. The unbounded Gaussian latent is not the actuator command. |
| Absolute target size | `sum(target_i**2)` | rad² | Penalizes distance from the joint coordinate origin. Neutral already costs 7.5. Contains posture-dependent cross terms; not equivalent to offset-square plus a constant. |
| Actuator torque | `sum(tau_i**2)` | (Nm)² | Mechanical effort proxy; closest to Ant's physical effect, but not the same interface term. Requires the realized state and servo model. Standing can incur it. |
| Signed power | `sum(tau_i*qdot_i)` | W | Instantaneous net mechanical power at joints. Positive and negative joint contributions cancel. Not electrical input power. |
| Absolute power | `sum(abs(tau_i*qdot_i))` | W | Counts mechanical exchange magnitude without cancellation; not battery consumption. Work would integrate power over time in joules. |
| Action change | `sum((offset_t-offset_prev)**2)` | rad²/action difference | Penalizes command changes; distinct from size. First difference uses zero offset. Dividing by dt² would make a different rate measure. |

Do not label all these quantities "energy cost". A coefficient would have different
meaning for each. No normalization or coefficient has been chosen here.

## Measurement and preservation

Source: `telemetry/reinforce-debug/20260915T010216441278Z/evaluation-00020-011012369488/`.
Output: `telemetry/control-cost/20260915T012137933382Z/`.
The diagnostic reads all 28 saved episodes, with initial/trained sampled policies
paired on 12 action seeds; mean policies and controls run once in the source suite.
It restores saved model/state pairs and calls `mj_forward`, never `mj_step`.
All source episode file hashes matched before and after. Baseline weights, learner,
model, observations and rewards are unchanged. Outputs go to a new directory.

`samples.csv` contains the full distribution; `episodes.csv` contains per-episode
mean, median, p95, maximum and sample sums; `summary.csv` compares episode means and
their ranges. `distributions.png` plots cumulative distributions. A point at y=0.95
means 95% of saved endpoints are at or below that x value. Correlated time samples
are not independent trials. `sum_samples` is not an integral or a discounted return.

Targets and action differences are exact recorded commands. Torque/power are
recomputed instantaneous **post-action endpoint** values, every 20 ms. The original
physics timestep is 2 ms: endpoint distributions can miss switching transients and
impact peaks. No work integral is reported. Signed endpoint power averages cannot
establish net energy consumption or regeneration over the rollout.

## Results: episode-mean quantities, averaged across seeds where applicable

| Treatment | Offset² (rad²) | Absolute target² (rad²) | Action change² (rad²) | Endpoint torque² ((Nm)²) | Endpoint absolute power (W) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 0 | 7.50000 | 0 | 38.80 | 0.257 |
| Initial sampled | 0.004499 | 7.50283 | 0.008813 | 510.73 | 78.51 |
| Trained sampled | 0.004484 | 7.49721 | 0.008676 | 512.54 | 78.08 |
| Initial mean | 0.000690 | 7.49909 | 0.000009 | 49.68 | 0.372 |
| Trained mean | 0.000706 | 7.49167 | 0.000008 | 49.43 | 0.347 |
| Fixed shuffle | 0.017100 | 7.57710 | 0.002845 | 488.88 | 46.06 |

Trained sampled offset-square episode means span 0.004359-0.004593; initialization
spans 0.004364-0.004609. Their distributions nearly overlap. Mean per-episode p95
action-change-square is 0.013924 trained vs 0.014099 initial; fixed shuffle p95 is
0.037065. The shuffle changes less often but has larger changes at transitions.

The current evidence **does not identify an acquired large-command exploit**:
initial and trained sampled policies have similar magnitudes, action changes and
endpoint effort. Both differ strongly from their mean-action evaluations.
This does not rule out a useful regularizer or identify the cause of the box step.

A plain offset-square penalty would charge the fixed shuffle about 3.8 times as
much per action as the trained sampled policy, despite the shuffle's much greater
forward travel. It suppresses command amplitude, not specifically the unwanted
behavior. Torque, absolute power and action-change measures rank treatments
differently. Do not select a term just because it is called control cost.

This is observational evidence, not a causal ablation. It supports inspecting the
sampling-related motion and command changes; it does not establish that adding a
particular penalty will improve locomotion. No coefficient or retraining follows.

## Next user decision

Which behavior do you want to discourage: large deviations from neutral, large
realized torques, or frequent target changes? Name it and the measurement that
distinguishes it from the useful fixed shuffle before selecting a coefficient.

Reproduce the diagnostic from the repo root:

```powershell
.venv/Scripts/python -m lab.control_cost_diagnostic telemetry/reinforce-debug/20260915T010216441278Z/evaluation-00020-011012369488
```
