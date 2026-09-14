# From shuffle to a legible PPO stride

Updated 2026-09-14. Proposed work packages, not completed experiments.
Start in [notebook 03](lab/notebooks/03_coordinated_baseline.ipynb).
[LEARNING.md](LEARNING.md) selects the active step; this file owns its planned
sequence. The archived test-bench queue does not select current work.

## Outcome and ownership

Produce repeatable forward motion that visibly consists of steps, while the user
can trace how observations, actions, rewards, and gradients produced it.
PPO is an optimizer for the chosen policy and objective, not a definition of walking.
A successful training update and an earned STRIDE capability are separate results.

The user owns hypotheses, observation/action/reward choices, learning code, and
interpretation. The agent supplies small stubs, deliberately flawed starter lines
when requested, diagnostics, simulator plumbing, and verification. Keep the math
in ordinary PyTorch tensors and loops. No trainer framework or trainer class.
An implementation request may delegate that part explicitly; record the distinction.
Use change → run → watch → inspect → revise. Review useful chunks, not every line.
Do not run a new learning experiment merely to prepare its notebook cells.

## Starting evidence and the actual gap

Already present: `LearningSimulation.reset/step`, 18 neutral-relative joint targets,
recorded rollouts, exact-state replay, a stochastic Normal policy, summed log-probs,
explicit reward-to-go, one REINFORCE update, and gradient/weight diagnostics.
The newer feedback policy chooses sweep while lift and period stay fixed.

The saved five-second control moves +0.140608 m forward and +0.003545 m sideways.
Feedback moves -0.037958 m forward and +0.150726 m sideways. Its sweep reaches
both bounds. These are observations; a bad feedback sign is no longer an established
explanation after the user's corrections. See the [run receipt](lab/notes/2026-09-14-policy-and-shuffle.md).

The minimal missing pieces are contracts and tensors, not software layers:

| Piece | Smallest next addition |
| --- | --- |
| Task and observation | Named units, command/frame convention, gait phase, episode ending rules |
| Action | Bounded stochastic adjustment to fixed-gait parameters; retain the latent sample |
| Rollout storage | Detached old log-probs/values, next observation, terminal and cutoff flags |
| Credit assignment | A value estimate, advantages, and correct boundary bootstrapping |
| PPO objective | Recomputed log-probs, old/new probability ratio, clipped actor objective |
| Training schedule | Explicit shuffled minibatches and a few epochs per fresh rollout |
| Evidence | Fixed baselines, paired evaluation conditions, ordinary-speed replay, saved settings |

The 18-offset simulator interface can stay unchanged. The proposed initial actor
controls one sweep adjustment; lift stays fixed until the one-output path is
understood. A second lift output is a later comparison. The earlier 18-output
REINFORCE policy remains a reference. Hand-tuning perfect walking is not a gate.

## Five notebook experiments

### 1. Establish the task and a predictable action interface

- [ ] Inspect why the corrected feedback differs from the fixed shuffle.

**Question:** does changing sweep every 20 ms help, or does it disturb the gait?
The current generator switches targets across phases, and changing amplitude also
changes those targets. Treat sensitivity to update timing as a hypothesis.

**Implement:** preserve `geometry_gait`, `gait_parameter_policy`, and the zero
policy. Compare fixed sweep, current per-action feedback, and the same feedback
with its chosen sweep held for a full cycle. Change only the update timing in
that comparison. Print phase, measured velocity, raw/clipped sweep, action change,
and actual foot contacts. Predicted swing phase is not measured ground contact.

Write a small `make_observation(state, phase, command)` with named fields/units.
A candidate is forward/sideways velocity, torso height, roll/pitch, and sin/cos
of phase plus target speed. This is a proposal, not a claim of a fully Markov state;
joint positions/velocities and the held previous action may be needed. Name omitted
state when it explains a failure. Keep world +X consistent initially; switching
to body-frame velocity also requires a heading objective.

**Inspect:** overlay sweep with velocity and phase, then replay control and feedback
with the same camera and timescale. Check zero adjustment reproduces the fixed gait.
The user explains one observed difference and selects the action update interval.
Keep phase continuous; do not tune period by recomputing `time / new_period`.

**Why next:** the optimizer needs a clear meaning for an action. This experiment
can establish that interface even if the robot still shuffles poorly.

### 2. Give the gait policy a stochastic action and an honest score

- [ ] Reuse the known REINFORCE path on the selected gait-parameter interface.

**Implement:** a small network produces a Normal mean; start with fixed standard
deviation. Sample a latent sweep adjustment, save it, and map it through a fixed
bounded transform around the baseline. Send numeric offsets to MuJoCo. Score the
latent sample with summed log-probs; do not score a clipped action as though it
were the original Normal sample. Keep the action map fixed during learning.

Define transition reward terms explicitly. Candidate terms are forward-speed
tracking, lateral motion, torso tilt, and action change. Height is not uprightness;
position is not velocity. Choose coefficients by inspecting their contributions
on zero, fixed shuffle, and feedback trajectories. If using effort, distinguish
target magnitude from measured torque or mechanical power. Do not reward visual
animation alone. Specify action duration and any time scaling of each term.

Store each actual transition as observation, latent action, bounded parameters,
applied 18D offsets, log-prob, decomposed reward, next observation/state, and end
flags. A cycle-held action owns the rewards over its full hold interval. Use one
consistent decision interval for the initial learning run.

**Inspect:** trace one row from sampled action to resulting reward; the reset state
receives no action reward. Compute returns backwards and reuse one visible
REINFORCE update. Check finite gradients and changed weights, not guaranteed reward
improvement. Preserve settings and before/after weights. No new trainer is needed.

**Why next:** the existing gradient mechanism now optimizes the behavior and action
interface selected in experiment 1. Reward design stays distinct from optimizer design.

### 3. Predict returns, then measure surprise with an advantage

- [ ] Fit a tiny value network and implement return/advantage estimation.

**Implement:** start with a separate `value(observation) -> scalar` network. Freeze
the actor while fitting values to detached Monte Carlo returns from one batch.
Plot prediction and target by time. Then inspect `A = return - value` before adding
the backward GAE recurrence. Store rollout-time values separately from updated values.
Use PyTorch MSE and an explicit optimizer step; a falling value loss only proves
fit to those targets, not better actions.

For GAE, the TD residual uses reward plus discounted next value minus current value.
Accumulate residuals backwards with gamma and lambda. Use distinct masks: a true
terminal state has no next-value bootstrap; a collection cutoff may bootstrap.
Write both recurrences explicitly: `delta_t = r_t + gamma * b_t * V_old(next_o_t) - V_old(o_t)`
and `A_t = delta_t + gamma * lambda * c_t * A_(t+1)`.
Here `b_t` permits bootstrapping and `c_t` permits continuation within the same
trajectory segment. A detached critic target is `A_t + V_old(o_t)`.
Stop the recurrence at a reset/cutoff so it cannot cross into another episode.
Use the final observation before reset for a truncated transition. Decide whether
the task itself ends at its horizon; if so, expose remaining time when necessary.
See [time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

**Inspect:** hand-check a three-transition trace, a fall terminal, and a nonterminal
cutoff. Compare Monte Carlo advantages with GAE. Freeze/detach actor advantages
and value targets; do not let the actor update alter its own labels through autograd.
The [GAE paper](https://arxiv.org/abs/1506.02438) explains the bias/variance tradeoff.

**Why next:** the actor can distinguish better-than-expected outcomes from states
that were already likely to score well. This supplies PPO's advantage weights.

### 4. Make PPO clipping legible on one frozen batch

- [ ] Replace the actor objective, before adding repeated minibatch updates.

**Implement:** snapshot detached `old_log_prob`, old values, advantages, and value
targets at collection. Recompute current log-probs for the stored observations
and the exact stored latent actions. Define `ratio = exp(new_log_prob - old_log_prob)`.
The actor loss is the negative mean of the minimum of `ratio * advantage` and
`clamp(ratio, 1-epsilon, 1+epsilon) * advantage`.
Keep critic loss separate and make any entropy term explicit; omit it at first.

**Inspect:** at unchanged weights, ratios should be one within tolerance. Plot the
objective versus ratio for both positive and negative advantages. Identify which
side becomes flat in each case. Check the gradient direction on tiny tensors,
then try one actor step on the frozen batch. Log ratio range, clipping fraction,
gradient norm, and a named approximate-KL estimator.
Probability-ratio clipping is not actuator clipping or a guarantee that all ratios
stay in range. Do not refresh the old policy after each optimizer step.

**Why next:** PPO limits the incentive for some large probability changes so a
rollout can support multiple updates. It does not guarantee monotonic improvement.
Use the [original PPO paper](https://arxiv.org/abs/1707.06347) for this objective.

### 5. Train in explicit minibatches, then evaluate visible stride

- [ ] Add the collection/update loop and evaluate against the preserved controls.

**Implement:** collect a fresh on-policy batch; calculate fixed targets; shuffle
indices; loop over a small declared number of epochs and minibatches. Recompute
the current forward pass and loss each minibatch, zero gradients, backpropagate,
and step the optimizers. Never reuse the old autograd graph. Recollect after the
update phase. Keep old log-probs/advantages/targets fixed throughout that phase.
Handle a final short minibatch. If normalizing advantages, do it once on the
frozen rollout and record that choice. Name optimizer rates, gamma/lambda,
epsilon, batch size, epoch count, and any gradient/KL limit.

Start with one environment and a small explicit rollout budget. Inspect one batch
before scaling. Vectorized environments and framework refactors can wait.
Log reward terms, value error, action saturation, gradient norms, ratio/KL metrics,
and completed/failed episodes. Save model/optimizer state, model parameters,
observation ordering, action map, reward definition, seeds, and exact replays.

**Inspect:** use the fixed evaluation suite below for zero, fixed shuffle, the
handwritten policy, REINFORCE, and PPO. Separate deterministic mean-action evaluation
from sampled-policy evaluation. Show all cases, including failures; retain the
last pre-update checkpoint. Change reward, gait generator, or PPO settings in
separate comparisons so causes remain distinguishable.

**Why this is the final transition:** the minibatch loop turns the inspected PPO
objective into training. Controlled forward stepping still needs robot evidence.

## Proposed STRIDE evidence — agree before using it as a gate

These are starting criteria for the user to refine, not an earned checkpoint or
an automatic definition of walking. Use a nominal flat-ground test first.

- A 10-second ordinary-speed replay shows repeated foot lift, forward recovery,
  and supporting stance across at least three cycles. Sliding or a startup lurch
  alone does not count. Inspect foot height/contact traces alongside the view.
- Net forward travel reaches at least one measured torso length. Choose a target
  speed consistent with that duration/distance, and report speed error after a
  declared startup interval. Compare motion during successive cycles.
- Proposed direction limits: lateral drift below a quarter torso length and heading
  error below 15 degrees. Also report roll/pitch and a stated fall/body-contact rule.
- Require finite states, correct timestamps, and no unreported resets. Separate
  commanded limits, applied clipping, and actual foot slip/contact behavior.
- First inspect at least five fixed held-out conditions, including declared initial
  phases and sampling seeds. Repeat training across three seeds before a reliability
  claim. A random seed alone does not vary a deterministic controller's trajectory.
  Report each run, failures, and aggregate results; do not select only the best video.

If PPO optimizes the score but still shuffles, return to the action/reward contract.
If the fixed gait family cannot express the desired steps, compare a smooth foot
trajectory, additional gait parameters, or bounded joint residuals as a separate
treatment. Preserve the previous action space and objective for comparison.
Do not assume that tuning sweep alone can create a proper stride.

STAND remains earned with its existing limits. Disturbance recovery is excluded.
No STRIDE claim, deployment, hardware work, or checkpoint update follows from
completing this checklist. Use the repository's commit-boundary contract when
actual evidence later warrants a capability claim.
