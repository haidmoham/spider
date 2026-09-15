# Held-out REINFORCE: next diagnosis

2026-09-14 local date. STRIDE remains unearned. User owns interpretation and fixes.

## Evidence

Baseline: `telemetry/reinforce-debug/20260915T010216441278Z/`, checkpoint
`block-00000/checkpoint-00020.pt`. Saved setup and learning functions match the
current saved notebook 04. One training initialization, 20 updates, 80 episodes.
No additional training ran for this diagnostic.

Evaluation: `evaluation-00020-011012369488/` under that baseline directory.
Open `review.md`, `diagnostic.png`, `comparison.csv`, and `ranges.csv` there.
The `baseline/` copy and `receipt.json` preserve weights, sources and input hashes.
All 28 rollouts completed with finite summary values and no falls. Every transition
reward equals its logged terms. New action seeds 201-212 are disjoint from training
and earlier evaluation seeds. Reset/physics stayed fixed. This is not terrain/reset
generalization or independent-training evidence. Controls and mean policies ran once.

| Treatment | Mean forward travel in 5 s | Mean reward |
| --- | ---: | ---: |
| Initial sampled | -0.11 mm | 11.35185 |
| Trained sampled | 22.89 mm | 12.48525 |
| Initial mean | 0.62 mm | 11.34473 |
| Trained mean | 0.75 mm | 11.35218 |
| Neutral | 0 mm | 11.30830 |
| Fixed shuffle | 140.61 mm | 18.21687 |

Trained sampled displacement spans -71.27 to +70.06 mm. All outcomes are saved.
Replay uses seed 201, selected before collection, with initialization, trained
policy (n=20), neutral and fixed shuffle at 1x. The plot shows displacement,
progress/height reward totals and foot-contact count. Count alone cannot establish
stepping or distinguish sliding.

Reproduce from the repository root; this executes trusted saved setup/functions
and evaluation rollouts, never the training cell:

```powershell
.venv/Scripts/python -m lab.heldout_reinforce telemetry/reinforce-debug/20260915T010216441278Z --updates 20
```

## User's next attempt

1. Describe the trained pane's repeated motion without using its score.
2. Which term pays for that behavior? Compare the same term in neutral control.
3. Propose one measurement or change that could disprove your explanation.

Keep `observation_from` and `reward_terms` editable in notebook 04's **EDITABLE
LEARNING FUNCTIONS** cell. Choose the hypothesis before changing either.

## Reference shelf: use only on an observed need

- Objective/measurement mismatch: [Ant, Rewards and Episode End](https://gymnasium.farama.org/environments/mujoco/ant/#rewards).
  Start with one term, its units and activation condition. Check observations and
  actions before borrowing. Ant commands torques; C-1N commands position targets.
  An action penalty is not automatically measured torque, energy or work.
- Noisy credit: [GAE, section 3](https://arxiv.org/abs/1506.02438).
  Read one estimator equation when return diagnostics motivate it.
- Large or inefficient updates: [Spinning Up PPO, Key Equations](https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations)
  and [PPO, section 3](https://arxiv.org/abs/1707.06347).
  Compare old/new action probabilities on the same samples before implementing
  clipping. PPO does not define the desired walking behavior.
- Foot/contact failure: [Learning Quadrupedal Locomotion over Challenging Terrain](https://leggedrobotics.github.io/rl-blindloco/).
  Select one relevant supplementary reward/observation definition after a measured
  failure identifies the need. Do not import a complete reward stack.
- Reliability: [Deep RL at the Edge of the Statistical Precipice, section 4](https://arxiv.org/abs/2108.13264).
  Separate training-seed variation from action-seed variation. Report every trial.

OpenAI dexterity and emergent-tool-use work remain optional references for a
specific exploration/curriculum need, not the next reading assignment.
The active 48/50 STRIDE target and practice boundaries are in `TODO.md`.
