# Ant policy learning: objective → policy → behavior

## Learning goal

This is a guided, deliberately incomplete bridge from prompting toward writing
small pieces of simulation and policy code in VS Code. It traces one loop:

`observation → policy → action → MuJoCo step → reward → policy update → changed behavior`

Gymnasium's built-in `Ant-v5` is the temporary quadruped. It is a learning
fixture, not a C-1N adapter and not evidence about C-1N locomotion.

## Sources

- [Farama: load a custom quadruped model](https://gymnasium.farama.org/v1.1.1/tutorials/gymnasium_basics/load_quadruped_model/)
- [Farama: Ant environment contract](https://gymnasium.farama.org/environments/mujoco/ant/)
- [Stable-Baselines3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

The notebook uses Stable-Baselines3 PPO so the learning task is understanding
the interface and objective, not implementing an optimizer from scratch.

## PC setup and VS Code

The shared environment lives one directory above this repository at
`../.venv`. In PowerShell, from this experiment directory:

```powershell
..\..\..\.venv\Scripts\python.exe -m pip install -r ..\..\requirements-policy.txt
..\..\..\.venv\Scripts\python.exe -m ipykernel install --user --name robotics-shared --display-name "robotics shared (.venv)"
code .\ant_policy_learning.ipynb
```

Open `ant_policy_learning.ipynb` in VS Code's Jupyter editor and select
`robotics shared (.venv)`. If the shared environment does not exist yet, create
it at the parent workspace level according to the workspace setup before
installing these requirements.

## Run stages

1. Run only the setup and one-step Ant smoke cell. It resets the environment,
   samples a legal action, and takes one step; it does not train.
2. Implement the TODO cells in the notebook using autocomplete and the linked
   API documentation. Start with an inspectable random rollout and one
   transition trace.
3. Use a `10_000`-step headless PPO smoke run to test your wiring.
4. Resume only after inspecting its outputs, up to `1_000_000` steps on the
   PC. Keep one training seed per treatment for this learning exercise.
5. Evaluate deterministic policies on fixed seeds `[0, 1, 2, 3, 4]`; do not
   treat that small sample as a benchmark.

Training stays headless. Render only evaluation rollouts as inline RGB frames.
Generated checkpoints, logs, TensorBoard data, videos, and rollout outputs
stay local under this experiment and are ignored by Git.

## Boundary

The final notebook section is a prompts-only transfer worksheet. It asks how
the Ant loop would map to C-1N later, but it contains no C-1N environment,
policy, reward, training code, or integration change. This scaffold does not
modify C-1N, `integrations/`, or the current experiment queue.
