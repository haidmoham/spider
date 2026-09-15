# C-1N

A six-legged MuJoCo robot for learned locomotion and simulation experiments.

For integrated walking demos and auditable weights, use the [named checkpoint catalog](CHECKPOINTS.md).

**C-1N v0.3 STRIDE is complete with `walk_fast_500`.** It reached 0.888 m/s
mean-policy speed with zero falls in 24 five-second evaluations. See the
[capability record and evidence limits](docs/checkpoints/stride.md).

[04_mdp_contract.ipynb](lab/notebooks/04_mdp_contract.ipynb) preserves the frozen
learning baseline. Integrated policy work lives in Python under `spider/`.
[LEARNING.md](LEARNING.md) explains setup and ownership.
[01_control_step.ipynb](lab/notebooks/01_control_step.ipynb) is the earlier control-step exercise.
The [experiment index](lab/notebooks/README.md) maps each bounded notebook to its
concept and the relevant robot implementation. This repository now owns the
smaller experiments previously kept in robotics-test-bench.

## Structure

| Location | Responsibility |
| --- | --- |
| `spider/simulation.py` | Model loading, neutral reset, measurements, physics stepping |
| `spider/controllers.py` | Existing STAND controller and legacy SHUFFLE gait |
| `spider/runtime.py` | Controller execution, disturbances, live commands |
| `spider/learning.py` | Policy action adapter, short recordings, notebook plots |
| `spider/recording.py` | STAND telemetry and state capture for replay |
| `spider/viewing/` | Live display, replay, appearance, matched renders |
| `lab/notebooks/` | User policy, experiment settings, measurements, interpretation |
| `lab/` | Isolated mechanics fixtures used by the notebooks |
| `tests/` | Physics, adapter, recording, and visual regression checks |
| `model/` | Robot XML and frozen physics reference |
| `telemetry/` | Local traces and executed notebook archives; ignored by Git |

Follow **notebook → learning → simulation** when writing a policy.
Controllers use simulation measurements to choose targets. The runtime advances
those controllers through the same physics step. Viewing code does not own reset
or a second physics loop. Replay restores recorded states without integrating.

## Run

Use the same Python environment for notebooks and commands:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install --upgrade -r requirements-bootstrap.txt
.venv/Scripts/python -m pip install -r requirements-learning.txt
.venv/Scripts/python -m jupyter lab lab/notebooks/04_mdp_contract.ipynb
```

`requirements.txt` pins the runtime; `requirements-test.txt` adds headless test tools;
`requirements-learning.txt` adds the notebook/PyTorch stack. Use Python 3.12 and the
**C-1N pairing** kernel. The legacy Ant notebook uses a separate `.venv-ant` and
**C-1N Ant** kernel; see [LEARNING.md](LEARNING.md#environment).

One command entry point serves all runtime operations:

```powershell
python -m spider run
python -m spider run --headless --experiment none --seconds 1
python -m spider run --headless --experiment stand --seconds 10 --trace telemetry/stand.npz
python -m spider run --experiment shuffle
python -m spider run --seconds 10 --shove-suite telemetry/shoves
python -m spider command state
python -m spider command perturb 1 0 0 --seconds 0.2
python -m spider replay telemetry/path-to-recording --speed 0.25
python -m spider render
python -m unittest discover -s tests -v
```

`run` defaults to STAND. `none` holds neutral targets. `shuffle` runs the preserved
legacy gait and its six joint/torque plots. Add `--headless` for repeatable recordings.
The shove suite retains one control and eight directions for each nonzero force:
0, 0.25, 0.5, 0.75, and 1 mg, held for 200 ms. Angles start at world +X and increase
counter-clockwise toward +Y. Its three live support plots and trace format are preserved.

The old `simulate.py`, `interact.py`, `walk.py`, `view.py`, and `simctl.py` launchers
are replaced by the commands above. Import from `spider` modules, not CLI re-exports.
`python -m spider --help` lists commands; each command has `--help`.

CI uses Python 3.12 and the same `unittest` command. It checks physics, recording,
the public CLI, and notebook structure/syntax. It does not run lesson cells or
graphical viewers. Native display checks remain local. The required GitHub check
is named **Headless C-1N simulation**.

## Integrated PPO policy

The accepted PPO actor now has a Python runtime entry point, separate from the
frozen notebook. With the C-1N environment installed:

```powershell
.venv/Scripts/python -m spider policy --compare
```

This runs four bounded episodes and opens the stitched viewer: accepted policy,
lower posture, filtered commands, and both together. The three treatments are
unvalidated; the physics model stays fixed. See [the integration guide](docs/ppo-platform.md)
for exact changes, inference-only dependencies, recordings, and review criteria.

## Preserved capability and evidence

STAND is the earned six-contact baseline. Disturbance recovery is excluded.
SPAWN and SHUFFLE remain historical comparisons. STRIDE is complete through
the user-accepted [walk_fast_500](artifacts/walk_fast_500/README.md).
The [crude PPO baseline](artifacts/ppo-crude-baseline-20260915/README.md) preserves
the user-accepted 50/100-update comparison, both model checkpoints, and four recorded
replays. It records learned forward travel on the fixed task, not a reliable crawl.
The completed learned-locomotion work is tracked by [spider #17](https://github.com/haidmoham/spider/issues/17)
with [test-bench #25](https://github.com/haidmoham/robotics-test-bench/issues/25)
retained as source provenance. `LEARNING.md` selects the current work.

`C-1N v0.2 - STAND` is supported by a deterministic 10-second headless
baseline: all six feet remained in contact, support margin stayed at or above
`0.2346 m`, torso height stayed at or above `0.4495 m`, and torso angular speed
remained numerically zero. The declared `1 mg` shove is retained as a failed
recovery case; disturbance recovery is not part of this checkpoint.

Neutral reset uses torso height 0.45 m and joint targets `(0, -0.2, 1.1)` radians
per leg. `simulation.py` alone owns that reset. The stance controller applies a
bounded all-foot correction only with six declared contacts and adequate support margin.
It is not an attitude-recovery controller. Telemetry v1 remains instrumentation,
not an additional capability claim.

The Porcelain Surveyor appearance preserves mechanics. Extra visual sites have no
mass or contact behavior. Responsive pupils modify render-site positions only.
The frozen reference is `model/spider_physics_baseline.xml`. `render` compares matched
cameras; its `--before-directory` option accepts earlier render output.

Existing recordings remain under `telemetry/` and `artifacts/c1n_redesign/`.
Historical verification is in [docs/history](docs/history/).
The 2026-09-14 pre-trim source and executed notebooks are also preserved locally
under `telemetry/architecture-before/`.
