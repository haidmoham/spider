# PPO in the C-1N platform

Notebook 04 is frozen at the accepted
[PPO-100 crude baseline](../artifacts/ppo-crude-baseline-20260915/README.md).
It owns the experiment record and the user's learner. Runtime policy execution
lives in `spider/policy.py` and `spider/policy_run.py`. Neither imports `lab` or
reads notebook code. These modules perform inference; they do not train PPO.

## Run the prepared comparison

Use the existing C-1N environment. A new runtime-only environment can install
`requirements-policy.txt`; Jupyter is not required.

```powershell
.venv/Scripts/python -m spider policy --compare
```

This command **runs four new five-second episodes**, then opens the synchronized
viewer. It does not train. Default: checkpoint 100, mean action, seed 201, normal
playback speed. New recordings and the treatment receipt go into a unique folder
under `telemetry/policy/`.

| Pane | Treatment | Difference from accepted policy |
| --- | --- | --- |
| Top left | baseline | None |
| Top right | lower | Hip -0.12 rad and knee +0.12 rad per leg, ramped over 2 s |
| Bottom left | smooth | Exponential filter on policy offsets, time constant 0.08 s |
| Bottom right | stalk | Both interventions |

The filter starts at zero offset from the neutral command. The posture ramp
starts at zero and reaches its target smoothly. Final actuator range clipping
remains active. The settings are named constants in `spider/policy.py` and are
written into each recording's `metadata.json`.

For the accepted policy alone:

```powershell
.venv/Scripts/python -m spider policy --treatment baseline --presentation original
```

Use `--headless` to record without opening a viewer. Use `--sampled` to restore
the checkpoint's Gaussian exploration. Use `--seed` for a declared comparison
seed. The actor runs once per 10 physics steps (20 ms). Episodes stop on the
saved fall threshold or the requested duration, within the trained horizon.

## What this first treatment can establish

The user's direction is a **low, deliberate stalk inspired by RS3 Araxxor**.
[Jagex's Araxxor BTS video](https://www.youtube.com/watch?v=SScsXx3H-SQ) identifies
the reference; [RS3 combat footage](https://img.pvme.io/images/DK7CuBC.gif) shows
the low, wide silhouette. Combat footage does not establish exact walk-cycle
timing. No game models or animation assets are copied into C-1N.

This is the first bounded control comparison, not a completed Araxxor gait.
The hip/knee bias is intended to lower and widen the stance. The resulting body
height and stability are unknown until the treatment runs. Filtering can remove
useful timing from the learned actions. A smoother command need not produce a
better walk. Splitting the two interventions makes that failure visible.

After running, compare actual torso height, foot clearance, forward travel,
sideways travel, falls, and whether each foot plants cleanly. `trace.csv` records
height, contacts, support margin, displacement, and applied joint targets beside
the exact state replay. This single-seed view is a design check, not population
validation. Repeat fixed seeds before accepting a new robotics capability.

If posture improves while travel survives, the next separate treatment can address
stance duration and leg sequencing. Do not add an authored gait and silently call
it learned behavior. If filtering removes travel, keep the unfiltered branch.
The user reviews each change before the next behavioral treatment.

## Presentation boundary

The optional `stalk` presentation uses steady key/fill/rim lighting, a charcoal
floor, and a low three-quarter camera. The camera follows each recorded torso;
the visible displacement readout and logs remain the measurement source.
Use `--presentation original` for the fixed camera comparison.

The visual preset is applied only to loaded replay models. It never modifies
recorded poses, body dimensions, masses, joints, contacts, friction, actuators,
solver settings, gravity, or integration timestep. Replay uses the original
recorded frames; it does not interpolate away robot motion. Steady lighting and
a separate atmosphere layer adapt the design-library composition approach without
changing the physical subject.

![Presentation of an existing baseline frame](../artifacts/stalk-preview.png)

This image is a saved baseline frame with the new presentation, **not a result
from the lower/stalk controller**. The first pass was too dark; brighter broad
fill and a charcoal floor improved leg visibility in the inspected static image.
Moving-view appearance still needs user review. A shareable video comes after
an accepted motion treatment; no Twitter post or video export was produced here.

## Preparation checks

Checked without new physics rollouts: baseline action parity on synthetic states
and exact agreement with 20 consecutive saved commands for each of the accepted
mean and sampled replays, local sampling RNG isolation, target bounds and reset, CLI dispatch, ten-step
control cadence using a fake stepper, and unchanged physics arrays under styling.
The static preview restores one saved state and computes display transforms.
Notebook 04, the accepted artifacts, and both model XML files remain unchanged.
