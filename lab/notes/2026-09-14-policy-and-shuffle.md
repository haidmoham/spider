# Policy update and shuffle study — 2026-09-14

The current source and saved outputs are in
[notebook 03](../notebooks/03_coordinated_baseline.ipynb).
The user corrected the sampled-policy gradient path and the two-sided joint
probe. The agent supplied the visible multi-step REINFORCE implementation and
the video presentation. Execution is not a claim of demonstrated understanding.

## Preserved run evidence

- One REINFORCE update: saved global gradient norm `1350.26953125`, weight-change
  L2 `0.014654039405286312`. Reward remains `next_x + 0.1 * next_z`.
- The latest saved notebook comparison points to
  `telemetry/reinforce/20260914T200149052808Z/`. Earlier outputs and checkpoints
  also remain in `telemetry/reinforce/20260914T193259163316Z/`.
- The joint probe now divides the position difference by the full angular
  separation. Saved estimates remain close when probe spacing halves.
- The geometry-informed tripod uses coxa sweep `0.04 rad`, lift `0.05 rad`,
  period `0.8 s`, and 200 actions held for 10 physics steps each.
- The four-second shuffle replay used for the demo is
  `telemetry/geometry-tripod-v0/treatment-6n3wvcg6/`. Its saved states are finite;
  final displacement is approximately `+0.1168 m` forward and `-0.0099 m` sideways.
  Torso height ranges from `0.4483 m` to `0.4515 m`.

These local telemetry paths are ignored by Git. The short shuffle is a motion
study, not population-level validation, learned walking, or a new checkpoint.
STAND and its limits remain unchanged. No experiment ran during wrapup.

### Later feedback comparison

The user corrected forward-axis selection, error sign, and sweep clipping in
`gait_parameter_policy`. The saved five-second comparison reports forward/sideways
displacements of `+0.140608 / +0.003545 m` for the fixed shuffle and
`-0.037958 / +0.150726 m` for feedback. Several early commands reach the `0` or
`0.08 rad` sweep bound. This is an observed failure, not a causal diagnosis.
Exact replays remain under `telemetry/gait-policy-iterations/` as
`treatment-cp863h3r` (control) and `treatment-09ndxul6` (feedback).

## Shareable demo

`telemetry/exports/c1n-shuffle-social.mp4` is five seconds, 1080 square, 30 fps,
H.264/yuv420p with fast-start metadata. It replays the recorded four-second
trajectory at 0.8x speed. The camera is fixed. The ground grid is display-only.
Displacement and time readouts come from the displayed recorded state.

Reproduce from the repository root with FFmpeg available:

```powershell
.venv/Scripts/python lab/render_shuffle_demo.py --recording telemetry/geometry-tripod-v0/treatment-6n3wvcg6
```

The renderer uses Windows system fonts. The original visual treatment combines
large warm-white typography, small monospaced measurements, a coral accent,
and a quiet dark stage. It adapts hierarchy from recent frontend work without
copying private design assets. Source hashes are in the local export receipt.
