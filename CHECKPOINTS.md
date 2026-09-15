# Named walking checkpoints

Start here for the demo skill. Render saved measured states at 1x. These names
identify exact assets and their approval status, not whichever training file is newest.

| Name | Status | Weights and replay | Evidence |
| --- | --- | --- | --- |
| `walk_stable_100` | Locked; user-approved walking baseline | [Manifest](artifacts/walk_stable_100/manifest.json) | 1.1 Hz; 100 PPO updates; 0/24 falls; mean 0.317506 m/s |
| `walk_fast_200` | User-selected speed checkpoint; stride-quality limits remain | [Manifest](artifacts/walk_fast_200/manifest.json) | n=200; mean 0.597616 m/s; 0/24 falls; preserved before continuation |
| `walk_fast_300` | User-requested named checkpoint; stride-quality limits remain | [Manifest](artifacts/walk_fast_300/manifest.json) | n=300; mean 0.758568 m/s; 0/24 falls |
| `walk_fast_500` | User-requested named checkpoint; training stopped at 500 | [Manifest](artifacts/walk_fast_500/manifest.json) | Mean 0.887987 m/s; sampled average 0.821139 m/s; 0/24 falls |

`walk_stable_100` has the approved metallic-black chassis, red lighting, six legs
and gravity-responsive googly eyes. Its mean replay is in
`artifacts/walk_stable_100/replay`; weights are `walk_stable_100.pt` in its parent
directory. Use `spider.reference_training.ReferenceResidualPolicy` to load it.
The legacy `spider policy` command does not implement this residual controller.

For a four-pane baseline replay using the real renderer:

```powershell
.venv/Scripts/python -m spider replay-grid artifacts/walk_stable_100/replay artifacts/walk_stable_100/replay artifacts/walk_stable_100/replay artifacts/walk_stable_100/replay --speed 1 --presentation stalk
```

For composed demo footage, `spider.viewing.film.render_comparison` accepts four
recording directories and four explicit display labels. Generated video belongs
under ignored `telemetry/`. Do not label a cadence-only probe as further training.

## Experimental lineage

- `reference-ppo-round-01`: fresh n=0→100, 1.1 Hz; parent of `walk_stable_100`.
- `cadence-evaluation-01`: same n=100 weights at 1.1/1.35/1.5 Hz; three five-second
  probes, no PPO updates. [Evidence](artifacts/cadence-evaluation-20260915/README.md).
- `reference-ppo-cadence-15-round-01`: fixed 1.5 Hz continuation from n=100;
  user stopped it at saved n=185. Last evaluated checkpoint n=150 reached
  0.518150 m/s mean, 0.516098 m/s sampled average, zero falls in 24 evaluations.
  This is an experiment, not `walk_fast_200`. It must not be relabelled as n=200.
- Next requested approach: return to `walk_stable_100` and let the policy adjust
  gait generation, initially cadence with continuous phase. Implemented in
  `spider/cadence_action_training.py`; experiment `policy-cadence-action-round-01`
  is a separate n=100→200 continuation. Its initial five-second mean replay was
  byte-for-byte equal in state arrays to the locked baseline. At n=200 it reached
  0.597616 m/s mean and 0.591694 m/s sampled average with zero falls and zero
  joint-limit violations in 24 evaluations. [Archived evidence](artifacts/walk_fast_200/experiments/policy-cadence-action-20260915/README.md).
  The user subsequently selected this as `walk_fast_200` before requesting
  a speed-comparison demo and n=200→300 continuation. Contact fragmentation
  remains documented; this selection does not complete the full stride goal.

The original chassis, crude PPO-100 and Notebook 4 remain controls. The original
speed threshold is 0.37882745 m/s. User approval of the stable walking baseline
does not lower the fast-walk target or establish terrain/push robustness.

## Requested stopping point

The user requested closure once the 500-update run completed. Training stopped
at 500 and the final 24 five-second evaluations passed the numerical gate.
PPO optimization for n=300 to n=500 ran on the RTX 3080; rollout physics and
inference stayed on CPU. Full experiment provenance is preserved with each name.

Mean-policy speeds at n=100/200/300/400/500 were
0.318/0.598/0.759/0.829/0.888 m/s. Gains slowed; the near-linear extrapolation
from the first two intervals did not persist. The n=500 stance-foot speed RMS
proxy was 0.204586 m/s versus 0.069600 m/s for the locked stable walk.
Contact fragmentation remains. Closure at the user's stopping point does not
establish a perfected deliberate stride, lower energy use, or terrain/push robustness.
