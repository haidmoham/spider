# Named walking checkpoints

Start here for the demo skill. Render saved measured states at 1x. These names
identify exact assets and their approval status, not whichever training file is newest.

| Name | Status | Weights and replay | Evidence |
| --- | --- | --- | --- |
| `walk_stable_100` | Locked; user-approved walking baseline | [Manifest](artifacts/walk_stable_100/manifest.json) | 1.1 Hz; 100 PPO updates; 0/24 falls; mean 0.317506 m/s |
| `walk_fast_200` | User-selected speed checkpoint; stride-quality limits remain | [Manifest](artifacts/walk_fast_200/manifest.json) | n=200; mean 0.597616 m/s; 0/24 falls; preserved before continuation |
| `walk_fast_300` | User-requested named checkpoint; stride-quality limits remain | [Manifest](artifacts/walk_fast_300/manifest.json) | n=300; mean 0.758568 m/s; 0/24 falls |
| `walk_fast_500` | Authorized training target; not yet recorded | Pending n=500 | Evaluate at 400 and 500; stop at 500 |

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
