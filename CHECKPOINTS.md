# Named walking checkpoints

Start here for the demo skill. Render saved measured states at 1x. These names
identify exact assets and their approval status, not whichever training file is newest.

| Name | Status | Weights and replay | Evidence |
| --- | --- | --- | --- |
| `walk_stable_100` | Locked; user-approved walking baseline | [Manifest](artifacts/walk_stable_100/manifest.json) | 1.1 Hz; 100 PPO updates; 0/24 falls; mean 0.317506 m/s |
| `walk_fast_200` | Development target; no accepted checkpoint assigned | None yet | Evaluate separately named ablations; user approval is required |

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
  gait generation, initially cadence with continuous phase. Not yet implemented.

The original chassis, crude PPO-100 and Notebook 4 remain controls. The original
speed threshold is 0.37882745 m/s. User approval of the stable walking baseline
does not lower the fast-walk target or establish terrain/push robustness.
