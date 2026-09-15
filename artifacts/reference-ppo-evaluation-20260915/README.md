# Reference PPO: 50 and 100 updates

Recorded on 2026-09-15. One continuous fresh run used seed 11, 800 five-second
episodes, and 200,000 control steps. Notebook 4 remained frozen. Its checkpoint
schedule and evaluation seeds were applied to the approved candidate chassis.

| Updates | Mean speed (m/s) | Sampled average (m/s) | Falls | Joint-limit violations |
| --- | ---: | ---: | ---: | ---: |
| 50 | 0.299001 | 0.288070 | 0/24 | 0/24 |
| 100 | 0.317506 | 0.321624 | 0/24 | 0/24 |

Each checkpoint was evaluated in sampled and mean modes on seeds 201–212.
All 48 recordings completed five seconds with finite states and valid actuator
targets. Mean-mode seeds share one initial condition and reproduce one trajectory.
These runs do not establish terrain or disturbance robustness.

**Neither checkpoint passes the accepted PPO-100 mean-speed threshold of
0.37882745001803586 m/s. Neither is promoted.** This archive records an experiment;
it does not advance a robot capability checkpoint or claim user visual approval.

`summary.json` preserves per-seed metrics and gate failures. `manifest.json`
hashes both saved policy files and `recordings.zip`. The ZIP contains all 48
model/state/trace recordings, evaluation sources, frozen training sources,
configuration, authorization, and update log. Policy files also include model
XML, optimizer state, reference configuration, and action mapping.

The local comparison windows were opened at 1x with frozen PPO-100 and approved
original tripod controls above candidate sampled/mean seed-201 recordings.
Each window produced a ready marker and screenshot with no stderr output.
See [the reproduction command](../../docs/gait-chassis.md#checkpoint-comparison).
