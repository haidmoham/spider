# Policy cadence action — 2026-09-15

This unapproved experiment continues the locked `walk_stable_100` policy from
n=100 to n=200. It adds cadence as action 19 while keeping the gait reference
at 1.1 Hz. The action commands a phase-continuous cadence from 0.8 to 1.8 Hz.

The archive preserves checkpoints n=150 and n=200. `evidence.zip` contains the
training authorization and configuration, update log, frozen training and
evaluation sources, and all 48 measured five-second evaluation recordings.
Each checkpoint was evaluated in mean and sampled modes for seeds 201 through
212. `evaluation-150.json` and `evaluation-200.json` give concise summaries.

| Update | Mean speed | Sampled average | Falls | Maximum joint-limit violation |
| --- | ---: | ---: | ---: | ---: |
| 150 | 0.462593 m/s | 0.448970 m/s | 0/24 | 0 rad |
| 200 | 0.597616 m/s | 0.591694 m/s | 0/24 | 0 rad |

Both checkpoints passed the recorded numerical gate of 0.37882745 m/s. The
n=200 deterministic recording reports a mean commanded cadence of 1.609139 Hz
and a 0.123560 m/s stance-foot speed RMS slip proxy for seed 201. These results
use a fixed initial state and flat-ground scenarios. They do not establish
terrain or push robustness.

`walk_fast_200` remains a development target. This experiment is not accepted
as its checkpoint. Visual review and explicit user approval remain required.
`manifest.json` records SHA-256 digests for the checkpoints, summaries, and
compressed evidence.
