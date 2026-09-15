# Policy cadence continuation — 2026-09-15

This experiment continued the user-approved `walk_fast_200` policy from n=200
to n=300 without changing the reward, model, PPO settings, or cadence bounds.
It preserves the requested n=250 and n=300 evaluation boundaries.

| Update | Mean speed | Sampled average | Falls | Maximum joint-limit violation |
| --- | ---: | ---: | ---: | ---: |
| 250 | 0.701920 m/s | 0.646578 m/s | 0/24 | 0 rad |
| 300 | 0.758568 m/s | 0.697075 m/s | 0/24 | 0 rad |

At n=300, mean speed increased by 0.160952 m/s from the accepted n=200 result
of 0.597616 m/s. The result was below the recorded linear extrapolation of
about 0.87 m/s by 0.111432 m/s. That extrapolation was a hypothesis, not an
acceptance threshold. The deterministic seed-201 recording reports a mean
commanded cadence of 1.683502 Hz and a 0.161192 m/s stance-foot speed RMS slip
proxy.

The user requested the `walk_fast_300` name after training reached this boundary.
This experiment archive preserves the measurements behind that request. The
named checkpoint, acceptance manifest, and catalog state are maintained outside
this experiment directory.

`evidence.zip` contains the authorization, configuration, update log, frozen
model and sources, evaluator sources, acceptance records, and all 48 measured
five-second recordings for mean and sampled modes across seeds 201 through 212.
Two empty viewer log files were excluded because the replay viewers still held
them open; they contain no measurements. `evaluation-250.json` and
`evaluation-300.json` provide concise summaries.

These fixed-initial-state, flat-ground results do not establish terrain or push
robustness. `manifest.json` records SHA-256 digests for all preserved files.
