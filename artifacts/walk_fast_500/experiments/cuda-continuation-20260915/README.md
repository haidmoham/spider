# CUDA cadence continuation — 2026-09-15

This experiment continued `walk_fast_300` from n=300 to n=500. PPO optimization
moved to an NVIDIA GeForce RTX 3080 with PyTorch 2.14.0+cu126. CPU MuJoCo
rollouts, rollout inference, reward, model, PPO settings, cadence bounds,
optimizer history, and seed progression remained unchanged. The exact parent
is `artifacts/walk_fast_300/walk_fast_300.pt`, SHA-256
`81fa8843b2d0ce5825cd3a653ea1cb71f39dc071b8bd907f0afad6d176f4431f`.

| Update | Mean speed | Sampled average | Falls | Maximum joint-limit violation |
| --- | ---: | ---: | ---: | ---: |
| 400 | 0.828779 m/s | 0.767651 m/s | 0/24 | 0 rad |
| 500 | 0.887987 m/s | 0.821139 m/s | 0/24 | 0 rad |

The deterministic seed-201 n=500 recording reports a mean commanded cadence of
1.724318 Hz and a 0.204586 m/s stance-foot speed RMS slip proxy. The numerical
evaluation recorded no failures, but it does not prove deliberate stride
quality, terrain robustness, or push recovery.

The user explicitly requested `walk_fast_500` and authorized closing the goal
after the n=500 run completed. The named checkpoint preserves that stopping
point. The name and goal closure do not strengthen the measured gait claims.

`evidence.zip` contains authorization, configuration, update history, frozen
model and sources, evaluator sources, full acceptance records, and all 48
five-second recordings from mean and sampled modes across seeds 201 through
212. Empty live-viewer output logs were excluded because they contain no
measurements. `evaluation-400.json` and `evaluation-500.json` provide concise
summaries. `manifest.json` records SHA-256 digests for every preserved file.
