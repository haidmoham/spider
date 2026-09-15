# PPO-100-CRUDE-20260915

The user accepted this result as a crude learned baseline on 2026-09-15 after
running notebook 04 and inspecting its four-pane 50/100 comparison. This preserves
learned forward travel under the fixed flat-ground task. It does not earn STRIDE,
disturbance recovery, terrain robustness, or an Araxxor-style crawl.

## Recorded comparison

One continuous PPO run, checkpoints at 50 and 100 updates. Each treatment used
12 development seeds (201–212), five seconds per episode.

| Updates | Action mode | Mean forward mm | Mean absolute lateral mm | Falls |
| --- | --- | ---: | ---: | ---: |
| 50 | Sampled | 305.04 | 83.43 | 0/12 |
| 100 | Sampled | 1371.38 | 98.39 | 0/12 |
| 50 | Mean | 2.87 | 2.18 | 0/12 |
| 100 | Mean | 1894.14 | 101.24 | 0/12 |

These values are recorded in notebook 04; `comparison.csv` holds the episode
measurements. The environment starts from the same pose. Seeds vary action noise;
mean-action repetitions do not constitute 12 distinct initial conditions.
The user's aesthetic acceptance is separate from a formal locomotion gate.

## Preservation and provenance

`manifest.json` identifies the exact local run and hashes every retained artifact.
The two checkpoints retain actor, critic, both Adam states, update counts, and
task/PPO/reward settings. `learning-functions.json` is the saved source used in the
second training block. `spider.xml` is the current unchanged model source; the
replay archive contains the exact compiled models saved during evaluation.
Notebook 04 retains its executed cells and outputs in this commit.
The exact execution Git revision and package versions were not recorded.

The full local telemetry remains at `telemetry/ppo/20260915T152940537445Z`.
This compact bundle retains all comparison rows and the four viewed seed-201
replays. Per-step logs and other seeds' trajectories remain in ignored telemetry.

## Replay when requested

From the repository root, extract the archive into ignored telemetry:

```powershell
python -m zipfile -e artifacts/ppo-crude-baseline-20260915/replays.zip telemetry/ppo-crude-baseline-replay
python -m spider replay-grid telemetry/ppo-crude-baseline-replay/0 telemetry/ppo-crude-baseline-replay/1 telemetry/ppo-crude-baseline-replay/2 telemetry/ppo-crude-baseline-replay/3 --speed 1
```

Top row: 50/100 sampled. Bottom row: 50/100 mean action. All use seed 201.
These commands replay recorded states; checkpoint preparation did not run them.
Keep this bundle unchanged when preparing the next treatment.
