# walk_stable_100 — locked walking baseline

The user explicitly locked this learned walk on 2026-09-15. Do not overwrite it.

- Checkpoint: `walk_stable_100.pt`, copied byte-for-byte from the original
  reference PPO-100 checkpoint.
- Model: six-leg candidate chassis, metallic black with red lights and accents.
- Reference cadence: 1.1 Hz. Mean speed: 0.317506 m/s. Sampled mean: 0.321624 m/s.
- Evidence: twelve sampled seeds and twelve mean seeds, 201–212, five seconds
  each; zero falls, finite states and no joint-limit violations. Mean seeds share
  one initial state. These tests do not establish terrain or push robustness.
- Replay: `replay/` contains measured mean seed 201. Use normal-speed playback.
- Audit: `manifest.json` records user approval, source, runtime class, metrics,
  evaluation path and SHA-256 hashes for weights and every replay file.

This is an accepted walking baseline, not a claim that the original PPO-100 speed
threshold was passed. Future approaches branch from these weights and receive
distinct experiment IDs. `walk_fast_200` remains a development target.
