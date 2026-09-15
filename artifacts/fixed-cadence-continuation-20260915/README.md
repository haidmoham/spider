# Fixed-cadence continuation — stopped

The user stopped this experiment at saved update 185 on 2026-09-15 and requested
a different approach from `walk_stable_100`. The original requested endpoint was
200; no completion at 200 is claimed. This experiment is not `walk_fast_200`.

It restored the reference PPO-100 weights and both optimizers, then continued at
fixed 1.5 Hz. The last evaluated checkpoint, n=150, passed the numerical gate:
mean 0.518150 m/s; sampled mean 0.516098 m/s; 0/24 falls; all episodes completed
five seconds with finite states and no joint-limit violations. User visual
acceptance and a new capability checkpoint were not granted.

Saved weights include n=150 and n=185. `evaluation-150.json` preserves all metrics.
`evidence.zip` contains the 24 recorded evaluations, evaluator sources, frozen
training sources, configuration, authorization, update log and stop receipt.
`manifest.json` hashes the binary evidence. The latest n=185 weights were not
evaluated after the stop request.
