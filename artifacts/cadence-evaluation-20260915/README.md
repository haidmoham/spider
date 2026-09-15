# Cadence probes: same learned weights

The user approved three five-second deterministic seed-201 evaluations on
2026-09-15. Cadences were 1.1, 1.35 and 1.5 Hz. All used the saved reference
PPO-100 actor, critic, optimizer states and physical model. Only gait cadence
and its corresponding reference-speed observation changed. No PPO updates ran.

Speeds were 0.317506, 0.396974 and 0.470187 m/s respectively. All probes completed
five seconds without falls or joint-limit violations. The two faster probes beat
the original PPO-100 mean speed in this single scenario. Population stability,
motion quality and promotion remain unproven. Foot-slip diagnostics increased.

`summary.json` preserves full metrics. `evidence.zip` contains the prepared
checkpoints/configuration/source snapshots, three measured model/state/trace
recordings, evaluator source and comparison launch command. `manifest.json`
hashes the ZIP. Actor tensors and model XML were checked equal against the saved
PPO-100 parent before archiving. The four-pane replay uses the original accepted
PPO-100 at top left, 1.1 Hz top right, 1.35 Hz bottom left, and 1.5 Hz bottom right.
The refreshed viewer rendered with readable headings and no stderr output.

This is an experiment record, not a new robot capability checkpoint.
