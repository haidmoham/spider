# walk_fast_300

User-requested checkpoint after 300 total PPO updates, continued from
`walk_fast_200`. Mean speed: 0.758568 m/s; sampled average: 0.697075 m/s;
zero falls in 24 five-second evaluations, zero joint-limit violations.

See [manifest.json](manifest.json) and the
[archived experiment](../walk_fast_200/experiments/policy-cadence-continuation-20260915/README.md).
Contact fragmentation and slip remain review items. This name does not claim
completion of the deliberate-stride goal or terrain/push robustness.

The next authorized continuation stops at 500 total updates. It moves PPO
optimization to CUDA while preserving CPU MuJoCo rollouts and the objective,
chassis, cadence bounds, weights, optimizer history and seed progression.
