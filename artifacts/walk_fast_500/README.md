# walk_fast_500

User-requested checkpoint after 500 total PPO updates, continued from
`walk_fast_300`. Mean speed: 0.887987 m/s; sampled average: 0.821139 m/s;
zero falls in 24 five-second evaluations and zero joint-limit violations.

PPO optimization for updates 301 through 500 ran on an NVIDIA GeForce RTX 3080
with the PyTorch CUDA 12.6 build. MuJoCo physics, rollout inference, and the
evaluation protocol remained on CPU. The objective, chassis, cadence bounds,
optimizer history, and seed progression remained unchanged.

See [manifest.json](manifest.json) and the
[archived experiment](experiments/cuda-continuation-20260915/README.md).
The user authorized the name `walk_fast_500` and said the goal could close when
the 500 run completed. This closes the requested training boundary. It does not
establish deliberate stride quality, terrain or push robustness. The n=500
deterministic seed-201 recording has a 0.204586 m/s stance-foot speed RMS slip
proxy, so gait quality remains an explicit review item.
