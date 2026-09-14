# Notebooks

- [02_first_policy.ipynb](02_first_policy.ipynb): current policy session. Run setup, then edit the marked policy cell.
- [01_control_step.ipynb](01_control_step.ipynb): earlier control-step experiment. Preserve the user's treatment and outputs.
- [stand_rollout_diagnostics.ipynb](stand_rollout_diagnostics.ipynb): reads saved STAND traces.

See [LEARNING.md](../LEARNING.md) for environment setup and the learning route.
Reusable helpers live in `c1n/`; experiment choices and interpretation live here.
Run All can execute physics and open viewers. Preparation checks must not run
learning experiments without the user's instruction.

Executed notebooks and traces are evidence. The 2026-09-10 diagnostics archive is
`telemetry/local-archive/2026-09-10-stand/`; its manifest covers the saved notebook
and 34 traces. The pre-trim 2026-09-14 notebooks are in `telemetry/architecture-before/`.
Both archives are local and ignored by Git. Existing trace paths are preserved.
