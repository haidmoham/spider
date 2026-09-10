# Notebook entry points

Start with [01_control_step.ipynb](01_control_step.ipynb).
Continue with [02_first_policy.ipynb](02_first_policy.ipynb) for the current
C-1N policy lesson. It provides setup and a function stub. The user writes the
body one small step at a time with conversational review.
Use [the learning guide](../LEARNING.md) to prepare the environment.
The cells run normally. Run All executes the control-step experiment and plots.
The agent gets your first attempt in conversation before advancing the learning
step. There is no prediction form or participation switch. Preparation checks
validate notebook structure and syntax without executing learning experiments.

[stand_rollout_diagnostics.ipynb](stand_rollout_diagnostics.ipynb) reads existing
STAND traces. Use it after the first control-step exercise. It is a diagnostic
tool, not a record of the human's understanding.

Local executed diagnostics and traces are evidence, not disposable notebook
noise. During the 2026-09-10 cleanup, the existing executed diagnostics notebook
was archived byte-for-byte under `telemetry/local-archive/2026-09-10-stand/`.
Its `manifest.json` records hashes for the notebook and 34 existing traces.
Those traces remain at their original `artifacts/c1n_redesign/` paths. These
local files are ignored by Git. No rollout was rerun or interpreted for this
archive. The tracked notebook preserves the user's source cells without stored
execution output. Reopen the archived copy to inspect the saved plots.

Keep reusable physics in Python modules. Keep predictions, experiment
parameters, plots, and interpretation in notebooks. Move reusable code from a
learning notebook into a module only after its behavior is understood.
