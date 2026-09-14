# C-1N standing requirements ledger

This is the versioned cross-repository ledger for `C-1N // 02 · STAND`.
Each source repository retains its own evidence and implementation details.
Update this ledger only when a requirement's status, evidence boundary, or next
integration gate changes.

| ID | Requirement | Canonical source | Status | Evidence or next gate |
| --- | --- | --- | --- | --- |
| STAND-01 | Make fixed-foot support measurable through contact geometry, COM projection, support load, and torso attitude. | `robotics-test-bench` #24 | completed | Fixed-foot fixture brackets margin exhaustion and rear unloading between `+0.96 m` and `+0.98 m` payload shift. |
| STAND-02 | Establish whether one C-1N leg can reach the required outward-and-down support state. | `robotics-test-bench` #31 | completed | The current two-DOF leg leaves a `0.067 m` residual; the experimental orthogonal proximal hinge reaches the same target to solver tolerance. |
| STAND-03 | Integrate a gait-disabled, support-aware C-1N stance. | `haidmoham/spider` README and `simulation.py`, `standing.py`, `telemetry.py` | completed | Recorded six-contact STAND baseline includes shared-frame support geometry, COM projection, contact loads, and torso attitude. |
| STAND-04 | Preserve a prediction-led physics repair analysis for a concrete standing failure. | `haidmoham/spider` #12 | historical learning requirement; not a locomotion gate | Preserve the diagnostics notebook and prior records. No new human prediction, interpretation, or demonstrated understanding is claimed by this reconciliation. |
| STAND-05 | Demonstrate reproducible C-1N stable support under a fixed evaluation and declared perturbation. | `haidmoham/spider` README; baseline commit `0e0cdb0` | superseded as an acceptance gate | The accepted checkpoint is the deterministic 10-second six-contact baseline. The declared 1 mg shove remains a failed recovery case. Recovery is excluded, not completed. |

## Accepted boundary — 2026-09-10

The user confirmed: STAND is earned as recorded; next comes a rudimentary
walking policy from a clean base in practice mode. The canonical
[C-1N README](https://github.com/haidmoham/spider/blob/master/README.md) records
the baseline and failure limits. This ledger now agrees with that boundary and
`TODO.md`. Preserve the original row IDs and the stronger historical
perturbation requirement above. Do not reinterpret a failed recovery as a pass.

Issue #24 was archived as not planned during the 2026-09-10 issue cleanup.
Its fixture evidence is preserved. Neither administrative closure nor the
robot checkpoint proves the user's present understanding. It is not a
standing prerequisite before the current walking-policy work.

## Update rule

- `active`: work can proceed now.
- `blocked by`: an explicit prerequisite has not met its evidence boundary.
- Do not mark a row complete from a plausible pose, code scaffold, or one attractive rollout.
- Add new rows only for durable capability or evidence requirements. Keep transient tasks in the owning issue or experiment log.
