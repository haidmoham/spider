# Notebook refactor verification

Checked on 2026-09-04 against pre-refactor commit `35514be`.
This is software compatibility evidence. It does not establish a new robotics
capability or human understanding. Related work: [C-1N #17](https://github.com/haidmoham/spider/issues/17)
and [test-bench #25](https://github.com/haidmoham/robotics-test-bench/issues/25).

## Environment and checks

The before and after checks used the same Python 3.12 environment with
MuJoCo 3.11.0, NumPy 2.2.4, and Pillow 12.2.0. The default system Python had
NumPy 2.5.0, so it was not used for the compatibility comparison.

All 17 tests passed:

```powershell
.venv/Scripts/python -m unittest -v test_simulation.py test_visual_invariants.py test_learning_env.py
```

Exact before/after equality held for final state and every saved trace array,
including metadata, in these cases:

- Neutral targets: 1 second.
- STAND: 10 seconds.
- SHUFFLE: 2 seconds.
- STAND with a 0.25 mg world +X shove: 1 second.
- STAND with a 1 mg, 90-degree shove: 1 second.

The shove duration was 0.2 seconds. Eight command responses also matched for
state, stepping, perturbation, pulse expiry, reset, power adjustment, and run.
Thirty additional per-step state/control snapshots matched exactly.
The comparison snapshots were temporary validation artifacts, not new
checkpoint evidence. Model and controller source behavior was unchanged.

The independent review checked extraction ASTs, compatibility imports, module
dependencies, and adapter validation. It found no introduced regression.
Two existing descriptions were corrected: the shove suite covers eight
directions, and CLI help now includes the existing 0.75 mg treatment.

## Notebook validation boundary

The new notebook passed nbformat validation and fresh-kernel Run All from both
the repository root and the notebooks directory. No learning experiment or
plot ran. The prediction-gated branch remains for the human's first attempt.
No stored output claims that the exercise has been completed.

Existing simulation and visual-invariant tests passed. A live interactive
viewer was not opened. Its extracted logic was reviewed for compatibility.
