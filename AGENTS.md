# Repository instructions

## Scope

These instructions apply to the entire repository.

## Robot identity

- The public robot identity is **C-1N**. Do not introduce new public references that call the robot Spider.
- `haidmoham/spider`, `model/spider.xml`, and other existing `spider` implementation identifiers are legacy compatibility names, not the public identity.
- Public checkpoints use `C-1N // NN · CODENAME`.
- Increment the checkpoint only for a robotics capability or understood failure worth preserving and comparing. Do not increment for instrumentation, cleanup, presentation polish, or elapsed time.
- `C-1N // 00 · POSE` is the historical motor-assisted static-pose baseline. Do not describe it as a demonstrated standing capability.
- The current public checkpoint is `C-1N // 01 · SHUFFLE`.
- `C-1N // 02 · STAND` and `C-1N // 03 · STRIDE` are reserved future boundaries. Do not present them as completed before the corresponding evidence lands.
- Task-space instrumentation from robotics-test-bench issue #6 may integrate without creating a public checkpoint. `FRAME` is not a reserved release name.
- Before landing a commit that changes public identity, checkpoint state, capability claims, or test-bench provenance, use the installed `commit-boundary` skill with `.ontology/commit-rules.md`.

## Working agreements

- Keep changes small and easy to review.
- Read relevant files before editing them.
- Prefer clear, direct documentation.
- Do not assume formal physics coursework when the mechanism is part of the learning target.
- Explain standing and locomotion from the physical load path before controller math. Name gravity and contact forces. State the torque they create about relevant joints. State the torque or force that must oppose them. Then introduce Jacobians, compensation terms, or controller equations.
- Verify important changes with a focused check.
- Do not commit secrets or generated credentials.
