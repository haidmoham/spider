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
- Before landing a commit that changes public identity, checkpoint state, capability claims, or test-bench provenance, use the installed `commit-boundary` skill with `.ontology/commit-rules.md`.

## Working agreements

- Keep changes small and easy to review.
- Read relevant files before editing them.
- Prefer clear, direct documentation.
- Do not assume formal physics coursework when the mechanism is part of the learning target.
- Explain standing and locomotion from the physical load path before controller math. Name gravity and contact forces. State the torque they create about relevant joints. State the force or torque that must oppose them. Then introduce Jacobians, compensation terms, or controller equations.
- Preserve the standing foundation. Robotics-test-bench issue #24 defines the support-state experiment that must precede `C-1N // 02 · STAND`.
- `STAND` requires support-aware, reproducible rollout evidence. A plausible initial pose or a single attractive run is not sufficient.
- After `STAND`, use C-1N primarily as a simulation subject for learned locomotion, evaluation, simulator calibration, uncertainty, differentiable dynamics, and simulation scaling.
- Treat controls, contact mechanics, actuator limits, estimation, and numerical methods as supporting mechanisms. Pull them in when a concrete simulation failure requires them.
- Do not route the project toward ROS 2, embedded systems, or physical hardware by default.
- Hardware is not a graduation requirement. Physical-system realism matters because simulator assumptions must remain interpretable and falsifiable.
- Prefer population-level evaluation across fixed scenarios, seeds, or parameter draws when comparing policies or simulator changes.
- Preserve objective terms, policy checkpoints, evaluation scenarios, and model parameters when learned behavior or statistical comparison is the evidence target.
- Keep browser and WASM work downstream of useful simulation behavior. The web surface should expose evidence, not create the learning target.
- Verify important changes with a focused check.
- Do not commit secrets or generated credentials.
