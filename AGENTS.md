# Repository instructions

## Scope

These instructions apply to the entire repository.

## Checkpoints

- Use checkpoints only for robotics capabilities or understood failures worth preserving and comparing.
- Do not create checkpoints for instrumentation, cleanup, presentation polish, or elapsed time.
- Do not claim a capability before reproducible evidence supports it.
- Before landing a commit that changes checkpoint state, capability claims, or test-bench provenance, use the installed `commit-boundary` skill with `.ontology/commit-rules.md`.

## Working agreements

- Keep changes small and easy to review.
- Read relevant files before editing them.
- Prefer clear, direct documentation.
- Do not assume formal physics coursework when the mechanism is part of the learning target.
- Explain standing and locomotion from the physical load path before controller math. Name gravity and contact forces. State the torque they create about relevant joints. State the force or torque that must oppose them. Then introduce Jacobians, compensation terms, or controller equations.
- Prefer eagerness to teach over eagerness to solve. Use Jupyter, telemetry, and diagrams to make the physical question observable. Let the user form and inspect their own prediction before running or interpreting a non-trivial analysis, unless they explicitly ask for the answer.
- Do not use goals, issue state, or capability checkpoints to rush the user. A C-1N capability is incomplete if its physical causal chain is not understood end to end.
- Separate analysis setup from analysis execution. Automate the setup, but do not execute or interpret a learning-target calculation on the user's behalf without explicit permission.
- Preserve the standing foundation. Robotics-test-bench issue #24 defines the support-state experiment that must precede a stable-standing capability claim.
- Stable standing requires support-aware, reproducible rollout evidence. A plausible initial pose or a single attractive run is not sufficient.
- After stable standing, use C-1N primarily as a simulation subject for learned locomotion, evaluation, simulator calibration, uncertainty, differentiable dynamics, and simulation scaling.
- Treat controls, contact mechanics, actuator limits, estimation, and numerical methods as supporting mechanisms. Pull them in when a concrete simulation failure requires them.
- C++ is in scope when it improves simulator performance, robotics software, autonomy integration, numerical code, or compatibility with target libraries. Do not use C++ as a reason to redirect the project into embedded or firmware work.
- ROS 2 is in scope when simulation or autonomy interoperability needs it. It is not a required destination.
- Do not route the project toward firmware, device drivers, microcontrollers, board-level electronics, or other EE/CPE-style low-level work by default.
- Hardware is not a graduation requirement. Physical-system realism matters because simulator assumptions must remain interpretable and falsifiable.
- Prefer population-level evaluation across fixed scenarios, seeds, or parameter draws when comparing policies or simulator changes.
- Preserve objective terms, policy checkpoints, evaluation scenarios, and model parameters when learned behavior or statistical comparison is the evidence target.
- Keep browser and WASM work downstream of useful simulation behavior. The web surface should expose evidence, not create the learning target.
- Verify important changes with a focused check.
- Do not commit secrets or generated credentials.
