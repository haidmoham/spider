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
- Preserve the existing STAND baseline and its limits. Issue #24 remains a support-understanding question; do not infer human understanding from the robot checkpoint. The current learning route is in `LEARNING.md`.
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

## Notebook learning contract

- Use `notebooks/01_control_step.ipynb` as the learning entry point. Keep reusable simulation behavior in Python modules.
- Keep notebook setup separate from learning calculations. Default Run All must not execute prediction-gated experiments or reveal their answers.
- The human writes RL and PPO with PyTorch operations, autograd, and optimizers. Do not prewrite algorithm solutions or substitute a ready-made trainer.
- The agent owns setup and verification. The human owns hypotheses, observation and reward design, algorithm implementation, and interpretation.
- Rebuild mathematics as needed. Use trace -> predict -> change -> diagnose as the readiness check for regular pairing.
