# Commit ontology contract

Use the installed `commit-boundary` skill before landing commits that change checkpoint state, claimed capability, or robotics-test-bench integration provenance.

## Blocking invariants

- Treat checkpoint state in `AGENTS.md` as canonical repository state.
- Do not advance a checkpoint for instrumentation, cleanup, presentation work, elapsed time, or an understood-but-unimplemented idea.
- A checkpoint advance must point to a demonstrated robotics capability or understood failure worth preserving and comparing.
- Do not describe an unverified capability as completed before its evidence lands.
- Robotics-test-bench instrumentation may integrate without creating a checkpoint when the current repository instructions allow it.
- A commit that changes a public capability claim must keep README wording and implementation evidence consistent.
- Do not invent simulator evidence or stable behavior to satisfy a checkpoint boundary.

## Cross-repository provenance

When a capability or instrumentation change derives from robotics-test-bench work, preserve the source issue, experiment, or commit reference when available. Do not convert source issue numbers into checkpoint chronology.
