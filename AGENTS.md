# Repository instructions

## Scope

These instructions apply to the entire repository.

## Experiment queue

`TODO.md` is the authoritative ordered queue for Spider experiments.

- Treat the first experiment with `Status: NEXT` as the next experiment.
- Exactly one experiment should have `Status: NEXT`.
- Do not skip, reorder, or replace queued experiments unless the user explicitly changes priority.
- Preserve completed experiments with `Status: DONE` and a short result note or commit reference.

## Working agreements

- Keep changes small and easy to review.
- Read relevant files before editing them.
- Prefer clear, direct documentation.
- Verify important changes with a focused check.
- Do not commit secrets or generated credentials.
