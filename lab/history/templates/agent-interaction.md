# Agent Interaction: <short title>

```yaml
id: AI-YYYYMMDD-NNN
date: YYYY-MM-DD
sources:
  - kind: coding-agent # coding-agent | chat | human-observation | external-reference
    system: Codex
    reference: # URL, issue/comment, session label, or blank when unavailable
status: open # open | acted | resolved | abandoned
evaluation: unconfirmed # unconfirmed | confirmed
repo_state:
  repository: robotics-test-bench
  branch:
  commit:
  changed_files: []
related:
  experiment:
  hypotheses: []
  experiments: []
  claims: []
  decisions: []
  issues: []
  files: []
objects:
  question: Q-YYYYMMDD-NNN
  response: R-YYYYMMDD-NNN
  evaluation: E-YYYYMMDD-NNN
  action: A-YYYYMMDD-NNN
  outcome: O-YYYYMMDD-NNN
librarian:
  status: pending # pending | synced | not-needed
  record_ids: []
```

`NNN` is repository-wide for that date, not experiment-local. Allocate the next unused suffix across all experiment logs. Never reuse an `AI/Q/R/E/A/O` ID in another experiment. Preserve an allocated ID across later updates. Keep interaction entries in causal order so prediction -> observation -> interpretation can be reconstructed from the file as well as Git history.

Every new entry records provenance in `sources`. Keep distinct sources distinct. A chat explanation, a coding-agent inspection, a human observation, and an external reference can support the same belief change, but they are not interchangeable evidence. Use multiple `sources` items when required rather than collapsing them into one unattributed response. Existing entries that use the legacy `agent:` field remain valid.

## Q — Question

State the decision-relevant question.

### Human prediction

Record the user's actual initial attempt when it matters to this experiment.
A conversational prediction, code attempt, explanation, or diagnosis counts.
This log documents evidence after the interaction; it is not a form the user
must fill out before receiving help.

### Purpose

Explain why the question matters to the experiment or implementation.

### Context supplied

List only the files, constraints, observations, data, or prior claims that shaped the interaction.

## R — Response summary

Summarize the useful claims, options, warnings, or proposed tests. Do not paste the full response by default.

## E — Human evaluation

### Accepted

- Useful or correct item accepted by the human.

### Rejected

- Item rejected and why.

### Unresolved

- Item that still needs evidence or a decision.

### Verification required

- Code, data, documentation, inspection, or experiment required before treating the response as evidence.

## A — Action

Record the test, inspection, decision, code change, or commit caused by the interaction. Link repository-relative files and other evidence.

If the action changes code, use the commit message as the translation layer between reasoning and implementation: state why the current belief, evidence, or decision justified the change. Leave exact implementation detail to the diff.

## O — Outcome

Record the observed result, or write `Pending` until it is known.

### Effect on current belief

- Before: State the prior working model or uncertainty.
- After: State the repaired, weakened, strengthened, rejected, or superseded model.
- Evidence status: State whether the change is a human interpretation, agent explanation, code inspection, external reference, or experimentally observed result. Provenance does not by itself establish truth.

For conversation-derived interactions, this before -> after delta is the primary artifact. Preserve the change in the mental model, not the transcript.

## Librarian update

Pass this compact payload during the next Librarian invocation. Preserve the stable IDs and source provenance when updating the entry.

```yaml
source:
  repository: robotics-test-bench
  path: experiments/<date-question>/agent-log.md
  commit:
provenance:
  - kind: coding-agent
    system: Codex
    reference:
objects:
  - id: Q-YYYYMMDD-NNN
    type: Question
    summary:
    status:
  - id: R-YYYYMMDD-NNN
    type: Response
    summary:
    status:
  - id: E-YYYYMMDD-NNN
    type: Evaluation
    summary:
    status:
  - id: A-YYYYMMDD-NNN
    type: Action
    summary:
    status:
  - id: O-YYYYMMDD-NNN
    type: Outcome
    summary:
    status:
relations:
  - subject: Q-YYYYMMDD-NNN
    predicate: receives
    object: R-YYYYMMDD-NNN
  - subject: R-YYYYMMDD-NNN
    predicate: receives
    object: E-YYYYMMDD-NNN
  - subject: E-YYYYMMDD-NNN
    predicate: causes
    object: A-YYYYMMDD-NNN
  - subject: A-YYYYMMDD-NNN
    predicate: produces
    object: O-YYYYMMDD-NNN
unresolved_questions: []
superseded_claims: []
```

Do not include secrets, hidden reasoning, full chat transcripts, or unsupported conclusions presented as verified facts.
